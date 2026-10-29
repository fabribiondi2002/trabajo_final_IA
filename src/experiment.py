"""
Entrena el mismo agente con varias seeds y resume los resultados (media y desvío).
Con una sola corrida no se sabe cuánto de un resultado es azar; con varias sí.
Se corre con:  python -m src.experimento --seeds 0 1 2 3 4
"""
import argparse
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from src.baseline import evaluar_baseline
from src.env import EntradaPool, TerrainLevelingEnv
from src.evaluation import ResultadoEvaluacion, costos_referencia, resumir
from src.train import ARQUITECTURAS, ConfigEntrenamiento, entrenar, evaluar_red, guardar_historial
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResultadoSeed:
    seed: int
    costo_validacion: float          
    resultado: ResultadoEvaluacion   


@dataclass(frozen=True)
class ResumenExperimento:
    seeds: tuple[int, ...]
    costo_medio: float
    costo_desvio: float
    brecha_media: float
    brecha_desvio: float
    costo_greedy: float
    costo_referencia: float


def estadisticas(valores: list[float]) -> tuple[float, float]:
    """Media y desvío estándar muestral."""
    if not valores:
        raise ValueError("No hay valores para resumir.")
    arr = np.asarray(valores, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def resumir_experimento(resultados: list[ResultadoSeed]) -> ResumenExperimento:
    """Resume los resultados de varias seeds. Greedy y referencia son los mismos para todas."""
    if not resultados:
        raise ValueError("No hay resultados para resumir.")
    costo_medio, costo_desvio = estadisticas([r.resultado.costo_agente for r in resultados])
    brecha_media, brecha_desvio = estadisticas([r.resultado.brecha_cerrada for r in resultados])
    return ResumenExperimento(
        seeds=tuple(r.seed for r in resultados),
        costo_medio=costo_medio,
        costo_desvio=costo_desvio,
        brecha_media=brecha_media,
        brecha_desvio=brecha_desvio,
        costo_greedy=resultados[0].resultado.costo_greedy,
        costo_referencia=resultados[0].resultado.costo_referencia,
    )


def correr_experimento(
    env: TerrainLevelingEnv,
    cfg_base: ConfigEntrenamiento,
    seeds: list[int],
    carpeta: str,
    entradas_test: list[EntradaPool] | None = None,
    reinicios: int = 8,
) -> list[ResultadoSeed]:
    """
    Para cada seed: entrena, guarda el mejor modelo y el historial en `carpeta`
    y mide el modelo guardado en los terrenos de test. Greedy y la referencia se calculan una sola vez.
    """
    if not seeds:
        raise ValueError("Hay que pasar al menos una seed.")
    entradas = entradas_test if entradas_test is not None else env.pool_test

    costos_greedy = evaluar_baseline(env, entradas)
    costos_ref = costos_referencia(entradas, reinicios)

    resultados = []
    for seed in seeds:
        cfg = replace(cfg_base, seed=seed, ruta_modelo=str(Path(carpeta) / f"modelo_seed_{seed}.pt"))
        logger.info("Entrenando seed %d...", seed)
        red, historial = entrenar(env, cfg)
        guardar_historial(historial, str(Path(carpeta) / f"historial_seed_{seed}.json"), cfg)
        resultado = resumir(evaluar_red(env, red, entradas), costos_greedy, costos_ref)
        resultados.append(ResultadoSeed(seed, historial["mejor_costo_validacion"][0], resultado))
        logger.info("seed %d: costo test %.2f, brecha cerrada %.1f%%", seed, resultado.costo_agente, 100 * resultado.brecha_cerrada)
    return resultados


def formatear_experimento(resultados: list[ResultadoSeed], resumen: ResumenExperimento) -> str:
    """Tabla de texto: una fila por seed y el resumen."""
    filas = ["seed   costo_validacion   costo_test   brecha_cerrada"]
    for r in resultados:
        filas.append(f"{r.seed:>4}   {r.costo_validacion:>15.2f}   {r.resultado.costo_agente:>10.2f}   {r.resultado.brecha_cerrada:>13.1%}")
    filas += [
        "",
        f"Agente (media ± desvío, {len(resultados)} seeds):  {resumen.costo_medio:.2f} ± {resumen.costo_desvio:.2f}",
        f"Brecha cerrada (media ± desvío):        {resumen.brecha_media:.1%} ± {resumen.brecha_desvio:.1%}",
        f"Greedy:                                 {resumen.costo_greedy:.2f}",
        f"Referencia:                             {resumen.costo_referencia:.2f}",
    ]
    return "\n".join(filas)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Entrena con varias seeds y resume los resultados en test.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--carpeta", default="models/base")
    parser.add_argument("--arquitectura", choices=sorted(ARQUITECTURAS), default="plana")
    parser.add_argument("--episodios", type=int, default=ConfigEntrenamiento().episodios)
    parser.add_argument("--oculta", type=int, default=ConfigEntrenamiento().oculta)
    parser.add_argument("--lr", type=float, default=ConfigEntrenamiento().lr)
    parser.add_argument("--evaluar-cada", type=int, default=ConfigEntrenamiento().evaluar_cada)
    parser.add_argument("--peso-distancia", type=float, default=1.0, help="escala del reward (parámetro del entorno)")
    args = parser.parse_args()

    entorno = TerrainLevelingEnv(peso_distancia=args.peso_distancia)
    cfg = ConfigEntrenamiento(
        arquitectura=args.arquitectura,
        episodios=args.episodios,
        episodios_decay=args.episodios // 2,
        oculta=args.oculta,
        lr=args.lr,
        evaluar_cada=args.evaluar_cada,
    )
    resultados = correr_experimento(entorno, cfg, args.seeds, args.carpeta)
    print(formatear_experimento(resultados, resumir_experimento(resultados)))