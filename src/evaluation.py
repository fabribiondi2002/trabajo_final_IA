"""
Evaluación del agente entrenado contra el baseline greedy y la referencia (mejor orden conocido),
sobre los mismos terrenos.
"""
import argparse
import logging
from dataclasses import dataclass

import numpy as np

from src.baseline import evaluar_baseline
from src.env import EntradaPool, TerrainLevelingEnv
from src.reference import brecha_cerrada, mejor_orden_conocido
from src.train import cargar_modelo, evaluar_red

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResultadoEvaluacion:
    n_terrenos: int
    costo_agente: float
    costo_greedy: float
    costo_referencia: float
    brecha_cerrada: float
    terrenos_mejores_que_greedy: int   # en cuántos terrenos el agente gana estrictamente al greedy
    terrenos_peores_que_greedy: int


def costos_referencia(entradas: list[EntradaPool], reinicios: int = 8) -> list[float]:
    """Costo del mejor orden conocido en cada terreno (es lento: una búsqueda local por reinicio)."""
    return [float(mejor_orden_conocido(list(e.viajes), reinicios=reinicios)[1]) for e in entradas]


def resumir(
    costos_agente: list[float], costos_greedy: list[float], costos_ref: list[float]
) -> ResultadoEvaluacion:
    """Convierte tres listas de costos (uno por terreno, mismo orden) en el resumen comparativo."""
    if not (len(costos_agente) == len(costos_greedy) == len(costos_ref)) or not costos_agente:
        raise ValueError("Las tres listas deben tener el mismo largo y no estar vacías.")
    agente, greedy, ref = (np.asarray(c, dtype=float) for c in (costos_agente, costos_greedy, costos_ref))
    return ResultadoEvaluacion(
        n_terrenos=len(agente),
        costo_agente=float(agente.mean()),
        costo_greedy=float(greedy.mean()),
        costo_referencia=float(ref.mean()),
        brecha_cerrada=brecha_cerrada(float(agente.mean()), float(greedy.mean()), float(ref.mean())),
        terrenos_mejores_que_greedy=int((agente < greedy).sum()),
        terrenos_peores_que_greedy=int((agente > greedy).sum()),
    )


def comparar(
    env: TerrainLevelingEnv, red, entradas: list[EntradaPool], reinicios: int = 8
) -> ResultadoEvaluacion:
    """Mide agente, greedy y referencia sobre los mismos terrenos."""
    return resumir(
        evaluar_red(env, red, entradas),
        evaluar_baseline(env, entradas),
        costos_referencia(entradas, reinicios),
    )


def evaluar_modelo(ruta: str, env: TerrainLevelingEnv, entradas: list[EntradaPool], reinicios: int = 8):
    """Carga un modelo guardado y lo compara. Falla si el modelo no corresponde a este entorno."""
    red, meta = cargar_modelo(ruta)
    if meta["obs_dim"] != env.observation_space.shape[0] or meta["n_acciones"] != env.action_space.n:
        raise ValueError(
            f"El modelo espera obs_dim={meta['obs_dim']} y n_acciones={meta['n_acciones']}, "
            f"pero el entorno tiene {env.observation_space.shape[0]} y {env.action_space.n}."
        )
    return comparar(env, red, entradas, reinicios)


def formatear(r: ResultadoEvaluacion) -> str:
    """Tabla de texto con el resultado."""
    return "\n".join(
        [
            f"Terrenos evaluados:       {r.n_terrenos}",
            f"Costo medio agente:       {r.costo_agente:8.2f}",
            f"Costo medio greedy:       {r.costo_greedy:8.2f}",
            f"Costo medio referencia:   {r.costo_referencia:8.2f}",
            f"Brecha cerrada:           {r.brecha_cerrada:8.1%}   (0% = igual a greedy, 100% = igual a la referencia)",
            f"Agente mejor que greedy:  {r.terrenos_mejores_que_greedy} terrenos",
            f"Agente peor que greedy:   {r.terrenos_peores_que_greedy} terrenos",
        ]
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evalúa un modelo guardado sobre los terrenos de test.")
    parser.add_argument("--modelo", default="models/dqn.pt")
    parser.add_argument("--reinicios", type=int, default=8)
    args = parser.parse_args()

    entorno = TerrainLevelingEnv()
    logger.info("Evaluando %s sobre %d terrenos de test...", args.modelo, len(entorno.pool_test))
    print(formatear(evaluar_modelo(args.modelo, entorno, entorno.pool_test, args.reinicios)))