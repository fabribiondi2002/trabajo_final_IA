"""
Compara varios experimentos (cada uno es una carpeta con historial_seed_N.json y modelo_seed_N.pt,
como las que genera src.experiment) y arma un gráfico con:
  1. la curva de validación de cada variante (media ± desvío entre seeds) y la línea del greedy,
  2. el costo en test de cada variante (una marca por seed) con las líneas del greedy y la referencia,
  3. cuántos episodios entrenó cada variante.
Se corre con:
  python -m src.comparison "base=models/base" "por viaje=models/por_viaje" \
      "vacío + paciencia=models/por_viaje_vacio" --salida graficos/comparacion.png
"""
import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.baseline import evaluar_baseline
from src.env import TerrainLevelingEnv
from src.evaluation import costos_referencia
from src.train import cargar_historial, cargar_modelo, evaluar_red

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experimento:
    nombre: str
    historiales: list[dict]
    costos_test: list[float]   


def parsear_variante(texto: str) -> tuple[str, str]:
    """'nombre=carpeta' -> (nombre, carpeta)."""
    nombre, separador, carpeta = texto.partition("=")
    if not separador or not nombre.strip() or not carpeta.strip():
        raise ValueError(f"Formato inválido: '{texto}'. Usar nombre=carpeta.")
    return nombre.strip(), carpeta.strip()


def _numero_de_seed(ruta: Path) -> int:
    return int(re.search(r"seed_(\d+)", ruta.name).group(1))


def archivos_por_seed(carpeta: str, patron: str) -> list[Path]:
    """Archivos de la carpeta ordenados por número de seed (seed 2 antes que seed 10)."""
    return sorted(Path(carpeta).glob(patron), key=_numero_de_seed)


def episodios_entrenados(historial: dict) -> int:
    """Episodios que realmente se entrenaron."""
    if "episodios_entrenados" in historial:
        return int(historial["episodios_entrenados"][0])
    return len(historial["recompensa"])


def curva_validacion(historiales: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Media y desvío del costo de validación entre seeds, por episodio de evaluación.
    Si una seed se detuvo antes (parada anticipada), en los episodios siguientes
    se promedian solo las seeds que siguen entrenando.
    """
    episodios = sorted({ep for h in historiales for ep, _ in h["validacion"]})
    tabla = np.full((len(historiales), len(episodios)), np.nan)
    for i, h in enumerate(historiales):
        por_episodio = dict(h["validacion"])
        for j, ep in enumerate(episodios):
            if ep in por_episodio:
                tabla[i, j] = por_episodio[ep]
    cantidad = np.sum(~np.isnan(tabla), axis=0)
    media = np.nansum(tabla, axis=0) / cantidad
    varianza = np.nansum((tabla - media) ** 2, axis=0) / np.maximum(cantidad - 1, 1)   # desvío muestral (ddof=1)
    desvio = np.where(cantidad > 1, np.sqrt(varianza), 0.0)
    return np.asarray(episodios), media, desvio


def cargar_experimento(nombre: str, carpeta: str, env: TerrainLevelingEnv) -> Experimento:
    """Lee los historiales y mide en test cada modelo guardado de la carpeta."""
    rutas_hist = archivos_por_seed(carpeta, "historial_seed_*.json")
    rutas_modelo = archivos_por_seed(carpeta, "modelo_seed_*.pt")
    if not rutas_hist or not rutas_modelo:
        raise FileNotFoundError(f"No hay historiales y modelos en '{carpeta}'.")
    historiales = [cargar_historial(str(r))[0] for r in rutas_hist]
    costos = []
    for ruta in rutas_modelo:
        red, _ = cargar_modelo(str(ruta))
        costos.append(float(np.mean(evaluar_red(env, red, env.pool_test))))
    return Experimento(nombre, historiales, costos)


def graficar(
    experimentos: list[Experimento],
    greedy_validacion: float,
    greedy_test: float,
    referencia_test: float | None,
    salida: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colores = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax_val, ax_test, ax_ep) = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.5, 1.2, 0.8]})

    # 1. curvas de validación
    for i, exp in enumerate(experimentos):
        x, media, desvio = curva_validacion(exp.historiales)
        ax_val.plot(x, media, marker="o", ms=3, color=colores[i % len(colores)], label=exp.nombre)
        ax_val.fill_between(x, media - desvio, media + desvio, color=colores[i % len(colores)], alpha=0.2)
    ax_val.axhline(greedy_validacion, color="black", ls="--", lw=1, label=f"greedy ({greedy_validacion:.2f})")
    ax_val.set(title="Costo en validación durante el entrenamiento", xlabel="episodio", ylabel="costo medio (menor es mejor)")
    ax_val.legend(fontsize=8)

    # 2. costo en test por seed
    for i, exp in enumerate(experimentos):
        media = float(np.mean(exp.costos_test))
        desvio = float(np.std(exp.costos_test, ddof=1)) if len(exp.costos_test) > 1 else 0.0
        color = colores[i % len(colores)]
        ax_test.errorbar(i, media, yerr=desvio, fmt="_", ms=28, mew=2, color=color, capsize=5, zorder=2)
        sacudida = np.linspace(-0.12, 0.12, len(exp.costos_test)) if len(exp.costos_test) > 1 else [0.0]
        ax_test.scatter(i + np.asarray(sacudida), exp.costos_test, color=color, edgecolor="black", zorder=3, s=22)
        ax_test.annotate(f"{media:.2f} ± {desvio:.2f}", (i, max(exp.costos_test)), xytext=(0, 9),
                         textcoords="offset points", ha="center", fontsize=8)
    ax_test.axhline(greedy_test, color="black", ls="--", lw=1, label=f"greedy ({greedy_test:.2f})")
    todos = [c for e in experimentos for c in e.costos_test] + [greedy_test]
    if referencia_test is not None:
        ax_test.axhline(referencia_test, color="green", ls=":", lw=1.5, label=f"referencia ({referencia_test:.2f})")
        todos.append(referencia_test)
    ax_test.set_ylim(min(todos) - 4, max(todos) + 8)
    ax_test.set_xlim(-0.6, len(experimentos) - 0.1)
    ax_test.set_xticks(range(len(experimentos)), [e.nombre for e in experimentos], rotation=20, ha="right", fontsize=8)
    ax_test.set(title="Costo en test (una marca por seed)", ylabel="costo medio (menor es mejor)")
    ax_test.legend(fontsize=8)

    # 3. episodios entrenados
    for i, exp in enumerate(experimentos):
        eps = [episodios_entrenados(h) for h in exp.historiales]
        ax_ep.bar(i, np.mean(eps), color=colores[i % len(colores)], alpha=0.5)
        ax_ep.annotate(f"{np.mean(eps):.0f}", (i, np.mean(eps)), ha="center", va="bottom", fontsize=8)
    ax_ep.set_xticks(range(len(experimentos)), [e.nombre for e in experimentos], rotation=20, ha="right", fontsize=8)
    ax_ep.set(title="Episodios entrenados (media)", ylabel="episodios")

    fig.tight_layout()
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def formatear_tabla(experimentos: list[Experimento], greedy_test: float) -> str:
    filas = ["variante                          costo_test (media ± desvío)   vs greedy   episodios"]
    for e in experimentos:
        media = float(np.mean(e.costos_test))
        desvio = float(np.std(e.costos_test, ddof=1)) if len(e.costos_test) > 1 else 0.0
        eps = np.mean([episodios_entrenados(h) for h in e.historiales])
        filas.append(f"{e.nombre:<32}  {media:>8.2f} ± {desvio:<6.2f}            {media - greedy_test:>+7.2f}   {eps:>8.0f}")
    return "\n".join(filas)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Compara experimentos (nombre=carpeta) en un gráfico.")
    parser.add_argument("variantes", nargs="+", help='cada una como "nombre=carpeta"')
    parser.add_argument("--salida", default="graficos/comparacion.png")
    parser.add_argument("--sin-referencia", action="store_true", help="no calcular la referencia (más rápido)")
    args = parser.parse_args()

    entorno = TerrainLevelingEnv()
    experimentos = [cargar_experimento(n, c, entorno) for n, c in map(parsear_variante, args.variantes)]
    greedy_val = float(np.mean(evaluar_baseline(entorno, entorno.pool_validacion)))
    greedy_test = float(np.mean(evaluar_baseline(entorno, entorno.pool_test)))
    referencia = None if args.sin_referencia else float(np.mean(costos_referencia(entorno.pool_test)))
    graficar(experimentos, greedy_val, greedy_test, referencia, args.salida)
    print(formatear_tabla(experimentos, greedy_test))
    logger.info("Gráfico guardado en %s", args.salida)