"""
Resolver un terreno con un modelo entrenado: orden de viajes del agente y del greedy, con sus costos.
Lo usa la API, pero no depende de ella, así que se puede probar sola.
"""
import threading
from typing import Callable

import numpy as np

from src.agent import seleccionar_accion
from src.baseline import elegir_accion_greedy
from src.env import EntradaPool, TerrainLevelingEnv
from src.terrain_generator import generar_terreno


class TerrenoNoResoluble(ValueError):
    """El terreno de la semilla pedida necesita más viajes de los que soporta el modelo."""


# El entorno guarda el estado del episodio, así que no se puede usar desde dos hilos a la vez.
CANDADO_ENTORNO = threading.Lock()


def _recorrer(env: TerrainLevelingEnv, entrada: EntradaPool, elegir: Callable) -> dict:
    """Corre un episodio completo y devuelve el costo y el detalle de cada viaje en el orden elegido."""
    obs, info = env.reset(options={"entrada": entrada})
    pasos = []
    terminado = False
    while not terminado:
        accion = elegir(env, obs, info)
        viaje = env.viajes[accion]
        obs, _, terminado, _, info = env.step(accion)
        pasos.append(
            {
                "paso": len(pasos) + 1,
                "viaje": int(accion),
                "origen": list(viaje.origen),
                "destino": list(viaje.destino),
                "volumen": int(viaje.volumen),
                "distancia": int(info["distancia_paso"]),
            }
        )
    return {"costo": float(env.costo_acumulado), "orden": pasos}


def describir_terreno(env: TerrainLevelingEnv, seed: int) -> dict:
    """
    Terreno de la semilla (alturas y objetivo) sin resolverlo, para poder verlo antes de pedir la solución.
    `resoluble` es False si necesita más viajes que el máximo del modelo, en ese caso `n_viajes` es None.
    """
    alturas, objetivo = generar_terreno(env.tamano, seed)
    entrada = env.entrada_de_semilla(seed)
    return {
        "seed": seed,
        "terreno": {
            "tamano": env.tamano,
            "alturas": alturas.astype(int).tolist(),
            "objetivo": objetivo.astype(int).tolist(),
        },
        "n_viajes": None if entrada is None else len(entrada.viajes),
        "resoluble": entrada is not None,
    }


def resolver_terreno(env: TerrainLevelingEnv, red, seed: int) -> dict:
    """
    Resuelve el terreno de la semilla con el agente (sin exploración) y con el greedy.
    Lanza TerrenoNoResoluble si el terreno excede el máximo de viajes del modelo.
    """
    entrada = env.entrada_de_semilla(seed)
    if entrada is None:
        raise TerrenoNoResoluble(
            f"El terreno de la semilla {seed} necesita más de {env.max_viajes} viajes, "
            "que es el máximo con el que se entrenó el modelo."
        )
    rng = np.random.default_rng(0)

    def con_el_agente(e, obs, info):
        return seleccionar_accion(red, obs, info["mascara_acciones"], 0.0, rng)

    with CANDADO_ENTORNO:
        agente = _recorrer(env, entrada, con_el_agente)
        greedy = _recorrer(env, entrada, lambda e, obs, info: elegir_accion_greedy(e))

    mejora = 100.0 * (greedy["costo"] - agente["costo"]) / greedy["costo"] if greedy["costo"] else 0.0
    return {
        "seed": seed,
        "terreno": describir_terreno(env, seed)["terreno"],
        "n_viajes": len(entrada.viajes),
        "agente": agente,
        "greedy": greedy,
        "mejora_vs_greedy_pct": mejora,
    }