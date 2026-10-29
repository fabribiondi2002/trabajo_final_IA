"""
Referencia para comparar con el agente entrenado.
Funciona como el agente pero con una regla fija eligiendo el viaje mas cercano (greedy).
"""

from src.env import TerrainLevelingEnv, EntradaPool
from src.utils import distancia_manhattan

def elegir_accion_greedy(env: TerrainLevelingEnv) -> int:
    """
    Vecino más cercano: entre los viajes pendientes, elige el que tiene el
    origen más cerca de la máquina. Si hay empate, el de menor índice.
    """
    mejor_accion = None
    mejor_distancia = None
    for i, viaje in enumerate(env.viajes):
        if env.completados[i]:
            continue
        d = distancia_manhattan(env.posicion, viaje.origen)
        if mejor_distancia is None or d < mejor_distancia:
            mejor_accion, mejor_distancia = i, d
    if mejor_accion is None:
        raise ValueError("No quedan viajes pendientes.")
    return mejor_accion


def correr_baseline(env: TerrainLevelingEnv, entrada: EntradaPool) -> float:
    """Corre un episodio completo con la política greedy y devuelve el costo total."""
    env.reset(options={"entrada": entrada})
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(elegir_accion_greedy(env))
    return env.costo_acumulado


def evaluar_baseline(env: TerrainLevelingEnv, entradas: list[EntradaPool]) -> list[float]:
    """Costo del baseline en cada terreno de la lista."""
    return [correr_baseline(env, e) for e in entradas]