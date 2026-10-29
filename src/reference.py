"""
Referencia de calidad: el mejor orden de viajes que se puede encontrar por búsqueda local(puede pero no es el óptimo).
"""

import numpy as np

from src.trip_builder import Viaje
from src.utils import distancia_manhattan

POSICION_INICIAL = (0, 0)


def _validar_permutacion(orden: list[int], n: int) -> None:
    if sorted(orden) != list(range(n)):
        raise ValueError(f"El orden debe ser una permutación de 0..{n - 1}: {orden}")


def costo_de_orden(
    orden: list[int], viajes: list[Viaje], inicio: tuple[int, int] = POSICION_INICIAL
) -> int:
    """Distancia total de hacer los viajes en ese orden: Ir al origen y llevar la carga al destino."""
    _validar_permutacion(orden, len(viajes))
    posicion, total = inicio, 0
    for i in orden:
        viaje = viajes[i]
        total += distancia_manhattan(posicion, viaje.origen)
        total += distancia_manhattan(viaje.origen, viaje.destino)
        posicion = viaje.destino
    return total


def orden_greedy(viajes: list[Viaje], inicio: tuple[int, int] = POSICION_INICIAL) -> list[int]:
    """Vecino más cercano, en empateelige menor índice."""
    pendientes = list(range(len(viajes)))
    posicion, orden = inicio, []
    while pendientes:
        # Elegir el viaje pendiente cuyo origen esté más cerca de la posición actual
        mejor = min(pendientes, key=lambda i: distancia_manhattan(posicion, viajes[i].origen))
        orden.append(mejor)
        pendientes.remove(mejor)
        posicion = viajes[mejor].destino
    return orden


def _busqueda_local(
    orden: list[int], viajes: list[Viaje], inicio: tuple[int, int]
) -> tuple[list[int], int]:
    """Mejora un orden con 2-opt (invertir un tramo) y reubicación de un viaje hasta no poder más."""
    mejor, costo = orden[:], costo_de_orden(orden, viajes, inicio)
    mejoro = True
    while mejoro:
        mejoro = False
        n = len(mejor)
        for i in range(n):
            for j in range(i + 1, n):
                #Invertir el tramo entre i y j
                candidato = mejor[:i] + mejor[i : j + 1][::-1] + mejor[j + 1 :]
                c = costo_de_orden(candidato, viajes, inicio)
                if c < costo:
                    mejor, costo, mejoro = candidato, c, True
            for j in range(n):
                # Reubicar el viaje en la posición i a la posición j
                if j == i:
                    continue
                candidato = mejor[:]
                candidato.insert(j, candidato.pop(i))
                c = costo_de_orden(candidato, viajes, inicio)
                if c < costo:
                    mejor, costo, mejoro = candidato, c, True
    return mejor, costo


def mejor_orden_conocido(
    viajes: list[Viaje],
    reinicios: int = 8,
    seed: int = 0,
    inicio: tuple[int, int] = POSICION_INICIAL,
) -> tuple[list[int], int]:
    """
    Hace una búsqueda local partiendo del greedy y usa reinicios
    partiendo de órdenes aleatorios para no quedarse en mínimos locales.
    """
    if not viajes:
        return [], 0
    rng = np.random.default_rng(seed)
    # Greedy + búsqueda local
    mejor, costo = _busqueda_local(orden_greedy(viajes, inicio), viajes, inicio)
    # Buscar mejores órdenes partiendo de lugares aleatorios
    for _ in range(reinicios):
        inicial = [int(x) for x in rng.permutation(len(viajes))]
        candidato, c = _busqueda_local(inicial, viajes, inicio)
        if c < costo:
            mejor, costo = candidato, c
    return mejor, costo


def brecha_cerrada(costo_agente: float, costo_greedy: float, costo_referencia: float) -> float:
    """
    Fracción de la brecha entre el greedy y la referencia que cierra el agente (con costos medios).
    0 = igual al greedy, 1 = igual a la referencia, negativo = peor que el greedy.
    """
    brecha = costo_greedy - costo_referencia
    if brecha <= 0:
        raise ValueError("El greedy ya iguala a la referencia: no hay brecha que cerrar.")
    return (costo_greedy - costo_agente) / brecha