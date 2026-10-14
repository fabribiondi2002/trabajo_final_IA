"""
Módulo creador de viajes.

Recibe la lista de `Flujo` que devuelve el transportation problem
(src/allocation.py) y el parámetro `capacidad_maquina`, y parte cada flujo
en uno o más viajes de capacidad fija — simulando que la maquinaria real no
transporta el volumen completo de una zona alta en un único traslado.

número de viajes = ceil(volumen del flujo / capacidad de la máquina)

Esto NO es un componente de IA: es aritmética determinística, igual que
allocation.py. El agente RL recibe la lista de viajes ya armada como dato.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Viaje:
    """Un viaje individual: transporta como máximo 'capacidad_maquina'
    unidades de volumen de la celda origen a la celda destino."""
    origen: tuple[int, int]
    destino: tuple[int, int]
    volumen: int


def construir_viajes(flujos: list, capacidad_maquina: int) -> list[Viaje]:
    """
    Parte cada Flujo en uno o más Viajes de capacidad fija.

    Cada flujo genera ceil(flujo.volumen / capacidad_maquina) viajes. Todos
    los viajes de un mismo flujo transportan 'capacidad_maquina' unidades,
    salvo el último, que transporta el remanente (que puede ser menor a la
    capacidad completa).
    """
    if capacidad_maquina <= 0:
        raise ValueError(
            f"capacidad_maquina debe ser positiva, recibido: {capacidad_maquina}"
        )

    viajes: list[Viaje] = []

    for flujo in flujos:
        n_viajes = math.ceil(flujo.volumen / capacidad_maquina)
        volumen_restante = flujo.volumen

        for _ in range(n_viajes):
            volumen_este_viaje = min(capacidad_maquina, volumen_restante)
            viajes.append(
                Viaje(
                    origen=flujo.origen,
                    destino=flujo.destino,
                    volumen=volumen_este_viaje,
                )
            )
            volumen_restante -= volumen_este_viaje

    return viajes
