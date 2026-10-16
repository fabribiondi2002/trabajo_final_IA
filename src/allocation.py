"""
Módulo de asignación de volúmenes (transportation problem / min-cost flow).

Resuelve, de forma determinística, cuánto volumen de tierra debe moverse
de cada zona "alta" (excedente) a cada zona "baja" (déficit), minimizando
el costo total (distancia * volumen). Esto NO es el componente de IA del
proyecto — es preprocesamiento que le da al agente RL el dato de qué zonas
altas rellenan qué zonas bajas (ver Project Charter, sección 1.1).

Requiere que el volumen total de excedente sea igual al volumen total de
déficit (supuesto de balance declarado en el Project Charter).
"""

from dataclasses import dataclass

import networkx as nx
import numpy as np
from src.utils import distancia_manhattan

@dataclass(frozen=True)
class Flujo:
    """Un flujo de tierra resultante de la asignación."""
    origen: tuple[int, int]   # coordenada (fila, col) de la zona alta
    destino: tuple[int, int]  # coordenada (fila, col) de la zona baja
    volumen: int


def calcular_excedentes_y_deficits(
    alturas: np.ndarray, objetivo: np.ndarray
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    """
    A partir de la grilla de alturas actuales y el perfil objetivo,
    calcula qué celdas son "altas" (excedente de tierra) y cuáles son
    "bajas" (déficit de tierra).

    Devuelve dos diccionarios: {celda: volumen} para excedentes y déficits.
    Los volúmenes se redondean a enteros porque min_cost_flow de networkx
    trabaja con capacidades enteras.
    """
    diferencia = np.round(alturas - objetivo).astype(int)

    excedentes = {}
    deficits = {}
    for fila in range(diferencia.shape[0]):
        for col in range(diferencia.shape[1]):
            valor = diferencia[fila, col]
            if valor > 0:
                excedentes[(fila, col)] = int(valor)
            elif valor < 0:
                deficits[(fila, col)] = int(-valor)

    return excedentes, deficits


def resolver_transportation_problem(
    alturas: np.ndarray, objetivo: np.ndarray
) -> list[Flujo]:
    """
    Resuelve el transportation problem completo: dado el terreno actual
    y el perfil objetivo, devuelve la lista de flujos (origen, destino,
    volumen) que minimizan el costo total de mover tierra.

    Lanza ValueError si el terreno no está balanceado (suma de excedentes
    != suma de déficits).
    """
    excedentes, deficits = calcular_excedentes_y_deficits(alturas, objetivo)

    total_excedente = sum(excedentes.values())
    total_deficit = sum(deficits.values())
    if total_excedente != total_deficit:
        raise ValueError(
            f"Terreno no balanceado: excedente={total_excedente}, "
            f"déficit={total_deficit}. El generador de terrenos debe "
            f"garantizar corte == relleno."
        )

    if total_excedente == 0:
        return []  # el terreno ya está nivelado

    grafo = nx.DiGraph()

    # Nodo de exceso: demand negativo = "produce" ese volumen
    for celda, volumen in excedentes.items():
        grafo.add_node(("alto", celda), demand=-volumen)

    # Nodo de déficit: demand positivo = "consume" ese volumen
    for celda, volumen in deficits.items():
        grafo.add_node(("bajo", celda), demand=volumen)

    # Arista entre cada alto y cada bajo, con costo = distancia y
    # capacidad suficiente para permitir cualquier reparto posible.
    capacidad_maxima = total_excedente
    for celda_alta, volumen_alto in excedentes.items():
        for celda_baja, volumen_bajo in deficits.items():
            grafo.add_edge(
                ("alto", celda_alta),
                ("bajo", celda_baja),
                capacity=min(volumen_alto, volumen_bajo, capacidad_maxima),
                weight=distancia_manhattan(celda_alta, celda_baja),
            )
    # Devuelve los flujos resultantes de la asignación
    # origen -> destino optimo
    flujo_resultado = nx.min_cost_flow(grafo)

    # Pasa el grafo a la lista del flujo 
    # seleccionando solo los flujos de celda alta a baja
    # y con volumen mayor a 0
    flujos: list[Flujo] = []
    for nodo_origen, destinos in flujo_resultado.items():
        if nodo_origen[0] != "alto":
            continue
        for nodo_destino, volumen in destinos.items():
            if volumen > 0:
                flujos.append(
                    Flujo(
                        origen=nodo_origen[1],
                        destino=nodo_destino[1],
                        volumen=volumen,
                    )
                )

    return flujos
