import pytest

from src.env import EstadoCelda, EstadoTierraFertil, inicializar_estado_celdas
from src.trip_builder import Viaje
from src.utils import distancia_manhattan


# ---------- distancia_manhattan ----------

def test_distancia_a_si_misma_es_cero():
    assert distancia_manhattan((3, 4), (3, 4)) == 0


def test_distancia_es_simetrica():
    a, b = (1, 7), (4, 2)

    assert distancia_manhattan(a, b) == distancia_manhattan(b, a)


@pytest.mark.parametrize("a, b, esperado", [
    ((0, 0), (2, 3), 5),   # 2 filas + 3 columnas
    ((5, 5), (1, 2), 7),   # 4 filas + 3 columnas
    ((0, 0), (0, 9), 9),   # solo columnas
    ((6, 1), (2, 1), 4),   # solo filas
])
def test_distancia_calculada_a_mano(a, b, esperado):
    assert distancia_manhattan(a, b) == esperado


# ---------- inicializar_estado_celdas ----------

def test_celda_con_varios_viajes_suma_sus_volumenes():
    # un flujo de 23 con capacidad 10 genera viajes de 10, 10 y 3
    viajes = [
        Viaje(origen=(0, 0), destino=(1, 1), volumen=10),
        Viaje(origen=(0, 0), destino=(1, 1), volumen=10),
        Viaje(origen=(0, 0), destino=(1, 1), volumen=3),
    ]

    estado = inicializar_estado_celdas(viajes)

    assert estado[(0, 0)].volumen_pendiente == 23
    assert estado[(1, 1)].volumen_pendiente == 23


def test_origen_compartido_suma_ambos_volumenes():
    viajes = [
        Viaje(origen=(0, 0), destino=(1, 1), volumen=5),
        Viaje(origen=(0, 0), destino=(2, 2), volumen=7),
    ]

    estado = inicializar_estado_celdas(viajes)

    assert estado[(0, 0)].volumen_pendiente == 12
    assert estado[(1, 1)].volumen_pendiente == 5
    assert estado[(2, 2)].volumen_pendiente == 7


def test_todas_las_celdas_arrancan_in_situ():
    viajes = [
        Viaje(origen=(0, 0), destino=(1, 1), volumen=5),
        Viaje(origen=(2, 2), destino=(3, 3), volumen=4),
    ]

    estado = inicializar_estado_celdas(viajes)

    assert len(estado) == 4
    assert all(c.fertil == EstadoTierraFertil.IN_SITU for c in estado.values())


def test_celda_ajena_a_los_viajes_no_aparece():
    viajes = [Viaje(origen=(0, 0), destino=(1, 1), volumen=5)]

    estado = inicializar_estado_celdas(viajes)

    assert (9, 9) not in estado
    assert set(estado.keys()) == {(0, 0), (1, 1)}


def test_lista_vacia_da_diccionario_vacio():
    assert inicializar_estado_celdas([]) == {}


def test_cada_valor_es_un_estado_celda():
    viajes = [Viaje(origen=(0, 0), destino=(1, 1), volumen=5)]

    estado = inicializar_estado_celdas(viajes)

    assert all(isinstance(c, EstadoCelda) for c in estado.values())