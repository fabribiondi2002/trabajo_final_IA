import pytest

from src.allocation import Flujo
from src.trip_builder import construir_viajes


def test_volumen_multiplo_exacto_de_la_capacidad():
    flujos = [Flujo(origen=(0, 0), destino=(1, 1), volumen=20)]

    viajes = construir_viajes(flujos, capacidad_maquina=10)

    assert len(viajes) == 2
    assert [v.volumen for v in viajes] == [10, 10]


def test_volumen_con_remanente():
    flujos = [Flujo(origen=(0, 0), destino=(1, 1), volumen=23)]

    viajes = construir_viajes(flujos, capacidad_maquina=10)

    assert len(viajes) == 3
    assert [v.volumen for v in viajes] == [10, 10, 3]


def test_volumen_menor_a_la_capacidad_genera_un_solo_viaje():
    flujos = [Flujo(origen=(0, 0), destino=(1, 1), volumen=4)]

    viajes = construir_viajes(flujos, capacidad_maquina=10)

    assert len(viajes) == 1
    assert viajes[0].volumen == 4


def test_suma_de_volumenes_de_viajes_igual_al_volumen_del_flujo():
    flujos = [
        Flujo(origen=(0, 0), destino=(1, 1), volumen=37),
        Flujo(origen=(2, 2), destino=(3, 3), volumen=8),
    ]

    viajes = construir_viajes(flujos, capacidad_maquina=6)

    volumen_por_flujo = {}
    for v in viajes:
        clave = (v.origen, v.destino)
        volumen_por_flujo[clave] = volumen_por_flujo.get(clave, 0) + v.volumen

    assert volumen_por_flujo[((0, 0), (1, 1))] == 37
    assert volumen_por_flujo[((2, 2), (3, 3))] == 8


def test_origen_y_destino_se_preservan_en_cada_viaje():
    flujos = [Flujo(origen=(5, 1), destino=(0, 9), volumen=15)]

    viajes = construir_viajes(flujos, capacidad_maquina=10)

    assert all(v.origen == (5, 1) and v.destino == (0, 9) for v in viajes)


def test_varios_flujos_generan_la_cantidad_total_esperada_de_viajes():
    flujos = [
        Flujo(origen=(0, 0), destino=(1, 1), volumen=10),   # 1 viaje
        Flujo(origen=(2, 2), destino=(3, 3), volumen=25),   # 3 viajes
        Flujo(origen=(4, 4), destino=(5, 5), volumen=1),    # 1 viaje
    ]

    viajes = construir_viajes(flujos, capacidad_maquina=10)

    assert len(viajes) == 1 + 3 + 1


def test_lista_de_flujos_vacia_devuelve_lista_de_viajes_vacia():
    assert construir_viajes([], capacidad_maquina=10) == []


def test_capacidad_invalida_lanza_error():
    flujos = [Flujo(origen=(0, 0), destino=(1, 1), volumen=10)]

    with pytest.raises(ValueError):
        construir_viajes(flujos, capacidad_maquina=0)

    with pytest.raises(ValueError):
        construir_viajes(flujos, capacidad_maquina=-5)
