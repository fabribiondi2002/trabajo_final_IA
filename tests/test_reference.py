import itertools

import numpy as np
import pytest

from src.baseline import correr_baseline
from src.env import TerrainLevelingEnv
from src.reference import (
    brecha_cerrada,
    costo_de_orden,
    mejor_orden_conocido,
    orden_greedy,
)
from src.trip_builder import Viaje


@pytest.fixture(scope="module")
def env():
    return TerrainLevelingEnv(tamano=6, capacidad_maquina=5, max_viajes=60, tamano_pool=3, master_seed=1)


VIAJES = [Viaje((0, 1), (0, 3), 5), Viaje((2, 3), (2, 0), 5)]


# ---------- costo_de_orden
def test_costo_calculado_a_mano():
    # (0,0)->(0,1)=1, carga a (0,3)=2, vacío a (2,3)=2, carga a (2,0)=3
    assert costo_de_orden([0, 1], VIAJES) == 8
    # (0,0)->(2,3)=5, carga a (2,0)=3, vacío a (0,1)=3, carga a (0,3)=2
    assert costo_de_orden([1, 0], VIAJES) == 13


def test_costo_de_lista_vacia_es_cero():
    assert costo_de_orden([], []) == 0


@pytest.mark.parametrize("orden", [[0, 0], [0], [0, 1, 2], [1, 2]])
def test_orden_que_no_es_permutacion_lanza_error(orden):
    with pytest.raises(ValueError):
        costo_de_orden(orden, VIAJES)


def test_costo_coincide_con_el_del_entorno(env):
    rng = np.random.default_rng(0)
    for entrada in env.pool_test[:10]:
        viajes = list(entrada.viajes)
        orden = [int(x) for x in rng.permutation(len(viajes))]
        env.reset(options={"entrada": entrada})
        for a in orden:
            env.step(a)
        assert costo_de_orden(orden, viajes) == env.costo_acumulado


# ---------- greedy
def test_greedy_empieza_por_el_origen_mas_cercano():
    viajes = [Viaje((5, 5), (5, 4), 1), Viaje((1, 0), (2, 0), 1)]
    assert orden_greedy(viajes)[0] == 1


def test_greedy_es_permutacion_y_coincide_con_el_baseline(env):
    for entrada in env.pool_test[:5]:
        viajes = list(entrada.viajes)
        orden = orden_greedy(viajes)
        assert sorted(orden) == list(range(len(viajes)))
        assert costo_de_orden(orden, viajes) == correr_baseline(env, entrada)


# ---------- búsqueda local
def test_mejor_orden_es_permutacion_y_su_costo_es_consistente(env):
    viajes = list(env.pool_test[0].viajes)
    orden, costo = mejor_orden_conocido(viajes, reinicios=2)
    assert sorted(orden) == list(range(len(viajes)))
    assert costo == costo_de_orden(orden, viajes)


def test_nunca_es_peor_que_el_greedy(env):
    for entrada in env.pool_test[:10]:
        viajes = list(entrada.viajes)
        _, costo = mejor_orden_conocido(viajes, reinicios=2)
        assert costo <= costo_de_orden(orden_greedy(viajes), viajes)


def test_es_deterministico_para_una_seed(env):
    viajes = list(env.pool_test[0].viajes)
    assert mejor_orden_conocido(viajes, reinicios=3, seed=5) == mejor_orden_conocido(viajes, reinicios=3, seed=5)


def test_sin_viajes_devuelve_orden_vacio():
    assert mejor_orden_conocido([]) == ([], 0)


def test_coincide_con_la_fuerza_bruta_en_instancias_chicas():
    rng = np.random.default_rng(3)
    for _ in range(8):
        viajes = [
            Viaje(tuple(rng.integers(0, 6, 2)), tuple(rng.integers(0, 6, 2)), 5)
            for _ in range(6)
        ]
        optimo = min(costo_de_orden(list(p), viajes) for p in itertools.permutations(range(6)))
        _, hallado = mejor_orden_conocido(viajes, reinicios=8)
        assert hallado == optimo


# ---------- brecha cerrada
def test_brecha_cerrada_valores_clave():
    assert brecha_cerrada(100, 100, 90) == pytest.approx(0.0)
    assert brecha_cerrada(90, 100, 90) == pytest.approx(1.0)
    assert brecha_cerrada(95, 100, 90) == pytest.approx(0.5)
    assert brecha_cerrada(110, 100, 90) < 0


def test_brecha_cerrada_sin_brecha_lanza_error():
    with pytest.raises(ValueError):
        brecha_cerrada(100, 90, 90)