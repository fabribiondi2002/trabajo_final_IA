import numpy as np
import pytest

from src.baseline import correr_baseline, elegir_accion_greedy, evaluar_baseline
from src.trip_builder import Viaje


@pytest.fixture(scope="module")
def env():
    from src.env import TerrainLevelingEnv
    return TerrainLevelingEnv(tamano=6, capacidad_maquina=5, max_viajes=60, tamano_pool=5, master_seed=1)


def armar(env, viajes, posicion=(0, 0)):
    env.reset(seed=0)
    env.viajes = viajes
    env.completados = [False] * len(viajes)
    env.posicion = posicion


def test_elige_el_origen_mas_cercano(env):
    armar(env, [Viaje((5, 5), (5, 4), 1), Viaje((1, 0), (2, 0), 1), Viaje((3, 3), (3, 2), 1)])
    assert elegir_accion_greedy(env) == 1


def test_cercania_se_mide_desde_la_posicion_actual(env):
    armar(env, [Viaje((5, 5), (5, 4), 1), Viaje((1, 0), (2, 0), 1)], posicion=(5, 4))
    assert elegir_accion_greedy(env) == 0


def test_empate_se_resuelve_por_menor_indice(env):
    armar(env, [Viaje((0, 2), (1, 2), 1), Viaje((2, 0), (2, 1), 1)])
    assert elegir_accion_greedy(env) == 0


def test_ignora_viajes_completados(env):
    armar(env, [Viaje((1, 0), (2, 0), 1), Viaje((5, 5), (5, 4), 1)])
    env.completados[0] = True
    assert elegir_accion_greedy(env) == 1


def test_sin_viajes_pendientes_lanza_error(env):
    armar(env, [Viaje((1, 0), (2, 0), 1)])
    env.completados[0] = True
    with pytest.raises(ValueError):
        elegir_accion_greedy(env)


def test_correr_baseline_completa_todos_los_viajes(env):
    correr_baseline(env, env.pool_test[0])
    assert all(env.completados)


def test_correr_baseline_es_determinista(env):
    entrada = env.pool_test[0]
    assert correr_baseline(env, entrada) == correr_baseline(env, entrada)


def test_evaluar_baseline_devuelve_un_costo_por_terreno(env):
    costos = evaluar_baseline(env, env.pool_test[:5])
    assert len(costos) == 5
    assert all(c > 0 for c in costos)


def test_greedy_le_gana_en_promedio_a_una_politica_aleatoria(env):
    rng = np.random.default_rng(0)
    entradas = env.pool_test[:30]
    greedy = np.mean(evaluar_baseline(env, entradas))
    aleatorio = []
    for e in entradas:
        env.reset(options={"entrada": e})
        terminated = False
        while not terminated:
            a = int(rng.choice(np.flatnonzero(env._mascara_acciones())))
            _, _, terminated, _, _ = env.step(a)
        aleatorio.append(env.costo_acumulado)
    assert greedy < np.mean(aleatorio)