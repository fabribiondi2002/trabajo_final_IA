import pytest

from src.env import (
    RANGO_SEMILLAS_ENTRENAMIENTO,
    SEMILLAS_TEST,
    TerrainLevelingEnv,
)


@pytest.fixture(scope="module")
def env():
    return TerrainLevelingEnv(tamano=6, capacidad_maquina=5, max_viajes=60, tamano_pool=20, master_seed=1)


def test_espacios_tienen_el_tamano_esperado(env):
    assert env.action_space.n == 60
    assert env.observation_space.shape == (2 + 6 * 60,)


def test_pool_tiene_el_tamano_pedido(env):
    assert len(env.pool_entrenamiento) == 20


def test_semillas_de_entrenamiento_estan_en_su_rango(env):
    assert all(e.seed in RANGO_SEMILLAS_ENTRENAMIENTO for e in env.pool_entrenamiento)


def test_semillas_de_test_estan_en_su_rango(env):
    assert all(e.seed in SEMILLAS_TEST for e in env.pool_test)


def test_rangos_de_semillas_son_disjuntos():
    assert set(SEMILLAS_TEST).isdisjoint(RANGO_SEMILLAS_ENTRENAMIENTO)


def test_pool_sin_semillas_repetidas(env):
    semillas = [e.seed for e in env.pool_entrenamiento]
    assert len(semillas) == len(set(semillas))


def test_ninguna_entrada_excede_max_viajes(env):
    todas = env.pool_entrenamiento + env.pool_test
    assert all(len(e.viajes) <= env.max_viajes for e in todas)


def test_mismo_master_seed_mismo_pool():
    a = TerrainLevelingEnv(tamano=6, max_viajes=60, tamano_pool=5, master_seed=7)
    b = TerrainLevelingEnv(tamano=6, max_viajes=60, tamano_pool=5, master_seed=7)
    assert [e.seed for e in a.pool_entrenamiento] == [e.seed for e in b.pool_entrenamiento]


def test_distinto_master_seed_distinto_pool():
    a = TerrainLevelingEnv(tamano=6, max_viajes=60, tamano_pool=5, master_seed=7)
    b = TerrainLevelingEnv(tamano=6, max_viajes=60, tamano_pool=5, master_seed=8)
    assert [e.seed for e in a.pool_entrenamiento] != [e.seed for e in b.pool_entrenamiento]


def test_max_viajes_muy_chico_lanza_error():
    with pytest.raises(RuntimeError):
        TerrainLevelingEnv(tamano=8, capacidad_maquina=1, max_viajes=1, tamano_pool=3)