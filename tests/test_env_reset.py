import numpy as np
import pytest

from src.env import EstadoTierraFertil, TerrainLevelingEnv


@pytest.fixture(scope="module")
def env():
    return TerrainLevelingEnv(tamano=6, capacidad_maquina=5, max_viajes=60, tamano_pool=10, master_seed=1)


def test_reset_devuelve_observacion_y_info(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert {"seed_terreno", "n_viajes", "mascara_acciones"} <= info.keys()


def test_mascara_tiene_tantos_true_como_viajes(env):
    _, info = env.reset(seed=0)
    mascara = info["mascara_acciones"]
    n = info["n_viajes"]
    assert mascara.shape == (env.max_viajes,)
    assert mascara.sum() == n
    assert mascara[:n].all() and not mascara[n:].any()


def test_mismo_seed_en_reset_da_el_mismo_terreno(env):
    _, a = env.reset(seed=42)
    _, b = env.reset(seed=42)
    assert a["seed_terreno"] == b["seed_terreno"]


def test_options_permite_forzar_un_terreno_de_test(env):
    entrada = env.pool_test[0]
    _, info = env.reset(options={"entrada": entrada})
    assert info["seed_terreno"] == entrada.seed
    assert info["n_viajes"] == len(entrada.viajes)


def test_estado_inicial_de_la_maquina_y_costo(env):
    env.reset(seed=0)
    assert env.posicion == (0, 0)
    assert env.costo_acumulado == 0
    assert not any(env.completados)


def test_modificar_estado_celdas_no_contamina_el_pool(env):
    env.reset(seed=0)
    for estado in env.estado_celdas.values():
        estado.fertil = EstadoTierraFertil.REPUESTA
        estado.volumen_pendiente = 0
    env.reset(options={"entrada": env.entrada_actual})
    assert all(e.fertil == EstadoTierraFertil.IN_SITU for e in env.estado_celdas.values())
    assert all(e.volumen_pendiente > 0 for e in env.estado_celdas.values())


def test_padding_de_la_observacion_es_cero(env):
    _, info = env.reset(seed=0)
    obs = env._armar_observacion()
    assert not obs[2 + 6 * info["n_viajes"]:].any()


def test_observacion_normalizada_en_rango(env):
    obs, _ = env.reset(seed=0)
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_viaje_completado_desaparece_de_obs_y_mascara(env):
    env.reset(seed=0)
    env.completados[0] = True
    obs = env._armar_observacion()
    assert not obs[2:8].any()
    assert not env._mascara_acciones()[0]


def test_distancia_en_observacion_es_desde_la_maquina(env):
    env.reset(seed=0)
    v = env.viajes[0]
    env.posicion = v.origen
    obs = env._armar_observacion()
    assert obs[2 + 5] == 0.0