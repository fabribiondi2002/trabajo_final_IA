import numpy as np
import pytest
import torch

from src.agent import QNetwork
from src.env import TerrainLevelingEnv
from src.train import (
    ConfigEntrenamiento, cargar_modelo, correr_agente, entrenar,
    epsilon_en, evaluar_red, guardar_modelo,
)


@pytest.fixture(scope="module")
def env():
    return TerrainLevelingEnv(tamano=5, capacidad_maquina=5, max_viajes=30, tamano_pool=6, master_seed=1)


def cfg_chica(tmp_path=None, **extra):
    base = dict(
        episodios=6, batch=8, empezar_a_aprender=16, capacidad_buffer=500,
        sincronizar_cada=20, evaluar_cada=3, episodios_decay=4, oculta=16,
        ruta_modelo=str(tmp_path / "m.pt") if tmp_path else None,
    )
    base.update(extra)
    return ConfigEntrenamiento(**base)


def test_epsilon_arranca_en_el_inicial_y_termina_en_el_final():
    cfg = ConfigEntrenamiento(epsilon_inicial=1.0, epsilon_final=0.1, episodios_decay=100)
    assert epsilon_en(0, cfg) == pytest.approx(1.0)
    assert epsilon_en(100, cfg) == pytest.approx(0.1)


def test_epsilon_a_mitad_de_camino_es_el_promedio():
    cfg = ConfigEntrenamiento(epsilon_inicial=1.0, epsilon_final=0.1, episodios_decay=100)
    assert epsilon_en(50, cfg) == pytest.approx(0.55)


def test_epsilon_no_baja_del_minimo_despues_del_decay():
    cfg = ConfigEntrenamiento(epsilon_inicial=1.0, epsilon_final=0.1, episodios_decay=100)
    assert epsilon_en(10_000, cfg) == pytest.approx(0.1)


def test_correr_agente_completa_el_terreno_y_devuelve_su_costo(env):
    red = QNetwork(env.observation_space.shape[0], env.action_space.n, 16)
    costo = correr_agente(env, red, env.pool_test[0])
    assert all(env.completados)
    assert costo == env.costo_acumulado > 0


def test_evaluar_red_devuelve_un_costo_por_terreno_y_es_deterministica(env):
    torch.manual_seed(0)
    red = QNetwork(env.observation_space.shape[0], env.action_space.n, 16)
    a = evaluar_red(env, red, env.pool_validacion[:4])
    b = evaluar_red(env, red, env.pool_validacion[:4])
    assert len(a) == 4 and a == b


def test_guardar_y_cargar_da_la_misma_red(env, tmp_path):
    torch.manual_seed(0)
    cfg = cfg_chica(tmp_path)
    red = QNetwork(env.observation_space.shape[0], env.action_space.n, cfg.oculta)
    guardar_modelo(red, cfg.ruta_modelo, cfg, costo_validacion=123.0)
    cargada, meta = cargar_modelo(cfg.ruta_modelo)
    x = torch.rand(3, env.observation_space.shape[0])
    assert torch.equal(red(x), cargada(x))
    assert meta["costo_validacion"] == 123.0
    assert meta["config"]["seed"] == cfg.seed


def test_guardar_crea_la_carpeta_si_no_existe(env, tmp_path):
    ruta = str(tmp_path / "sub" / "dir" / "m.pt")
    red = QNetwork(env.observation_space.shape[0], env.action_space.n, 16)
    guardar_modelo(red, ruta, cfg_chica(), 1.0)
    assert (tmp_path / "sub" / "dir" / "m.pt").exists()


def test_entrenar_corre_y_devuelve_historial_completo(env, tmp_path):
    red, hist = entrenar(env, cfg_chica(tmp_path))
    assert len(hist["recompensa"]) == 6 and len(hist["costo"]) == 6 and len(hist["perdida"]) == 6
    assert [e for e, _ in hist["validacion"]] == [3, 6]
    assert hist["mejor_costo_validacion"][0] == min(c for _, c in hist["validacion"])


def test_entrenar_guarda_el_mejor_modelo_y_se_puede_cargar(env, tmp_path):
    cfg = cfg_chica(tmp_path)
    entrenar(env, cfg)
    red, meta = cargar_modelo(cfg.ruta_modelo)
    assert meta["obs_dim"] == env.observation_space.shape[0]
    assert meta["n_acciones"] == env.action_space.n
    costos = evaluar_red(env, red, env.pool_validacion)
    assert np.mean(costos) == pytest.approx(meta["costo_validacion"])


def test_la_red_devuelta_es_la_mejor_en_validacion(env, tmp_path):
    red, hist = entrenar(env, cfg_chica(tmp_path))
    costo = float(np.mean(evaluar_red(env, red, env.pool_validacion)))
    assert costo == pytest.approx(hist["mejor_costo_validacion"][0])


def test_entrenar_es_reproducible_con_la_misma_seed(env):
    _, a = entrenar(env, cfg_chica(seed=3))
    _, b = entrenar(env, cfg_chica(seed=3))
    assert a["recompensa"] == b["recompensa"]
    assert a["validacion"] == b["validacion"]


def test_semillas_distintas_dan_entrenamientos_distintos(env):
    _, a = entrenar(env, cfg_chica(seed=1))
    _, b = entrenar(env, cfg_chica(seed=2))
    assert a["recompensa"] != b["recompensa"]


def test_entrenar_no_toca_los_terrenos_de_test(env, tmp_path):
    test_antes = [(e.seed, e.viajes) for e in env.pool_test]
    entrenar(env, cfg_chica(tmp_path))
    assert [(e.seed, e.viajes) for e in env.pool_test] == test_antes