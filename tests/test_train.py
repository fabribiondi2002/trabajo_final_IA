import numpy as np
import pytest
import torch
import json 

from src.agent import QNetwork
from src.env import TerrainLevelingEnv
from src.train import (
    ConfigEntrenamiento, cargar_modelo, correr_agente, entrenar,
    epsilon_en, evaluar_red, guardar_modelo, crear_red
)
from src.train import cargar_historial, guardar_historial 


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

def historial_de_ejemplo():
    return {
        "recompensa": [-3.0, -2.5],
        "costo": [30.0, 25.0],
        "perdida": [float("nan"), 0.5],
        "validacion": [(100, 390.0), (200, 388.5)],
        "mejor_costo_validacion": [388.5],
    }


def test_historial_guardado_y_cargado_es_el_mismo(tmp_path):
    ruta = str(tmp_path / "h.json")
    guardar_historial(historial_de_ejemplo(), ruta, ConfigEntrenamiento(seed=7))
    hist, config = cargar_historial(ruta)
    assert hist["recompensa"] == [-3.0, -2.5] and hist["costo"] == [30.0, 25.0]
    assert hist["validacion"] == [(100, 390.0), (200, 388.5)]
    assert hist["mejor_costo_validacion"] == [388.5]
    assert config["seed"] == 7


def test_los_nan_se_guardan_como_null_y_se_recuperan_como_nan(tmp_path):
    ruta = str(tmp_path / "h.json")
    guardar_historial(historial_de_ejemplo(), ruta, ConfigEntrenamiento())
    # el archivo tiene que ser JSON estándar (sin NaN suelto)
    json.loads((tmp_path / "h.json").read_text(), parse_constant=lambda c: pytest.fail(f"JSON no estándar: {c}"))
    hist, _ = cargar_historial(ruta)
    assert np.isnan(hist["perdida"][0]) and hist["perdida"][1] == 0.5


def test_guardar_historial_crea_la_carpeta(tmp_path):
    ruta = tmp_path / "a" / "b" / "h.json"
    guardar_historial(historial_de_ejemplo(), str(ruta), ConfigEntrenamiento())
    assert ruta.exists()


def test_el_historial_de_un_entrenamiento_real_se_puede_guardar_y_cargar(env, tmp_path):
    _, hist = entrenar(env, cfg_chica(tmp_path))
    ruta = str(tmp_path / "real.json")
    guardar_historial(hist, ruta, cfg_chica(tmp_path))
    cargado, _ = cargar_historial(ruta)
    assert cargado["recompensa"] == hist["recompensa"]
    assert cargado["validacion"] == hist["validacion"]
    assert cargado["mejor_costo_validacion"] == hist["mejor_costo_validacion"]
    assert np.allclose(cargado["perdida"], hist["perdida"], equal_nan=True)

def test_crear_red_devuelve_la_arquitectura_pedida(env):
    from src.agent import QNetworkPorViaje
    obs, n = env.observation_space.shape[0], env.action_space.n
    assert type(crear_red(obs, n, 16, "plana")) is QNetwork
    assert type(crear_red(obs, n, 16, "por_viaje")) is QNetworkPorViaje


def test_crear_red_con_arquitectura_desconocida_lanza_error(env):
    with pytest.raises(ValueError):
        crear_red(env.observation_space.shape[0], env.action_space.n, 16, "otra")


@pytest.mark.parametrize("arquitectura", ["plana", "por_viaje"])
def test_guardar_y_cargar_conserva_la_arquitectura(env, tmp_path, arquitectura):
    torch.manual_seed(0)
    cfg = cfg_chica(tmp_path, arquitectura=arquitectura)
    red = crear_red(env.observation_space.shape[0], env.action_space.n, cfg.oculta, arquitectura)
    guardar_modelo(red, cfg.ruta_modelo, cfg, costo_validacion=1.0)
    cargada, meta = cargar_modelo(cfg.ruta_modelo)
    assert type(cargada) is type(red)
    x = torch.rand(3, env.observation_space.shape[0])
    assert torch.equal(red(x), cargada(x))
    assert meta["config"]["arquitectura"] == arquitectura


def test_un_modelo_guardado_sin_campo_arquitectura_se_carga_como_plana(env, tmp_path):
    cfg = cfg_chica(tmp_path)
    red = QNetwork(env.observation_space.shape[0], env.action_space.n, cfg.oculta)
    guardar_modelo(red, cfg.ruta_modelo, cfg, costo_validacion=1.0)
    datos = torch.load(cfg.ruta_modelo, weights_only=False)
    del datos["config"]["arquitectura"]
    torch.save(datos, cfg.ruta_modelo)
    assert type(cargar_modelo(cfg.ruta_modelo)[0]) is QNetwork


def test_entrenar_con_la_red_por_viaje_corre_y_es_reproducible(env, tmp_path):
    cfg = cfg_chica(arquitectura="por_viaje", seed=5)
    red, a = entrenar(env, cfg)
    _, b = entrenar(env, cfg)
    from src.agent import QNetworkPorViaje
    assert isinstance(red, QNetworkPorViaje)
    assert a["recompensa"] == b["recompensa"] and a["validacion"] == b["validacion"]


def test_entrenar_crea_la_red_y_la_red_objetivo_con_la_misma_arquitectura(env, monkeypatch):
    import src.train as modulo
    pedidas = []
    real = modulo.crear_red
    monkeypatch.setattr(modulo, "crear_red", lambda o, n, h, arq: (pedidas.append(arq), real(o, n, h, arq))[1])
    entrenar(env, cfg_chica(arquitectura="por_viaje", episodios=1, evaluar_cada=1))
    assert pedidas == ["por_viaje", "por_viaje"]