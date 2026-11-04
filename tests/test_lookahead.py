import json

import numpy as np
import pytest
import torch

from src.agent import QNetworkPorViaje
from src.env import TerrainLevelingEnv, crear_entorno, inicializar_estado_celdas
from src.evaluation import evaluar_modelo
from src.train import ConfigEntrenamiento, cargar_modelo, entrenar, guardar_historial, guardar_modelo
from src.trip_builder import Viaje

PARAMS = dict(tamano=6, capacidad_maquina=5, max_viajes=60, tamano_pool=5, master_seed=1)


@pytest.fixture(scope="module")
def env_la():
    e = TerrainLevelingEnv(**PARAMS, lookahead=True)
    e.reset(seed=0)
    return e


@pytest.fixture
def env_a_mano(env_la):
    """Entorno con 3 viajes elegidos a mano. lado = 5, así que las distancias se normalizan por 10."""
    env_la.viajes = [
        Viaje(origen=(0, 0), destino=(1, 1), volumen=1),   # A
        Viaje(origen=(4, 0), destino=(2, 0), volumen=1),   # B
        Viaje(origen=(0, 3), destino=(3, 3), volumen=1),   # C
    ]
    env_la.completados = [False, False, False]
    env_la.estado_celdas = inicializar_estado_celdas(env_la.viajes)
    env_la.posicion = (0, 0)
    return env_la


def la(env, i):
    """Los 2 datos de lookahead del viaje i en la observación actual."""
    obs = env._armar_observacion()
    base = 2 + 8 * i
    return obs[base + 6], obs[base + 7]


# ---------- entorno: forma de la observación
def test_con_lookahead_la_observacion_tiene_8_features_por_viaje(env_la):
    obs, _ = env_la.reset(seed=0)
    assert obs.shape == (2 + 8 * 60,) and env_la.observation_space.shape == (2 + 8 * 60,)


def test_sin_lookahead_la_observacion_sigue_siendo_de_6_features_por_viaje():
    e = TerrainLevelingEnv(**PARAMS)
    obs, _ = e.reset(seed=0)
    assert obs.shape == (2 + 6 * 60,)


def test_las_primeras_6_features_no_cambian_con_lookahead():
    a, b = TerrainLevelingEnv(**PARAMS), TerrainLevelingEnv(**PARAMS, lookahead=True)
    entrada = a.pool_test[0]
    obs_a, _ = a.reset(options={"entrada": entrada})
    obs_b, _ = b.reset(options={"entrada": entrada})
    n = len(entrada.viajes)
    for i in range(n):
        assert np.array_equal(obs_a[2 + 6 * i : 2 + 6 * i + 6], obs_b[2 + 8 * i : 2 + 8 * i + 6])
    assert np.array_equal(obs_a[:2], obs_b[:2])


# ---------- entorno: valores del lookahead (calculados a mano)
def test_distancia_al_origen_mas_cercano_de_los_otros_viajes(env_a_mano):
    # A: destino (1,1) -> otros orígenes B(4,0)=4, C(0,3)=3  => 3/10
    # B: destino (2,0) -> A(0,0)=2, C(0,3)=5               => 2/10
    # C: destino (3,3) -> A(0,0)=6, B(4,0)=4               => 4/10
    assert [la(env_a_mano, i)[0] for i in range(3)] == pytest.approx([0.3, 0.2, 0.4])


def test_distancia_al_centroide_de_los_orígenes_de_los_otros_viajes(env_a_mano):
    # A: centroide de B y C = (2, 1.5) -> |1-2| + |1-1.5| = 1.5 => 0.15
    # B: centroide de A y C = (0, 1.5) -> |2-0| + |0-1.5| = 3.5 => 0.35
    # C: centroide de A y B = (2, 0)   -> |3-2| + |3-0|   = 4   => 0.40
    assert [la(env_a_mano, i)[1] for i in range(3)] == pytest.approx([0.15, 0.35, 0.40])


def test_un_viaje_completado_no_cuenta_como_siguiente_y_queda_en_cero(env_a_mano):
    env_a_mano.completados = [True, False, False]
    assert la(env_a_mano, 0) == (0.0, 0.0)
    assert la(env_a_mano, 1) == pytest.approx((0.5, 0.5))     # solo queda C: (2,0) -> (0,3) = 5
    assert la(env_a_mano, 2) == pytest.approx((0.4, 0.4))     # solo queda B: (3,3) -> (4,0) = 4


def test_con_un_solo_viaje_pendiente_no_hay_siguiente(env_a_mano):
    env_a_mano.completados = [True, True, False]
    assert la(env_a_mano, 2) == (0.0, 0.0)


def test_las_ranuras_de_relleno_valen_cero(env_a_mano):
    obs = env_a_mano._armar_observacion()
    assert not obs[2 + 8 * 3 :].any()


def test_el_lookahead_no_depende_de_donde_esta_la_maquina(env_a_mano):
    antes = [la(env_a_mano, i) for i in range(3)]
    env_a_mano.posicion = (5, 5)
    assert [la(env_a_mano, i) for i in range(3)] == antes


def test_el_lookahead_se_actualiza_despues_de_un_paso(env_a_mano):
    env_a_mano.step(0)     # se hace A: ahora quedan B y C
    assert la(env_a_mano, 1) == pytest.approx((0.5, 0.5))


# ---------- parámetros del entorno
def test_parametros_incluye_todo_lo_que_define_el_entorno(env_la):
    assert env_la.parametros() == dict(**PARAMS, peso_distancia=1.0, bonus_final=1.0, recompensa="total", lookahead=True)


def test_crear_entorno_a_partir_de_los_parametros_guardados(env_la):
    e = crear_entorno(env_la.parametros(), tamano_pool=2)
    assert e.lookahead and e.tamano == 6 and e.max_viajes == 60 and e.tamano_pool == 2
    assert e.observation_space.shape == env_la.observation_space.shape


def test_crear_entorno_sin_parametros_usa_los_de_por_defecto():
    e = crear_entorno(None, tamano_pool=2)
    assert not e.lookahead and e.recompensa == "total" and e.tamano == 8


def test_los_parametros_guardados_conservan_peso_y_recompensa():
    e = TerrainLevelingEnv(**PARAMS, peso_distancia=10.0, recompensa="vacio")
    nuevo = crear_entorno(e.parametros(), tamano_pool=2)
    assert nuevo.peso_distancia == 10.0 and nuevo.recompensa == "vacio"


# ---------- red por viaje con 8 features
def test_la_red_por_viaje_acepta_6_u_8_features():
    assert QNetworkPorViaje(2 + 6 * 5, 5, 8).features_por_viaje == 6
    assert QNetworkPorViaje(2 + 8 * 5, 5, 8).features_por_viaje == 8


def test_la_red_por_viaje_rechaza_una_observacion_que_no_encaja():
    with pytest.raises(ValueError):
        QNetworkPorViaje(2 + 8 * 5 + 1, 5, 8)
    with pytest.raises(ValueError):
        QNetworkPorViaje(2, 5, 8)


def test_la_red_por_viaje_con_8_features_da_un_q_por_viaje_y_no_depende_del_orden():
    torch.manual_seed(0)
    red = QNetworkPorViaje(2 + 8 * 4, 4, 8)
    x = torch.rand(3, 2 + 8 * 4)
    q = red(x)
    assert q.shape == (3, 4)
    orden = [2, 0, 3, 1]
    permutada = torch.cat([x[:, :2], x[:, 2:].reshape(3, 4, 8)[:, orden, :].reshape(3, -1)], dim=1)
    assert torch.allclose(red(permutada), q[:, orden], atol=1e-6)


# ---------- el modelo guarda el entorno y se puede reconstruir
def entrenar_chico(tmp_path):
    e = TerrainLevelingEnv(tamano=5, capacidad_maquina=5, max_viajes=30, tamano_pool=6, master_seed=1, lookahead=True)
    cfg = ConfigEntrenamiento(
        episodios=4, batch=8, empezar_a_aprender=16, capacidad_buffer=500, sincronizar_cada=20,
        evaluar_cada=2, episodios_decay=2, oculta=8, arquitectura="por_viaje", ruta_modelo=str(tmp_path / "m.pt"),
    )
    red, hist = entrenar(e, cfg)
    return e, cfg, red, hist


def test_el_modelo_entrenado_guarda_los_parametros_del_entorno(tmp_path):
    e, _, _, _ = entrenar_chico(tmp_path)
    _, meta = cargar_modelo(str(tmp_path / "m.pt"))
    assert meta["entorno"] == e.parametros() and meta["entorno"]["lookahead"] is True


def test_guardar_modelo_sin_entorno_guarda_none(tmp_path):
    e, cfg, red, _ = entrenar_chico(tmp_path)
    guardar_modelo(red, str(tmp_path / "otro.pt"), cfg, 1.0)
    assert cargar_modelo(str(tmp_path / "otro.pt"))[1]["entorno"] is None


def test_el_historial_guarda_los_parametros_del_entorno(tmp_path):
    e, cfg, _, hist = entrenar_chico(tmp_path)
    guardar_historial(hist, str(tmp_path / "h.json"), cfg, e.parametros())
    with open(tmp_path / "h.json", encoding="utf-8") as f:
        assert json.load(f)["entorno"] == e.parametros()


def test_un_modelo_con_lookahead_no_se_puede_evaluar_en_un_entorno_sin_lookahead(tmp_path):
    e, _, _, _ = entrenar_chico(tmp_path)
    sin = TerrainLevelingEnv(tamano=5, capacidad_maquina=5, max_viajes=30, tamano_pool=2, master_seed=1)
    with pytest.raises(ValueError):
        evaluar_modelo(str(tmp_path / "m.pt"), sin, sin.pool_test[:2], reinicios=1)


def test_un_modelo_con_lookahead_se_evalua_en_el_entorno_recreado_desde_el_modelo(tmp_path):
    e, _, _, _ = entrenar_chico(tmp_path)
    _, meta = cargar_modelo(str(tmp_path / "m.pt"))
    recreado = crear_entorno(meta["entorno"], tamano_pool=2)
    r = evaluar_modelo(str(tmp_path / "m.pt"), recreado, recreado.pool_test[:2], reinicios=1)
    assert r.n_terrenos == 2