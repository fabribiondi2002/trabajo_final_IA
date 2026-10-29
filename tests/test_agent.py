import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agent import QNetwork, QNetworkPorViaje, ReplayBuffer, seleccionar_accion, calcular_objetivo, paso_de_aprendizaje
from src.env import TerrainLevelingEnv



class RedFija(nn.Module):
    """Red de prueba que devuelve siempre los mismos valores Q."""

    def __init__(self, q):
        super().__init__()
        self.q = torch.tensor(q, dtype=torch.float32)

    def forward(self, x):
        return self.q.repeat(x.shape[0], 1)


# ---------- QNetwork
def test_red_devuelve_un_q_por_accion():
    red = QNetwork(obs_dim=12, n_acciones=5)
    assert red(torch.zeros(4, 12)).shape == (4, 5)


def test_red_acepta_una_observacion_sola_con_batch_de_uno():
    red = QNetwork(obs_dim=12, n_acciones=5)
    assert red(torch.zeros(1, 12)).shape == (1, 5)


def test_red_es_deterministica_en_evaluacion():
    red = QNetwork(obs_dim=12, n_acciones=5)
    x = torch.rand(3, 12)
    assert torch.equal(red(x), red(x))


def test_gradientes_llegan_a_todas_las_capas():
    red = QNetwork(obs_dim=12, n_acciones=5)
    red(torch.rand(3, 12)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in red.parameters())


# ---------- ReplayBuffer
def agregar_n(buf, n, obs_dim=4, n_acc=3, empieza=0):
    for k in range(empieza, empieza + n):
        buf.agregar(np.full(obs_dim, k), k % n_acc, float(k), np.full(obs_dim, k + 1), k % 2 == 0, np.ones(n_acc, bool))


def test_buffer_cuenta_lo_agregado():
    buf = ReplayBuffer(10, obs_dim=4, n_acciones=3)
    assert len(buf) == 0
    agregar_n(buf, 4)
    assert len(buf) == 4


def test_buffer_circular_pisa_lo_mas_viejo():
    buf = ReplayBuffer(3, obs_dim=4, n_acciones=3)
    agregar_n(buf, 5)
    assert len(buf) == 3
    assert sorted(buf.rewards.tolist()) == [2.0, 3.0, 4.0]


def test_buffer_guarda_y_devuelve_los_datos_sin_alterarlos():
    buf = ReplayBuffer(5, obs_dim=2, n_acciones=3)
    mascara = np.array([True, False, True])
    buf.agregar(np.array([0.1, 0.2]), 2, -1.5, np.array([0.3, 0.4]), True, mascara)
    lote = buf.muestrear(1)
    assert torch.allclose(lote["obs"], torch.tensor([[0.1, 0.2]]))
    assert lote["acciones"].item() == 2
    assert lote["rewards"].item() == pytest.approx(-1.5)
    assert torch.allclose(lote["obs_siguiente"], torch.tensor([[0.3, 0.4]]))
    assert lote["terminados"].item() == 1.0
    assert lote["mascaras_siguiente"][0].tolist() == mascara.tolist()


def test_muestrear_devuelve_los_tipos_y_formas_esperados():
    buf = ReplayBuffer(10, obs_dim=4, n_acciones=3)
    agregar_n(buf, 8)
    lote = buf.muestrear(5)
    assert lote["obs"].shape == (5, 4) and lote["obs"].dtype == torch.float32
    assert lote["acciones"].shape == (5,) and lote["acciones"].dtype == torch.int64
    assert lote["mascaras_siguiente"].shape == (5, 3) and lote["mascaras_siguiente"].dtype == torch.bool


def test_muestrear_no_repite_experiencias():
    buf = ReplayBuffer(10, obs_dim=4, n_acciones=3)
    agregar_n(buf, 10)
    rewards = buf.muestrear(10)["rewards"].tolist()
    assert sorted(rewards) == [float(k) for k in range(10)]


def test_muestrear_mas_de_lo_que_hay_lanza_error():
    buf = ReplayBuffer(10, obs_dim=4, n_acciones=3)
    agregar_n(buf, 3)
    with pytest.raises(ValueError):
        buf.muestrear(4)


def test_muestreo_es_reproducible_con_la_misma_seed():
    a, b = ReplayBuffer(10, 4, 3, seed=7), ReplayBuffer(10, 4, 3, seed=7)
    agregar_n(a, 10), agregar_n(b, 10)
    assert torch.equal(a.muestrear(4)["rewards"], b.muestrear(4)["rewards"])


# ---------- seleccionar_accion
def test_sin_exploracion_elige_la_mejor_accion_valida():
    red = RedFija([1.0, 9.0, 5.0, 3.0])
    mascara = np.array([True, False, True, True])     # la de Q=9 es inválida
    assert seleccionar_accion(red, np.zeros(4), mascara, epsilon=0.0, rng=np.random.default_rng(0)) == 2


def test_sin_exploracion_elige_el_maximo_cuando_todas_son_validas():
    red = RedFija([1.0, 9.0, 5.0])
    assert seleccionar_accion(red, np.zeros(4), np.ones(3, bool), 0.0, np.random.default_rng(0)) == 1


def test_con_epsilon_uno_explora_pero_solo_acciones_validas():
    red = RedFija([1.0, 9.0, 5.0, 3.0])
    mascara = np.array([True, False, False, True])
    rng = np.random.default_rng(0)
    elegidas = {seleccionar_accion(red, np.zeros(4), mascara, 1.0, rng) for _ in range(200)}
    assert elegidas == {0, 3}


def test_nunca_devuelve_una_accion_invalida_con_redes_y_mascaras_al_azar():
    rng = np.random.default_rng(1)
    for _ in range(200):
        red = QNetwork(obs_dim=6, n_acciones=8)
        mascara = rng.random(8) < 0.5
        if not mascara.any():
            continue
        a = seleccionar_accion(red, rng.random(6), mascara, epsilon=float(rng.random()), rng=rng)
        assert mascara[a]


def test_una_sola_accion_valida_se_elige_siempre():
    red = RedFija([9.0, 1.0, 1.0])
    mascara = np.array([False, False, True])
    assert seleccionar_accion(red, np.zeros(4), mascara, 0.0, np.random.default_rng(0)) == 2


def test_sin_acciones_validas_lanza_error():
    with pytest.raises(ValueError):
        seleccionar_accion(RedFija([1.0, 2.0]), np.zeros(4), np.zeros(2, bool), 0.0, np.random.default_rng(0))


def test_la_exploracion_depende_de_la_rng():
    red = RedFija([1.0, 2.0, 3.0, 4.0])
    mascara = np.ones(4, bool)
    a = [seleccionar_accion(red, np.zeros(4), mascara, 1.0, np.random.default_rng(3)) for _ in range(5)]
    b = [seleccionar_accion(red, np.zeros(4), mascara, 1.0, np.random.default_rng(3)) for _ in range(5)]
    assert a == b


# ---------- integración con el entorno
def test_la_red_y_la_seleccion_funcionan_con_el_entorno_real():
    env = TerrainLevelingEnv(tamano=6, capacidad_maquina=5, max_viajes=60, tamano_pool=3, master_seed=1)
    obs, info = env.reset(seed=0)
    red = QNetwork(env.observation_space.shape[0], env.action_space.n)
    rng = np.random.default_rng(0)
    terminated = False
    while not terminated:
        a = seleccionar_accion(red, obs, info["mascara_acciones"], 0.3, rng)
        assert info["mascara_acciones"][a]
        obs, _, terminated, _, info = env.step(a)
    assert all(env.completados)

def lote_a_mano(rewards, terminados, mascaras, obs_dim=4):
    n = len(rewards)
    return {
        "obs": torch.zeros(n, obs_dim),
        "acciones": torch.zeros(n, dtype=torch.int64),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "obs_siguiente": torch.zeros(n, obs_dim),
        "terminados": torch.tensor(terminados, dtype=torch.float32),
        "mascaras_siguiente": torch.tensor(mascaras, dtype=torch.bool),
    }


def test_objetivo_usa_el_maximo_de_las_acciones_validas():
    red = RedFija([1.0, 9.0, 5.0])
    lote = lote_a_mano([-1.0], [0.0], [[True, False, True]])   # la de Q=9 es inválida
    assert calcular_objetivo(red, lote, gamma=0.5).item() == pytest.approx(-1.0 + 0.5 * 5.0)


def test_objetivo_de_estado_terminal_es_solo_el_reward_y_sin_nan():
    red = RedFija([1.0, 9.0, 5.0])
    lote = lote_a_mano([2.0], [1.0], [[False, False, False]])
    objetivo = calcular_objetivo(red, lote, gamma=0.99)
    assert not torch.isnan(objetivo).any()
    assert objetivo.item() == pytest.approx(2.0)


def test_objetivo_mezcla_terminales_y_no_terminales_en_un_mismo_lote():
    red = RedFija([1.0, 9.0, 5.0])
    lote = lote_a_mano([0.0, 3.0], [0.0, 1.0], [[True, True, True], [False, False, False]])
    assert calcular_objetivo(red, lote, 1.0).tolist() == pytest.approx([9.0, 3.0])


def test_paso_de_aprendizaje_devuelve_un_float_y_mueve_los_pesos():
    torch.manual_seed(0)
    red, objetivo = QNetwork(4, 3), QNetwork(4, 3)
    opt = torch.optim.Adam(red.parameters(), lr=1e-2)
    lote = lote_a_mano([1.0, -1.0], [0.0, 1.0], [[True, True, True], [False, False, False]])
    lote["obs"] = torch.rand(2, 4)
    antes = [p.clone() for p in red.parameters()]
    perdida = paso_de_aprendizaje(red, objetivo, opt, lote, 0.99)
    assert isinstance(perdida, float)
    assert any(not torch.equal(a, p) for a, p in zip(antes, red.parameters()))


def test_paso_de_aprendizaje_no_toca_la_red_objetivo():
    torch.manual_seed(0)
    red, objetivo = QNetwork(4, 3), QNetwork(4, 3)
    opt = torch.optim.Adam(red.parameters(), lr=1e-2)
    lote = lote_a_mano([1.0], [0.0], [[True, True, True]])
    antes = [p.clone() for p in objetivo.parameters()]
    paso_de_aprendizaje(red, objetivo, opt, lote, 0.99)
    assert all(torch.equal(a, p) for a, p in zip(antes, objetivo.parameters()))


def test_la_perdida_baja_al_repetir_el_mismo_lote():
    torch.manual_seed(0)
    red, objetivo = QNetwork(4, 3), QNetwork(4, 3)
    opt = torch.optim.Adam(red.parameters(), lr=1e-2)
    lote = lote_a_mano([5.0, -3.0], [1.0, 1.0], [[False] * 3, [False] * 3])
    lote["obs"] = torch.rand(2, 4)
    lote["acciones"] = torch.tensor([0, 2])
    primera = paso_de_aprendizaje(red, objetivo, opt, lote, 0.99)
    for _ in range(300):
        ultima = paso_de_aprendizaje(red, objetivo, opt, lote, 0.99)
    assert ultima < primera * 0.1

# ---------- QNetworkPorViaje

def test_red_por_viaje_devuelve_un_q_por_viaje():
    red = QNetworkPorViaje(obs_dim=2 + 6 * 5, n_acciones=5)
    assert red(torch.zeros(4, 32)).shape == (4, 5)


def test_red_por_viaje_rechaza_dimensiones_incoherentes():
    with pytest.raises(ValueError):
        QNetworkPorViaje(obs_dim=10, n_acciones=5)


def test_red_por_viaje_puntua_igual_un_viaje_en_cualquier_ranura():
    torch.manual_seed(0)
    red = QNetworkPorViaje(obs_dim=2 + 6 * 3, n_acciones=3)
    pos, viaje = torch.rand(1, 2), torch.rand(1, 6)
    a = torch.cat([pos, viaje, torch.zeros(1, 12)], dim=1)   # el viaje en la ranura 0
    b = torch.cat([pos, torch.zeros(1, 12), viaje], dim=1)   # el mismo viaje en la ranura 2
    assert red(a)[0, 0].item() == pytest.approx(red(b)[0, 2].item())


def test_red_por_viaje_el_puntaje_depende_de_la_posicion_de_la_maquina():
    torch.manual_seed(0)
    red = QNetworkPorViaje(obs_dim=2 + 6 * 2, n_acciones=2)
    viajes = torch.rand(1, 12)
    a = torch.cat([torch.zeros(1, 2), viajes], dim=1)
    b = torch.cat([torch.ones(1, 2), viajes], dim=1)
    assert not torch.allclose(red(a), red(b))


def test_red_por_viaje_el_puntaje_de_un_viaje_no_depende_de_los_otros():
    torch.manual_seed(0)
    red = QNetworkPorViaje(obs_dim=2 + 6 * 3, n_acciones=3)
    pos, v0 = torch.rand(1, 2), torch.rand(1, 6)
    a = torch.cat([pos, v0, torch.rand(1, 12)], dim=1)
    b = torch.cat([pos, v0, torch.rand(1, 12)], dim=1)       # mismos datos en la ranura 0, otros distintos
    assert red(a)[0, 0].item() == pytest.approx(red(b)[0, 0].item())


def test_gradientes_llegan_a_todas_las_capas_de_la_red_por_viaje():
    red = QNetworkPorViaje(obs_dim=2 + 6 * 4, n_acciones=4)
    red(torch.rand(3, 26)).sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in red.parameters())