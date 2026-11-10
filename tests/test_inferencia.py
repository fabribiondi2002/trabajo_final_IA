import numpy as np
import pytest
import torch

from src.agent import QNetworkPorViaje
from src.baseline import correr_baseline
from src.env import crear_entorno
from src.inferencia import TerrenoNoResoluble, resolver_terreno
from src.train import correr_agente


@pytest.fixture(scope="module")
def env():
    return crear_entorno(dict(peso_distancia=10, recompensa="vacio", lookahead=True), tamano_pool=2)


@pytest.fixture(scope="module")
def red(env):
    torch.manual_seed(0)
    return QNetworkPorViaje(env.observation_space.shape[0], env.action_space.n, 16).eval()


def test_entrada_de_semilla_da_el_mismo_terreno_que_el_pool(env):
    entrada = env.pool_test[3]
    assert env.entrada_de_semilla(entrada.seed) == entrada


def test_entrada_de_semilla_devuelve_none_si_excede_max_viajes(env, monkeypatch):
    monkeypatch.setattr(env, "max_viajes", 1)
    assert env.entrada_de_semilla(0) is None


@pytest.mark.parametrize("quien", ["agente", "greedy"])
def test_el_orden_usa_cada_viaje_exactamente_una_vez(env, red, quien):
    r = resolver_terreno(env, red, 5)
    viajes = [paso["viaje"] for paso in r[quien]["orden"]]
    assert sorted(viajes) == list(range(r["n_viajes"]))
    assert [paso["paso"] for paso in r[quien]["orden"]] == list(range(1, r["n_viajes"] + 1))


@pytest.mark.parametrize("quien", ["agente", "greedy"])
def test_el_costo_es_la_suma_de_las_distancias_de_cada_paso(env, red, quien):
    r = resolver_terreno(env, red, 5)
    assert r[quien]["costo"] == sum(paso["distancia"] for paso in r[quien]["orden"])


def test_los_costos_coinciden_con_el_agente_y_el_greedy_de_la_evaluacion(env, red):
    for seed in (0, 7, 42):
        entrada = env.entrada_de_semilla(seed)
        r = resolver_terreno(env, red, seed)
        assert r["agente"]["costo"] == pytest.approx(correr_agente(env, red, entrada))
        assert r["greedy"]["costo"] == pytest.approx(correr_baseline(env, entrada))


def test_la_mejora_se_calcula_respecto_del_greedy(env, red):
    r = resolver_terreno(env, red, 11)
    a, g = r["agente"]["costo"], r["greedy"]["costo"]
    assert r["mejora_vs_greedy_pct"] == pytest.approx(100 * (g - a) / g)


def test_el_detalle_de_cada_viaje_es_el_del_terreno(env, red):
    r = resolver_terreno(env, red, 9)
    viajes = env.entrada_de_semilla(9).viajes
    for paso in r["agente"]["orden"]:
        viaje = viajes[paso["viaje"]]
        assert paso["origen"] == list(viaje.origen)
        assert paso["destino"] == list(viaje.destino)
        assert paso["volumen"] == viaje.volumen


def test_es_deterministico(env, red):
    assert resolver_terreno(env, red, 13) == resolver_terreno(env, red, 13)


def test_terreno_que_excede_el_maximo_de_viajes_lanza_error(env, red, monkeypatch):
    monkeypatch.setattr(env, "max_viajes", 1)
    with pytest.raises(TerrenoNoResoluble):
        resolver_terreno(env, red, 0)