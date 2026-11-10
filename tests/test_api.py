import json
import logging

import pytest
import torch
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import create_app
from src.agent import QNetworkPorViaje
from src.env import crear_entorno
from src.inferencia import TerrenoNoResoluble
from src.train import ConfigEntrenamiento, guardar_modelo


@pytest.fixture(scope="module")
def ruta_modelo(tmp_path_factory):
    """Un modelo sin entrenar (pesos al azar) pero con el formato real, para probar la API."""
    torch.manual_seed(0)
    env = crear_entorno(dict(peso_distancia=10, recompensa="vacio", lookahead=True), tamano_pool=2)
    red = QNetworkPorViaje(env.observation_space.shape[0], env.action_space.n, 16)
    ruta = str(tmp_path_factory.mktemp("modelo") / "modelo.pt")
    cfg = ConfigEntrenamiento(oculta=16, arquitectura="por_viaje", ruta_modelo=ruta)
    guardar_modelo(red, ruta, cfg, costo_validacion=123.0, entorno=env.parametros())
    return ruta


@pytest.fixture(scope="module")
def cliente(ruta_modelo):
    with TestClient(create_app(ruta_modelo)) as c:
        yield c


def test_health_responde_ok(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok"}


def test_modelo_devuelve_la_configuracion_y_el_entorno(cliente, ruta_modelo):
    r = cliente.get("/modelo")
    assert r.status_code == 200
    datos = r.json()
    assert datos["ruta"] == ruta_modelo
    assert datos["config"]["arquitectura"] == "por_viaje"
    assert datos["entorno"]["lookahead"] is True
    assert datos["costo_validacion"] == 123.0


def test_resolver_devuelve_el_orden_del_agente_y_del_greedy(cliente):
    r = cliente.post("/resolver", json={"seed": 7})
    assert r.status_code == 200
    datos = r.json()
    assert datos["seed"] == 7
    for quien in ("agente", "greedy"):
        assert len(datos[quien]["orden"]) == datos["n_viajes"]
        assert sorted(p["viaje"] for p in datos[quien]["orden"]) == list(range(datos["n_viajes"]))
        assert datos[quien]["costo"] == sum(p["distancia"] for p in datos[quien]["orden"])


def test_resolver_es_deterministico(cliente):
    assert cliente.post("/resolver", json={"seed": 21}).json() == cliente.post("/resolver", json={"seed": 21}).json()


@pytest.mark.parametrize("cuerpo", [{"seed": -1}, {"seed": 1_000_000}, {"seed": "abc"}, {"seed": 1.5}, {}])
def test_resolver_rechaza_semillas_invalidas(cliente, cuerpo):
    assert cliente.post("/resolver", json=cuerpo).status_code == 422


def test_resolver_responde_422_si_el_terreno_no_entra_en_el_modelo(cliente, monkeypatch):
    def no_resoluble(env, red, seed):
        raise TerrenoNoResoluble("demasiados viajes")

    monkeypatch.setattr(api_main, "resolver_terreno", no_resoluble)
    r = cliente.post("/resolver", json={"seed": 3})
    assert r.status_code == 422
    assert "demasiados viajes" in r.json()["detail"]


def test_resolver_registra_un_evento_estructurado(cliente, caplog):
    caplog.set_level(logging.INFO, logger="api")
    cliente.post("/resolver", json={"seed": 4})
    eventos = [json.loads(m) for m in caplog.messages if m.startswith("{")]
    resuelto = [e for e in eventos if e["evento"] == "terreno_resuelto"]
    assert resuelto and resuelto[-1]["seed"] == 4
    assert {"costo_agente", "costo_greedy", "milisegundos"} <= set(resuelto[-1])


def test_no_arranca_si_el_modelo_no_existe():
    with pytest.raises(RuntimeError, match="No se encontró el modelo"):
        with TestClient(create_app("/ruta/que/no/existe.pt")):
            pass


def test_usa_la_variable_de_entorno_MODELO_PATH(ruta_modelo, monkeypatch):
    monkeypatch.setenv("MODELO_PATH", ruta_modelo)
    with TestClient(create_app()) as c:
        assert c.get("/modelo").json()["ruta"] == ruta_modelo


def test_resolver_devuelve_el_terreno_de_la_semilla(cliente):
    from src.terrain_generator import generar_terreno

    datos = cliente.post("/resolver", json={"seed": 7}).json()
    alturas, objetivo = generar_terreno(datos["terreno"]["tamano"], 7)
    assert datos["terreno"]["alturas"] == alturas.astype(int).tolist()
    assert datos["terreno"]["objetivo"] == objetivo.astype(int).tolist()
    assert sum(map(sum, datos["terreno"]["alturas"])) == sum(map(sum, datos["terreno"]["objetivo"]))


def test_terreno_devuelve_alturas_y_objetivo_sin_resolver(cliente):
    from src.terrain_generator import generar_terreno

    r = cliente.get("/terreno/7")
    assert r.status_code == 200
    datos = r.json()
    alturas, objetivo = generar_terreno(datos["terreno"]["tamano"], 7)
    assert datos["seed"] == 7
    assert datos["terreno"]["alturas"] == alturas.astype(int).tolist()
    assert datos["terreno"]["objetivo"] == objetivo.astype(int).tolist()
    assert datos["resoluble"] is True and datos["n_viajes"] > 0
    assert "agente" not in datos and "greedy" not in datos


def test_terreno_coincide_con_el_que_devuelve_resolver(cliente):
    assert cliente.get("/terreno/12").json()["terreno"] == cliente.post("/resolver", json={"seed": 12}).json()["terreno"]


@pytest.mark.parametrize("seed", ["-1", "1000000", "abc", "1.5"])
def test_terreno_rechaza_semillas_invalidas(cliente, seed):
    assert cliente.get(f"/terreno/{seed}").status_code == 422


def test_terreno_marca_no_resoluble_si_excede_max_viajes(cliente):
    app = cliente.app
    original = app.state.env.max_viajes
    app.state.env.max_viajes = 0
    try:
        datos = cliente.get("/terreno/7").json()
    finally:
        app.state.env.max_viajes = original
    assert datos["resoluble"] is False and datos["n_viajes"] is None
    assert datos["terreno"]["alturas"]