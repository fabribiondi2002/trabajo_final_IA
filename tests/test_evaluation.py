import numpy as np
import pytest

from src.agent import QNetwork
from src.baseline import evaluar_baseline
from src.env import TerrainLevelingEnv
from src.evaluation import (
    comparar,
    costos_referencia,
    evaluar_modelo,
    formatear,
    resumir,
)
from src.train import ConfigEntrenamiento, evaluar_red, guardar_modelo


@pytest.fixture(scope="module")
def env():
    return TerrainLevelingEnv(tamano=5, capacidad_maquina=5, max_viajes=30, tamano_pool=4, master_seed=1)


@pytest.fixture(scope="module")
def red(env):
    return QNetwork(env.observation_space.shape[0], env.action_space.n, 16)


# ---------- resumir (función pura)
def test_resumir_calcula_las_medias_y_la_brecha_a_mano():
    r = resumir([9.0, 9.0], [10.0, 10.0], [8.0, 8.0])
    assert (r.costo_agente, r.costo_greedy, r.costo_referencia) == (9.0, 10.0, 8.0)
    assert r.brecha_cerrada == pytest.approx(0.5)
    assert r.n_terrenos == 2


def test_resumir_agente_igual_a_greedy_da_brecha_cero():
    assert resumir([10.0, 12.0], [10.0, 12.0], [9.0, 10.0]).brecha_cerrada == pytest.approx(0.0)


def test_resumir_agente_igual_a_la_referencia_da_brecha_uno():
    assert resumir([8.0, 9.0], [10.0, 12.0], [8.0, 9.0]).brecha_cerrada == pytest.approx(1.0)


def test_resumir_agente_peor_que_greedy_da_brecha_negativa():
    assert resumir([11.0], [10.0], [8.0]).brecha_cerrada < 0


def test_resumir_cuenta_terrenos_mejores_y_peores_sin_contar_empates():
    r = resumir([9.0, 12.0, 10.0], [10.0, 10.0, 10.0], [8.0, 8.0, 8.0])
    assert r.terrenos_mejores_que_greedy == 1
    assert r.terrenos_peores_que_greedy == 1


@pytest.mark.parametrize("a, g, r", [([1.0], [1.0, 2.0], [1.0]), ([], [], [])])
def test_resumir_rechaza_listas_de_distinto_largo_o_vacias(a, g, r):
    with pytest.raises(ValueError):
        resumir(a, g, r)


# ---------- referencia y comparación con el entorno real
def test_referencia_nunca_es_peor_que_greedy_en_ningun_terreno(env):
    entradas = env.pool_test[:3]
    ref = costos_referencia(entradas, reinicios=2)
    greedy = evaluar_baseline(env, entradas)
    assert all(r <= g for r, g in zip(ref, greedy))


def test_comparar_coincide_con_las_mediciones_por_separado(env, red):
    entradas = env.pool_validacion[:3]
    r = comparar(env, red, entradas, reinicios=2)
    assert r.costo_agente == pytest.approx(np.mean(evaluar_red(env, red, entradas)))
    assert r.costo_greedy == pytest.approx(np.mean(evaluar_baseline(env, entradas)))
    assert r.costo_referencia <= r.costo_greedy


def test_comparar_es_reproducible(env, red):
    entradas = env.pool_validacion[:3]
    assert comparar(env, red, entradas, reinicios=2) == comparar(env, red, entradas, reinicios=2)


# ---------- cargar un modelo guardado
def test_evaluar_modelo_da_lo_mismo_que_comparar_con_la_red_en_memoria(env, red, tmp_path):
    ruta = str(tmp_path / "m.pt")
    guardar_modelo(red, ruta, ConfigEntrenamiento(oculta=16), costo_validacion=1.0)
    entradas = env.pool_validacion[:3]
    assert evaluar_modelo(ruta, env, entradas, reinicios=2) == comparar(env, red, entradas, reinicios=2)


def test_evaluar_modelo_rechaza_un_modelo_de_otro_entorno(env, tmp_path):
    otra = QNetwork(10, 3, 8)
    ruta = str(tmp_path / "otro.pt")
    guardar_modelo(otra, ruta, ConfigEntrenamiento(oculta=8), costo_validacion=1.0)
    with pytest.raises(ValueError):
        evaluar_modelo(ruta, env, env.pool_validacion[:2], reinicios=1)


# ---------- formato
def test_formatear_muestra_los_numeros_clave():
    texto = formatear(resumir([9.0, 9.0], [10.0, 10.0], [8.0, 8.0]))
    assert "9.00" in texto and "10.00" in texto and "8.00" in texto and "50.0%" in texto


def test_evaluar_modelo_rechaza_distinto_obs_dim_aunque_coincidan_las_acciones(env, tmp_path):
    otra = QNetwork(10, env.action_space.n, 8)
    ruta = str(tmp_path / "otro_obs.pt")
    guardar_modelo(otra, ruta, ConfigEntrenamiento(oculta=8), costo_validacion=1.0)
    with pytest.raises(ValueError):
        evaluar_modelo(ruta, env, env.pool_validacion[:2], reinicios=1)


def test_costos_referencia_pasa_los_reinicios_pedidos(env, monkeypatch):
    pedidos = []

    def falsa(viajes, reinicios=8, **kw):
        pedidos.append(reinicios)
        return [], 1

    monkeypatch.setattr("src.evaluation.mejor_orden_conocido", falsa)
    costos_referencia(env.pool_validacion[:2], reinicios=5)
    assert pedidos == [5, 5]