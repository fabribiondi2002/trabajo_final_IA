from pathlib import Path

import numpy as np
import pytest

from src.env import TerrainLevelingEnv
from src.evaluation import resumir
from src.experiment import (
    ResultadoSeed,
    correr_experimento,
    estadisticas,
    formatear_experimento,
    resumir_experimento,
)
from src.train import ConfigEntrenamiento, cargar_historial, cargar_modelo


@pytest.fixture(scope="module")
def env():
    return TerrainLevelingEnv(tamano=5, capacidad_maquina=5, max_viajes=30, tamano_pool=6, master_seed=1)


def cfg_chica():
    return ConfigEntrenamiento(
        episodios=6, batch=8, empezar_a_aprender=16, capacidad_buffer=500,
        sincronizar_cada=20, evaluar_cada=3, episodios_decay=4, oculta=16,
    )


def resultado_seed(seed, costo_agente):
    return ResultadoSeed(seed, 1.0, resumir([costo_agente], [10.0], [8.0]))


# ---------- estadísticas (funciones puras)
def test_estadisticas_media_y_desvio_muestral_a_mano():
    media, desvio = estadisticas([2.0, 4.0, 6.0])
    assert media == pytest.approx(4.0)
    assert desvio == pytest.approx(2.0)          # ddof=1: sqrt(((-2)^2 + 0 + 2^2) / 2)


def test_estadisticas_con_un_solo_valor_tiene_desvio_cero():
    assert estadisticas([5.0]) == (5.0, 0.0)


def test_estadisticas_sin_valores_lanza_error():
    with pytest.raises(ValueError):
        estadisticas([])


def test_resumir_experimento_calcula_media_y_desvio_de_costo_y_brecha():
    resumen = resumir_experimento([resultado_seed(0, 9.0), resultado_seed(1, 9.5)])
    assert resumen.costo_medio == pytest.approx(9.25)
    assert resumen.costo_desvio == pytest.approx(np.std([9.0, 9.5], ddof=1))
    assert resumen.brecha_media == pytest.approx((0.5 + 0.25) / 2)     # brechas 0.5 y 0.25
    assert resumen.costo_greedy == 10.0 and resumen.costo_referencia == 8.0
    assert resumen.seeds == (0, 1)


def test_resumir_experimento_sin_resultados_lanza_error():
    with pytest.raises(ValueError):
        resumir_experimento([])


def test_formatear_incluye_una_fila_por_seed_y_el_resumen():
    resultados = [resultado_seed(0, 9.0), resultado_seed(1, 9.5)]
    texto = formatear_experimento(resultados, resumir_experimento(resultados))
    assert texto.count("\n") >= 6
    assert "9.25" in texto and "10.00" in texto and "8.00" in texto


# ---------- correr_experimento con el entorno real
def test_correr_experimento_devuelve_un_resultado_por_seed(env, tmp_path):
    resultados = correr_experimento(env, cfg_chica(), [0, 1], str(tmp_path), env.pool_validacion[:3], reinicios=1)
    assert [r.seed for r in resultados] == [0, 1]
    assert all(r.resultado.n_terrenos == 3 for r in resultados)


def test_correr_experimento_guarda_modelo_e_historial_de_cada_seed(env, tmp_path):
    correr_experimento(env, cfg_chica(), [0, 1], str(tmp_path), env.pool_validacion[:2], reinicios=1)
    for seed in (0, 1):
        assert (tmp_path / f"modelo_seed_{seed}.pt").exists()
        _, config = cargar_historial(str(tmp_path / f"historial_seed_{seed}.json"))
        assert config["seed"] == seed
        _, meta = cargar_modelo(str(tmp_path / f"modelo_seed_{seed}.pt"))
        assert meta["config"]["seed"] == seed


def test_greedy_y_referencia_son_los_mismos_para_todas_las_seeds(env, tmp_path):
    resultados = correr_experimento(env, cfg_chica(), [0, 1, 2], str(tmp_path), env.pool_validacion[:2], reinicios=1)
    assert len({r.resultado.costo_greedy for r in resultados}) == 1
    assert len({r.resultado.costo_referencia for r in resultados}) == 1


def test_seeds_distintas_dan_costos_de_validacion_distintos(env, tmp_path):
    resultados = correr_experimento(env, cfg_chica(), [0, 1], str(tmp_path), env.pool_validacion[:2], reinicios=1)
    assert resultados[0].costo_validacion != resultados[1].costo_validacion


def test_el_experimento_es_reproducible(env, tmp_path):
    a = correr_experimento(env, cfg_chica(), [3], str(tmp_path / "a"), env.pool_validacion[:2], reinicios=1)
    b = correr_experimento(env, cfg_chica(), [3], str(tmp_path / "b"), env.pool_validacion[:2], reinicios=1)
    assert a == b


def test_el_costo_de_validacion_es_el_del_modelo_guardado(env, tmp_path):
    resultados = correr_experimento(env, cfg_chica(), [0], str(tmp_path), env.pool_validacion[:2], reinicios=1)
    _, meta = cargar_modelo(str(tmp_path / "modelo_seed_0.pt"))
    assert resultados[0].costo_validacion == pytest.approx(meta["costo_validacion"])


def test_sin_seeds_lanza_error(env, tmp_path):
    with pytest.raises(ValueError):
        correr_experimento(env, cfg_chica(), [], str(tmp_path))


def test_crea_la_carpeta_si_no_existe(env, tmp_path):
    carpeta = tmp_path / "x" / "y"
    correr_experimento(env, cfg_chica(), [0], str(carpeta), env.pool_validacion[:2], reinicios=1)
    assert Path(carpeta).is_dir()