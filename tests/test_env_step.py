import numpy as np
import pytest

from src.env import (
    EstadoCelda,
    EstadoTierraFertil,
    TerrainLevelingEnv,
    aplicar_viaje_a_celdas,
    inicializar_estado_celdas,
)
from src.trip_builder import Viaje
from src.utils import distancia_manhattan


@pytest.fixture
def env():
    e = TerrainLevelingEnv(tamano=6, capacidad_maquina=5, max_viajes=60, tamano_pool=5, master_seed=1)
    e.reset(seed=0)
    return e


def primera_valida(env):
    return int(np.flatnonzero(env._mascara_acciones())[0])


# ---------- máquina de estados de la capa fértil (función pura)
def test_primer_toque_extrae_la_capa_fertil():
    viajes = [Viaje((0, 0), (1, 1), 3), Viaje((0, 0), (1, 1), 2)]
    celdas = inicializar_estado_celdas(viajes)
    aplicar_viaje_a_celdas(celdas, viajes[0])
    assert celdas[(0, 0)].fertil == EstadoTierraFertil.EXTRAIDA
    assert celdas[(0, 0)].volumen_pendiente == 2


def test_ultimo_viaje_repone_la_capa_fertil():
    viajes = [Viaje((0, 0), (1, 1), 3), Viaje((0, 0), (1, 1), 2)]
    celdas = inicializar_estado_celdas(viajes)
    for v in viajes:
        aplicar_viaje_a_celdas(celdas, v)
    for c in [(0, 0), (1, 1)]:
        assert celdas[c].fertil == EstadoTierraFertil.REPUESTA
        assert celdas[c].volumen_pendiente == 0


def test_celda_no_involucrada_no_cambia():
    viajes = [Viaje((0, 0), (1, 1), 3), Viaje((2, 2), (3, 3), 1)]
    celdas = inicializar_estado_celdas(viajes)
    aplicar_viaje_a_celdas(celdas, viajes[0])
    assert celdas[(2, 2)] == EstadoCelda(EstadoTierraFertil.IN_SITU, 1)


def test_viaje_unico_extrae_y_repone_en_el_mismo_paso():
    v = Viaje((0, 0), (1, 1), 3)
    celdas = inicializar_estado_celdas([v])
    aplicar_viaje_a_celdas(celdas, v)
    assert celdas[(0, 0)].fertil == EstadoTierraFertil.REPUESTA


# ---------- step
def test_step_devuelve_la_tupla_de_gymnasium(env):
    obs, reward, terminated, truncated, info = env.step(primera_valida(env))
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False and truncated is False
    assert "mascara_acciones" in info


def test_costo_del_paso_es_ir_al_origen_y_llevar_al_destino(env):
    v = env.viajes[0]
    esperado = distancia_manhattan((0, 0), v.origen) + distancia_manhattan(v.origen, v.destino)
    _, _, _, _, info = env.step(0)
    assert info["distancia_paso"] == esperado
    assert env.costo_acumulado == esperado


def test_la_maquina_termina_en_el_destino(env):
    env.step(0)
    assert env.posicion == env.viajes[0].destino


def test_viaje_se_marca_completado_y_sale_de_la_mascara(env):
    _, _, _, _, info = env.step(0)
    assert env.completados[0] is True
    assert not info["mascara_acciones"][0]


def test_reward_es_distancia_normalizada_negativa(env):
    lado = env.tamano - 1
    _, reward, terminated, _, info = env.step(0)
    assert not terminated
    assert reward == pytest.approx(-info["distancia_paso"] / (2 * lado))


def test_repetir_un_viaje_completado_lanza_error(env):
    env.step(0)
    with pytest.raises(ValueError):
        env.step(0)


def test_accion_de_padding_lanza_error(env):
    with pytest.raises(ValueError):
        env.step(len(env.viajes))


def test_accion_fuera_de_rango_lanza_error(env):
    with pytest.raises(ValueError):
        env.step(env.max_viajes)


def test_episodio_completo_termina_en_n_viajes_pasos(env):
    n = len(env.viajes)
    pasos = 0
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(primera_valida(env))
        pasos += 1
    assert pasos == n


def test_reward_total_es_menos_costo_normalizado_mas_bonus(env):
    lado = env.tamano - 1
    total, terminated = 0.0, False
    while not terminated:
        _, r, terminated, _, _ = env.step(primera_valida(env))
        total += r
    esperado = -env.costo_acumulado / (2 * lado) + env.bonus_final
    assert total == pytest.approx(esperado)


def test_bonus_solo_se_da_en_el_ultimo_paso(env):
    rewards, terminated = [], False
    while not terminated:
        _, r, terminated, _, _ = env.step(primera_valida(env))
        rewards.append(r)
    assert all(r <= 0 for r in rewards[:-1])


def test_al_terminar_todas_las_celdas_estan_repuestas(env):
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(primera_valida(env))
    assert all(
        c.fertil == EstadoTierraFertil.REPUESTA and c.volumen_pendiente == 0
        for c in env.estado_celdas.values()
    )


def test_invariante_nunca_repuesta_con_volumen_pendiente(env):
    rng = np.random.default_rng(0)
    terminated = False
    while not terminated:
        validas = np.flatnonzero(env._mascara_acciones())
        _, _, terminated, _, _ = env.step(int(rng.choice(validas)))
        for c in env.estado_celdas.values():
            assert c.volumen_pendiente >= 0
            assert (c.fertil == EstadoTierraFertil.REPUESTA) == (c.volumen_pendiente == 0)


def test_mismo_orden_de_viajes_da_mismo_costo(env):
    entrada = env.entrada_actual
    costos = []
    for _ in range(2):
        env.reset(options={"entrada": entrada})
        terminated = False
        while not terminated:
            _, _, terminated, _, _ = env.step(primera_valida(env))
        costos.append(env.costo_acumulado)
    assert costos[0] == costos[1]


def test_el_orden_de_los_viajes_cambia_el_costo(env):
    entrada = env.entrada_actual
    costos = []
    for orden in ("asc", "desc"):
        env.reset(options={"entrada": entrada})
        terminated = False
        while not terminated:
            validas = np.flatnonzero(env._mascara_acciones())
            a = int(validas[0] if orden == "asc" else validas[-1])
            _, _, terminated, _, _ = env.step(a)
        costos.append(env.costo_acumulado)
    assert costos[0] != costos[1]

# ---------- recompensa "vacio" (solo castiga el tramo en vacío)
def entorno_con(recompensa):
    e = TerrainLevelingEnv(tamano=6, capacidad_maquina=5, max_viajes=60, tamano_pool=5, master_seed=1, recompensa=recompensa)
    e.reset(seed=0)
    return e


def test_la_recompensa_por_defecto_es_total(env):
    assert env.recompensa == "total"


def test_recompensa_desconocida_lanza_error():
    with pytest.raises(ValueError):
        TerrainLevelingEnv(tamano=6, capacidad_maquina=5, max_viajes=60, tamano_pool=2, recompensa="otra")


def test_reward_vacio_es_solo_el_tramo_en_vacio_normalizado():
    e = entorno_con("vacio")
    lado = e.tamano - 1
    v = e.viajes[0]
    esperado = -distancia_manhattan((0, 0), v.origen) / (2 * lado)
    _, reward, terminated, _, _ = e.step(0)
    assert not terminated
    assert reward == pytest.approx(esperado)


def test_reward_vacio_no_cambia_el_costo_que_se_mide():
    a, b = entorno_con("total"), entorno_con("vacio")
    _, _, _, _, info_a = a.step(0)
    _, _, _, _, info_b = b.step(0)
    assert info_a["distancia_paso"] == info_b["distancia_paso"]
    assert a.costo_acumulado == b.costo_acumulado


def test_reward_total_de_un_episodio_en_vacio_es_menos_los_tramos_vacios_mas_bonus():
    e = entorno_con("vacio")
    lado = e.tamano - 1
    vacios, total, terminated = 0, 0.0, False
    while not terminated:
        accion = primera_valida(e)
        vacios += distancia_manhattan(e.posicion, e.viajes[accion].origen)
        _, r, terminated, _, _ = e.step(accion)
        total += r
    cargados = sum(distancia_manhattan(v.origen, v.destino) for v in e.viajes)
    assert e.costo_acumulado == vacios + cargados
    assert total == pytest.approx(-vacios / (2 * lado) + e.bonus_final)


def test_la_diferencia_entre_ambas_recompensas_es_la_misma_para_cualquier_orden():
    """Es la razón por la que el mejor orden no cambia: el tramo cargado suma lo mismo siempre."""
    def retorno(recompensa, elegir):
        e = entorno_con(recompensa)
        total, terminated = 0.0, False
        while not terminated:
            _, r, terminated, _, _ = e.step(elegir(e))
            total += r
        return total

    primero = primera_valida
    ultimo = lambda e: int(np.flatnonzero(e._mascara_acciones())[-1])
    dif_primero = retorno("total", primero) - retorno("vacio", primero)
    dif_ultimo = retorno("total", ultimo) - retorno("vacio", ultimo)
    assert dif_primero == pytest.approx(dif_ultimo)