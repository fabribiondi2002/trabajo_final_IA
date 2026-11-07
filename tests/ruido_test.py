import random

import pytest

from src.ruido import pnoise2

# Valores calculados con noise.pnoise2 (versión 1.2.2, código en C) para comprobar que el reemplazo
# en Python puro da exactamente lo mismo.
PUNTOS = [(0.0, 0.0), (0.3, 0.7), (1.25, 2.5), (-3.7, 4.2), (10.1, -0.9), (7.77, 7.77)]
UNA_OCTAVA = [0.0, 0.09597451984882355, 0.310302734375, 0.3139703869819641, 0.089676633477211, -0.28273066878318787]
CUATRO_OCTAVAS = [0.0, 0.09150972962379456, 0.23216146230697632, 0.047428857535123825, 0.004884668160229921, -0.1525256633758545]


@pytest.mark.parametrize("octavas, esperados", [(1, UNA_OCTAVA), (4, CUATRO_OCTAVAS)])
def test_da_los_mismos_valores_que_la_libreria_noise(octavas, esperados):
    for (x, y), esperado in zip(PUNTOS, esperados):
        assert pnoise2(x, y, octaves=octavas) == pytest.approx(esperado, abs=1e-12)


def test_coincide_con_la_libreria_noise_en_muchos_puntos_si_esta_instalada():
    noise = pytest.importorskip("noise")
    azar = random.Random(0)
    for octavas in (1, 2, 4):
        for _ in range(500):
            x, y = azar.uniform(-50, 50), azar.uniform(-50, 50)
            assert pnoise2(x, y, octaves=octavas) == noise.pnoise2(x, y, octaves=octavas)


def test_es_deterministico():
    assert pnoise2(3.3, 1.1, octaves=4) == pnoise2(3.3, 1.1, octaves=4)


def test_los_valores_quedan_en_el_rango_esperado():
    azar = random.Random(1)
    for _ in range(500):
        valor = pnoise2(azar.uniform(-20, 20), azar.uniform(-20, 20), octaves=4)
        assert -1.0 <= valor <= 1.0


def test_en_los_vertices_enteros_el_ruido_vale_cero():
    for x in range(-3, 4):
        for y in range(-3, 4):
            assert pnoise2(float(x), float(y)) == 0.0


def test_cero_octavas_es_un_error():
    with pytest.raises(ValueError):
        pnoise2(0.5, 0.5, octaves=0)