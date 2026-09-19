import numpy as np
import pytest

from src.allocation import (
    calcular_excedentes_y_deficits,
    resolver_transportation_problem,
)


def test_calcula_excedentes_y_deficits_correctamente():
    alturas = np.array([[5.0, 2.0], [1.0, 4.0]])
    objetivo = np.array([[3.0, 3.0], [3.0, 3.0]])

    excedentes, deficits = calcular_excedentes_y_deficits(alturas, objetivo)

    assert excedentes == {(0, 0): 2, (1, 1): 1}
    assert deficits == {(0, 1): 1, (1, 0): 2}


def test_terreno_ya_nivelado_no_genera_flujos():
    alturas = np.array([[3.0, 3.0], [3.0, 3.0]])
    objetivo = np.array([[3.0, 3.0], [3.0, 3.0]])

    flujos = resolver_transportation_problem(alturas, objetivo)

    assert flujos == []


def test_terreno_no_balanceado_lanza_error():
    alturas = np.array([[5.0, 2.0]])
    objetivo = np.array([[3.0, 3.0]])  # excedente=2, déficit=1 -> no balanceado

    with pytest.raises(ValueError):
        resolver_transportation_problem(alturas, objetivo)


def test_flujos_respetan_balance_de_oferta_y_demanda():
    """El volumen total que sale de los altos debe ser igual al volumen
    total que entra a los bajos, y cada celda debe agotar exactamente
    su excedente/déficit (verifica que el min-cost-flow cierra bien)."""
    alturas = np.array([[5.0, 2.0], [1.0, 4.0]])
    objetivo = np.array([[3.0, 3.0], [3.0, 3.0]])

    flujos = resolver_transportation_problem(alturas, objetivo)

    volumen_por_origen = {}
    volumen_por_destino = {}
    for f in flujos:
        volumen_por_origen[f.origen] = volumen_por_origen.get(f.origen, 0) + f.volumen
        volumen_por_destino[f.destino] = volumen_por_destino.get(f.destino, 0) + f.volumen

    assert volumen_por_origen == {(0, 0): 2, (1, 1): 1}
    assert volumen_por_destino == {(0, 1): 1, (1, 0): 2}


def test_muchos_a_muchos_una_zona_alta_reparte_entre_dos_bajas():
    """Verifica el caso central del dominio: una zona alta con excedente
    grande puede repartir su volumen entre varias zonas bajas."""
    alturas = np.array([[7.0, 2.0, 2.0]])
    objetivo = np.array([[3.0, 4.0, 4.0]])  # excedente=4, déficit=2+2=4

    flujos = resolver_transportation_problem(alturas, objetivo)
    volumen_total = sum(f.volumen for f in flujos)

    assert volumen_total == 4
    assert all(f.origen == (0, 0) for f in flujos)
