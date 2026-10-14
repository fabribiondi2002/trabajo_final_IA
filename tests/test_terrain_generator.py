import numpy as np

from src.allocation import resolver_transportation_problem
from src.terrain_generator import generar_terreno


def test_misma_semilla_genera_el_mismo_terreno():
    alturas_1, objetivo_1 = generar_terreno(tamano=6, seed=42)
    alturas_2, objetivo_2 = generar_terreno(tamano=6, seed=42)

    assert np.array_equal(alturas_1, alturas_2)
    assert np.array_equal(objetivo_1, objetivo_2)


def test_distintas_semillas_generan_terrenos_distintos():
    alturas_1, _ = generar_terreno(tamano=6, seed=1)
    alturas_2, _ = generar_terreno(tamano=6, seed=2)

    assert not np.array_equal(alturas_1, alturas_2)


def test_forma_correcta():
    alturas, objetivo = generar_terreno(tamano=10, seed=7)

    assert alturas.shape == (10, 10)
    assert objetivo.shape == (10, 10)


def test_balance_exacto_en_varias_semillas_y_tamanos():
    """El supuesto de balance del Project Charter: sum(alturas) == sum(objetivo)
    tiene que cumplirse EXACTO, no aproximado, en enteros."""
    for tamano in (4, 6, 10, 13):
        for seed in range(20):
            alturas, objetivo = generar_terreno(tamano=tamano, seed=seed)
            assert alturas.sum() == objetivo.sum(), (
                f"Desbalance en tamano={tamano}, seed={seed}: "
                f"alturas={alturas.sum()}, objetivo={objetivo.sum()}"
            )


def test_objetivo_tiene_pendiente_no_es_plano_uniforme():
    """Confirma que el objetivo realmente es un plano inclinado (con
    pendiente_maxima > 0 debería variar entre filas/columnas), no un
    valor constante como en la versión anterior."""
    _, objetivo = generar_terreno(tamano=10, seed=3, pendiente_maxima=0.5)

    assert objetivo.min() != objetivo.max()


def test_integra_correctamente_con_transportation_problem():
    """El terreno generado tiene que poder resolverse sin ValueError de
    desbalance — es la prueba de integración real entre los dos módulos."""
    for seed in range(15):
        alturas, objetivo = generar_terreno(tamano=8, seed=seed)
        flujos = resolver_transportation_problem(alturas, objetivo)
        volumen_total_flujos = sum(f.volumen for f in flujos)
        excedente_total = int(np.clip(alturas - objetivo, 0, None).sum())

        assert volumen_total_flujos == excedente_total
