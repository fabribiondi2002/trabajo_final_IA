from src.terrain_generator import generar_terreno


def test_semillas_con_igual_modulo_256_dan_terrenos_distintos():
    a, _ = generar_terreno(10, seed=5)
    b, _ = generar_terreno(10, seed=261)
    assert not (a == b).all()


def test_sin_solapamiento_entre_alturas_de_test_y_entrenamiento():
    test = {generar_terreno(10, seed=s)[0].tobytes() for s in range(0, 100)}
    entrenamiento = {generar_terreno(10, seed=s)[0].tobytes() for s in range(100, 400)}
    assert test.isdisjoint(entrenamiento)


def test_hay_diversidad_de_heightmaps():
    mapas = {generar_terreno(10, seed=s)[0].tobytes() for s in range(300)}
    assert len(mapas) == 300