"""
Módulo generador de terrenos sintéticos (heightmaps) para el entorno de nivelación.

Genera dos grillas de la misma forma: `alturas` (el terreno actual, con ruido
Perlin) y `objetivo` (el perfil deseado, un plano inclinado). Garantiza que
sum(alturas) == sum(objetivo) exactamente en enteros.
"""

import numpy as np
from noise import pnoise2


def _generar_heightmap(
    tamano: int, seed: int, escala: float, amplitud: int, nivel_base: int
) -> np.ndarray:
    """Genera una grilla de alturas con ruido Perlin, redondeada a enteros."""
    base = seed % 256  # 'noise' usa un entero 0-255 para variar el patrón
    alturas = np.zeros((tamano, tamano), dtype=float)
    for i in range(tamano):
        for j in range(tamano):
            ruido = pnoise2(i / escala, j / escala, octaves=4, base=base)
            alturas[i, j] = nivel_base + ruido * amplitud
    return np.round(alturas).astype(int)


def _redondear_preservando_suma(valores: np.ndarray, suma_objetivo: int) -> np.ndarray:
    """
    Redondea 'valores' (float) a enteros ajustando para que la suma total
    dé exactamente 'suma_objetivo' (método de los restos mayores).
    """
    piso = np.floor(valores).astype(int)
    resto = int(suma_objetivo - piso.sum())
    fraccion = (valores - piso).flatten()
    resultado = piso.flatten().copy()

    if resto > 0:
        indices_mayor_fraccion = np.argsort(-fraccion)[:resto]
        resultado[indices_mayor_fraccion] += 1
    elif resto < 0:
        indices_menor_fraccion = np.argsort(fraccion)[:-resto]
        resultado[indices_menor_fraccion] -= 1

    return resultado.reshape(valores.shape)


def _generar_plano_objetivo(
    tamano: int, pendiente_x: float, pendiente_y: float, suma_objetivo: int
) -> np.ndarray:
    """
    Genera un plano inclinado con las pendientes dadas, ajustado para que
    su suma total sea exactamente 'suma_objetivo'.
    """
    filas, columnas = np.indices((tamano, tamano))
    plano = (pendiente_x * filas + pendiente_y * columnas).astype(float)

    n_celdas = tamano * tamano
    offset = (suma_objetivo - plano.sum()) / n_celdas
    plano += offset

    return _redondear_preservando_suma(plano, suma_objetivo)


def generar_terreno(
    tamano: int,
    seed: int,
    escala: float = 8.0,
    amplitud: int = 10,
    nivel_base: int = 50,
    pendiente_maxima: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera un terreno sintético balanceado: una grilla de alturas actuales
    (ruido Perlin) y un perfil objetivo (plano inclinado), garantizando que
    el volumen total de corte sea igual al volumen total de relleno.

    Parámetros:
        tamano: dimensión de la grilla (tamano x tamano)
        seed: semilla para reproducibilidad (mismo seed -> mismo terreno)
        escala: escala del ruido Perlin (mayor = terreno más suave)
        amplitud: variación máxima de altura respecto al nivel base
        nivel_base: altura media de referencia
        pendiente_maxima: pendiente máxima del plano objetivo (altura por celda)

    Devuelve:
        (alturas, objetivo), dos arrays numpy de forma (tamano, tamano),
        con sum(alturas) == sum(objetivo) garantizado.
    """
    rng = np.random.default_rng(seed)

    alturas = _generar_heightmap(tamano, seed, escala, amplitud, nivel_base)

    pendiente_x = rng.uniform(-pendiente_maxima, pendiente_maxima)
    pendiente_y = rng.uniform(-pendiente_maxima, pendiente_maxima)

    objetivo = _generar_plano_objetivo(
        tamano, pendiente_x, pendiente_y, suma_objetivo=int(alturas.sum())
    )

    return alturas, objetivo