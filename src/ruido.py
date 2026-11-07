"""
Ruido de Perlin 2D en Python puro (sin compilar nada).

Reemplaza a `noise.pnoise2` de la librería `noise` (Casey Duncan, licencia MIT), que necesita un
compilador de C para instalarse. Esta versión reproduce paso a paso el código en C de esa librería
(`_perlin.c`), incluida la aritmética en punto flotante de 32 bits, así que da los mismos valores
y los terrenos generados con una misma semilla no cambian.

La tabla de permutación y los gradientes son los de la implementación de referencia de Ken Perlin
("improved noise"), tal como los usa `noise`.
"""

import math

import numpy as np

_f32 = np.float32

_GRAD3 = (
    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    (1, 0, -1), (-1, 0, -1), (0, -1, 1), (0, 1, 1),
)

_PERM = (
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
)


def _suavizar(t):
    """Curva de suavizado 6t^5 - 15t^4 + 10t^3, con la misma cuenta que el código en C."""
    return t * t * t * (t * (t * _f32(6) - _f32(15)) + _f32(10))


def _lerp(t, a, b):
    return a + t * (b - a)


def _grad2(hash_, x, y):
    g = _GRAD3[hash_ & 15]
    return x * _f32(g[0]) + y * _f32(g[1])


def _ruido_una_octava(x, y, repeatx, repeaty, base):
    i = int(np.floor(np.fmod(x, repeatx)))
    j = int(np.floor(np.fmod(y, repeaty)))
    ii = int(np.fmod(_f32(i + 1), repeatx))
    jj = int(np.fmod(_f32(j + 1), repeaty))
    i = (i & 255) + base
    j = (j & 255) + base
    ii = (ii & 255) + base
    jj = (jj & 255) + base

    x = x - np.floor(x)
    y = y - np.floor(y)
    fx = _suavizar(x)
    fy = _suavizar(y)

    a = _PERM[i]
    aa = _PERM[a + j]
    ab = _PERM[a + jj]
    b = _PERM[ii]
    ba = _PERM[b + j]
    bb = _PERM[b + jj]

    return _lerp(
        fy,
        _lerp(fx, _grad2(_PERM[aa], x, y), _grad2(_PERM[ba], x - _f32(1), y)),
        _lerp(fx, _grad2(_PERM[ab], x, y - _f32(1)), _grad2(_PERM[bb], x - _f32(1), y - _f32(1))),
    )


def pnoise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=0):
    """
    Ruido de Perlin 2D con la misma interfaz y los mismos resultados que `noise.pnoise2`.
    Con más de una octava devuelve la suma ponderada normalizada (fBm).
    """
    if octaves < 1:
        raise ValueError("Expected octaves value > 0")
    x, y = _f32(x), _f32(y)
    persistence, lacunarity = _f32(persistence), _f32(lacunarity)
    repeatx, repeaty = _f32(repeatx), _f32(repeaty)

    if octaves == 1:
        return float(_ruido_una_octava(x, y, repeatx, repeaty, base))

    freq, amp = _f32(1), _f32(1)
    maximo, total = _f32(0), _f32(0)
    with np.errstate(all="ignore"):
        for _ in range(octaves):
            total = total + _ruido_una_octava(x * freq, y * freq, repeatx * freq, repeaty * freq, base) * amp
            maximo = maximo + amp
            freq = freq * lacunarity
            amp = amp * persistence
        return float(total / maximo)