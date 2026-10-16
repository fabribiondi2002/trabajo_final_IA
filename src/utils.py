

def distancia_manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Distancia Manhattan entre dos celdas de la grilla."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])