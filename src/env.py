from dataclasses import dataclass
from enum import Enum

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.allocation import resolver_transportation_problem
from src.terrain_generator import generar_terreno
from src.trip_builder import Viaje, construir_viajes
from src.utils import distancia_manhattan

SEMILLAS_TEST = range(0, 100)
RANGO_SEMILLAS_ENTRENAMIENTO = range(100, 1_000_000)
MAX_INTENTOS_POR_TERRENO = 50

class EstadoTierraFertil(Enum):
    """
    IN_SITU: la tierra fertil está en su lugar original
    EXTRAIDA: la tierra fertil fue extraída de su celda original y está en tránsito
    REPUESTA: la tierra fertil fue extraída y luego repuesta en su celda original
    """
    IN_SITU = "in_situ"
    EXTRAIDA = "extraida"
    REPUESTA = "repuesta"

@dataclass
class EstadoCelda:
    """
    Representa el estado de una celda en el terreno. 
    Contiene información sobre si la tierra fertil está en su lugar original, si fue extraída o repuesta, 
    y cuánto volumen de tierra pendiente hay para esa celda.
    """
    fertil: EstadoTierraFertil
    volumen_pendiente: int

@dataclass(frozen=True)
class EntradaPool:
    seed: int
    viajes: tuple[Viaje, ...]

def inicializar_estado_celdas(viajes: list[Viaje]) -> dict[tuple[int, int], EstadoCelda]:
    """
    Recorre los viajes y arma un EstadoCelda por cada celda involucrada
    (como origen o como destino). Todas arrancan en IN_SITU y con
    volumen_pendiente = suma de los volúmenes de los viajes que la tocan.
    """
    estado_celdas: dict[tuple[int, int], EstadoCelda] = {}

    for viaje in viajes:
        for celda in [viaje.origen, viaje.destino]:
            if celda not in estado_celdas:
                estado_celdas[celda] = EstadoCelda(
                    fertil=EstadoTierraFertil.IN_SITU,
                    volumen_pendiente=0,
                )
            estado_celdas[celda].volumen_pendiente += viaje.volumen

    return estado_celdas

def _actualizar_celda(estado: EstadoCelda, volumen: int) -> None:
    """
    Un viaje toca esta celda: extrae la capa fértil si hacía falta y descuenta volumen.
    Cambia el estado de la celda
    """
    if estado.fertil == EstadoTierraFertil.IN_SITU:
        estado.fertil = EstadoTierraFertil.EXTRAIDA
    estado.volumen_pendiente -= volumen
    if estado.volumen_pendiente == 0:
        estado.fertil = EstadoTierraFertil.REPUESTA


def aplicar_viaje_a_celdas(estado_celdas: dict[tuple[int, int], EstadoCelda], viaje: Viaje) -> None:
    """
    Actualiza origen y destino del viaje.
    """
    for celda in (viaje.origen, viaje.destino):
        _actualizar_celda(estado_celdas[celda], viaje.volumen)

class TerrainLevelingEnv(gym.Env):
    """
    Entorno de Gym para el problema de nivelación de terreno.
    """
    def __init__(
        self,
        tamano: int = 8,
        capacidad_maquina: int = 5,
        max_viajes: int = 50,
        tamano_pool: int = 200,
        master_seed: int = 0,
        peso_distancia: float = 1.0,
        bonus_final: float = 1.0,
    ):
        super().__init__()
        self.tamano = tamano
        self.capacidad_maquina = capacidad_maquina
        self.max_viajes = max_viajes
        self.master_seed = master_seed
        self.action_space = spaces.Discrete(max_viajes)

        ## Observación es un vector de floats:
        ## 2 floats: posición de la máquina (x, y)
        ## 6 floats por viaje: (origen_x, origen_y, destino_x, destino_y, volumen, distancia_manhattan)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2 + 6 * max_viajes,), dtype=np.float32
        )

        ## Se contruye el pool de test y el de entrenamiento.
        self.pool_test = self._construir_entradas(list(SEMILLAS_TEST))
        self.pool_entrenamiento = self._construir_pool(tamano_pool)

        self.peso_distancia = peso_distancia
        self.bonus_final = bonus_final

    def _viajes_de_semilla(self, seed: int) -> tuple[Viaje, ...] | None:
        """
        Crear y devuelve los viajes del terreno de una semilla,
        devuelve None si la cantidad de viajes excede max_viajes.
        """
        
        ## Se genera el terreno
        alturas, objetivo = generar_terreno(self.tamano, seed)
        
        ## Se resuelve el transportation problem y se construyen los flujos
        flujos = resolver_transportation_problem(alturas, objetivo)
        
        ## Se construyen los viajes a partir de los flujos y la capacidad de la máquina
        viajes = construir_viajes(flujos, self.capacidad_maquina)
        
        if len(viajes) > self.max_viajes:
            return None
        return tuple(viajes)

    def _construir_entradas(self, semillas: list[int]) -> list[EntradaPool]:
        """
        Construye las entradas (conjuntos de viajes) a partir de una lista de semillas.
        """
        entradas = []
        for seed in semillas:
            viajes = self._viajes_de_semilla(seed)
            if viajes is not None:
                entradas.append(EntradaPool(seed=seed, viajes=viajes))
        return entradas

    def _construir_pool(self, tamano_pool: int) -> list[EntradaPool]:
        """
        Crea la lista de semillas de entrenamiento y construye el pool de entradas a partir de ellas.
        Se asegura de que no haya semillas repetidas y de que cada conjunto de viajes
        no exceda el máximo de viajes permitido."""

        ## Se inicializa un generador de números aleatorios con la semilla maestra
        rng = np.random.default_rng(self.master_seed)
        pool: list[EntradaPool] = []
        usadas: set[int] = set()
        intentos = 0
        limite = tamano_pool * MAX_INTENTOS_POR_TERRENO
        ## Se generan semillas aleatorias y se construyen los viajes correspondientes hasta llenar el pool
        while len(pool) < tamano_pool:
            if intentos >= limite:
                raise RuntimeError(
                    "No se pudo llenar el pool: max_viajes es muy chico para este tamano/capacidad."
                )
            intentos += 1
            seed = int(rng.integers(RANGO_SEMILLAS_ENTRENAMIENTO.start, RANGO_SEMILLAS_ENTRENAMIENTO.stop))
            ## Se asegura de que la semilla no se haya usado antes
            if seed in usadas:
                continue
            usadas.add(seed)
            viajes = self._viajes_de_semilla(seed)
            if viajes is not None:
                pool.append(EntradaPool(seed=seed, viajes=viajes))
        return pool

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ## Usar una entrada especifica del conjunto de test
        if options is not None and "entrada" in options:
            entrada = options["entrada"]
        ## Usar una entrada de entrenamiento aleatoria del pool
        else:
            indice = int(self.np_random.integers(len(self.pool_entrenamiento)))
            entrada = self.pool_entrenamiento[indice]

        self.entrada_actual = entrada
        self.viajes = list(entrada.viajes)
        self.completados = [False] * len(self.viajes)
        self.estado_celdas = inicializar_estado_celdas(self.viajes)
        self.posicion = (0, 0)
        self.costo_acumulado = 0.0

        return self._armar_observacion(), self._armar_info()

    def _armar_info(self) -> dict:
        """
        Devuelve un diccionario con información adicional sobre el estado del entorno.
        """
        return {
            "seed_terreno": self.entrada_actual.seed,
            "n_viajes": len(self.viajes),
            "mascara_acciones": self._mascara_acciones(),
        }

    def _mascara_acciones(self) -> np.ndarray:
        """
        Devuelve un array booleano de tamaño max_viajes donde:
        True = acción válida (viaje pendiente). 
        False = Completados y padding."""
        mascara = np.zeros(self.max_viajes, dtype=bool)
        for i, completado in enumerate(self.completados):
            mascara[i] = not completado
        return mascara

    def _armar_observacion(self) -> np.ndarray:
        """
        Arma la observación del entorno como un vector de floats normalizados
        [pos_x, pos_y] + 6 valores por viaje:
        (origen_x, origen_y, destino_x, destino_y, volumen, dist. máquina->origen).
        """
        lado = max(self.tamano - 1, 1)
        obs = np.zeros(2 + 6 * self.max_viajes, dtype=np.float32)
        ## normaliza la posicion de la máquina
        obs[0] = self.posicion[0] / lado
        obs[1] = self.posicion[1] / lado
        ## normaliza los datos de los viajes pendientes
        for i, viaje in enumerate(self.viajes):
            if self.completados[i]:
                continue
            base = 2 + 6 * i
            obs[base] = viaje.origen[0] / lado
            obs[base + 1] = viaje.origen[1] / lado
            obs[base + 2] = viaje.destino[0] / lado
            obs[base + 3] = viaje.destino[1] / lado
            obs[base + 4] = viaje.volumen / self.capacidad_maquina
            obs[base + 5] = distancia_manhattan(self.posicion, viaje.origen) / (2 * lado)
        return obs

    def step(self, action: int):
        """
        Ejecuta un paso en el entorno: la máquina realiza el viaje correspondiente a la acción.
        Devuelve la nueva observación, la recompensa, si el episodio terminó,
        si se truncó y un diccionario con información adicional.
        """
        action = int(action)
        if not (0 <= action < self.max_viajes) or action >= len(self.viajes) or self.completados[action]:
            raise ValueError(f"Acción inválida: {action}. Usar la máscara de acciones.")

        viaje = self.viajes[action]

        # La máquina va hasta el origen y luego lleva la carga al destino.
        distancia = (
            distancia_manhattan(self.posicion, viaje.origen)
            + distancia_manhattan(viaje.origen, viaje.destino)
        )
        # actualiza el costo acumulado y la posición de la máquina
        self.costo_acumulado += distancia
        self.posicion = viaje.destino

        # actualiza el estado de la capa fértil y volúmenes pendientes.
        aplicar_viaje_a_celdas(self.estado_celdas, viaje)
        self.completados[action] = True

        lado = max(self.tamano - 1, 1)
        # Calcula la recompensa: negativa proporcional a la distancia recorrida, 
        # más un bonus si se completan todos los viajes.
        reward = -self.peso_distancia * distancia / (2 * lado)

        terminated = all(self.completados)
        if terminated:
            reward += self.bonus_final

        info = self._armar_info()
        info["distancia_paso"] = distancia
        info["costo_acumulado"] = self.costo_acumulado
        return self._armar_observacion(), float(reward), terminated, False, info