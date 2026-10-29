import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Red que estima el valor Q de cada acción a partir de la observación.
    Entrada: vector de observación. Salida: un valor por cada acción posible (viaje).
    Una red feedforward con dos capas ocultas y ReLU con tamaño de la capa oculta configurable.
    """

    def __init__(self, obs_dim: int, n_acciones: int, oculta: int = 128):
        super().__init__()
        self.red = nn.Sequential(
            nn.Linear(obs_dim, oculta),
            nn.ReLU(),
            nn.Linear(oculta, oculta),
            nn.ReLU(),
            nn.Linear(oculta, n_acciones),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x)


class ReplayBuffer:
    """
    Memoria circular de experiencias. 
    Guarda por cada paso obs, acción, reward, obs siguiente, terminó, máscara de acciones válidas del estado siguiente.
    La máscara del estado siguiente hace falta para calcular el objetivo solo sobre acciones válidas.
    """

    def __init__(self, capacidad: int, obs_dim: int, n_acciones: int, seed: int = 0):
        self.capacidad = capacidad
        self.obs = np.zeros((capacidad, obs_dim), dtype=np.float32)
        self.obs_siguiente = np.zeros((capacidad, obs_dim), dtype=np.float32)
        self.acciones = np.zeros(capacidad, dtype=np.int64)
        self.rewards = np.zeros(capacidad, dtype=np.float32)
        self.terminados = np.zeros(capacidad, dtype=np.float32)
        self.mascaras_siguiente = np.zeros((capacidad, n_acciones), dtype=bool)
        self.indice = 0
        self.tamano = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.tamano

    def agregar(self, obs, accion, reward, obs_siguiente, terminado, mascara_siguiente) -> None:
        i = self.indice
        self.obs[i] = obs
        self.acciones[i] = accion
        self.rewards[i] = reward
        self.obs_siguiente[i] = obs_siguiente
        self.terminados[i] = float(terminado)
        self.mascaras_siguiente[i] = mascara_siguiente
        self.indice = (self.indice + 1) % self.capacidad
        self.tamano = min(self.tamano + 1, self.capacidad)

    def muestrear(self, n: int) -> dict[str, torch.Tensor]:
        """ 
        Devuelve un lote de n experiencias aleatorias como tensores de PyTorch.        
        """
        if n > self.tamano:
            raise ValueError(f"Se piden {n} muestras pero solo hay {self.tamano}.")
        idx = self.rng.choice(self.tamano, size=n, replace=False)
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "acciones": torch.from_numpy(self.acciones[idx]),
            "rewards": torch.from_numpy(self.rewards[idx]),
            "obs_siguiente": torch.from_numpy(self.obs_siguiente[idx]),
            "terminados": torch.from_numpy(self.terminados[idx]),
            "mascaras_siguiente": torch.from_numpy(self.mascaras_siguiente[idx]),
        }


def seleccionar_accion(
    red: QNetwork,
    obs: np.ndarray,
    mascara: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    """
    Epsilon-greedy restringido a las acciones válida.
    Si epsilon es menor a un número aleatorio, elige la acción con mayor Q entre las válidas.
    Si no, elige una acción válida al azar.
    Nunca devuelve una inválida.
    """
    validas = np.flatnonzero(mascara)
    if len(validas) == 0:
        raise ValueError("No hay acciones válidas.")
    if rng.random() < epsilon:
        return int(rng.choice(validas))
    with torch.no_grad():
        q = red(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0)
    q = q.masked_fill(~torch.as_tensor(mascara), float("-inf"))
    return int(q.argmax().item())

def calcular_objetivo(
    red_objetivo: QNetwork, lote: dict[str, torch.Tensor], gamma: float
) -> torch.Tensor:
    """
    Calcula el objetivo para un lote de experiencias.
    objetivo = reward + gamma * max(Q(siguiente_obs, a)) sobre acciones válidas.
    Si terminó, el objetivo es solo el reward.
    """
    with torch.no_grad():
        q_siguiente = red_objetivo(lote["obs_siguiente"])
        q_siguiente = q_siguiente.masked_fill(~lote["mascaras_siguiente"], float("-inf"))
        max_q = q_siguiente.max(dim=1).values
        max_q = torch.where(lote["terminados"] > 0, torch.zeros_like(max_q), max_q)
        return lote["rewards"] + gamma * max_q


def paso_de_aprendizaje(
    red: QNetwork,
    red_objetivo: QNetwork,
    optimizador: torch.optim.Optimizer,
    lote: dict[str, torch.Tensor],
    gamma: float,
) -> float:
    """
    Hace un paso de aprendizaje sobre un lote de experiencias.
    1. Predice Q(s, a) con la red, solo para la acción que se tomó en cada experiencia.
    2. Calcula el objetivo con la red objetivo (reward + gamma * mejor Q futuro válido).
    3. Calcula la pérdida de Huber entre la predicción y el objetivo.
    4. Hace backward, recorta el gradiente (norma máxima 10) y da un paso del optimizador.
    Solo se modifican los pesos de `red`; la red objetivo no se toca.
    Devuelve la pérdida como float.
    """
    q_predicho = red(lote["obs"]).gather(1, lote["acciones"].unsqueeze(1)).squeeze(1)
    objetivo = calcular_objetivo(red_objetivo, lote, gamma)
    perdida = F.smooth_l1_loss(q_predicho, objetivo)
    optimizador.zero_grad()
    perdida.backward()
    nn.utils.clip_grad_norm_(red.parameters(), max_norm=10.0)
    optimizador.step()
    return float(perdida.item())