import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.agent import QNetwork, ReplayBuffer, paso_de_aprendizaje, seleccionar_accion
from src.env import EntradaPool, TerrainLevelingEnv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigEntrenamiento:
    episodios: int = 3000
    gamma: float = 0.99
    lr: float = 1e-3
    batch: int = 64
    oculta: int = 128
    capacidad_buffer: int = 50_000
    epsilon_inicial: float = 1.0
    epsilon_final: float = 0.05
    episodios_decay: int = 1500        # episodios que tarda epsilon en llegar al mínimo
    sincronizar_cada: int = 500        # pasos entre copias red -> red objetivo
    empezar_a_aprender: int = 1000     # experiencias mínimas antes de aprender
    evaluar_cada: int = 100            # episodios entre evaluaciones en validación
    seed: int = 0
    ruta_modelo: str | None = "models/dqn.pt"


def epsilon_en(episodio: int, cfg: ConfigEntrenamiento) -> float:
    """Decaimiento lineal de epsilon; se queda en el mínimo después de episodios_decay."""
    fraccion = min(episodio / max(cfg.episodios_decay, 1), 1.0)
    return cfg.epsilon_inicial + fraccion * (cfg.epsilon_final - cfg.epsilon_inicial)


def correr_agente(env: TerrainLevelingEnv, red: QNetwork, entrada: EntradaPool) -> float:
    """Un episodio sin exploración (mejor acción válida). Devuelve el costo total."""
    obs, info = env.reset(options={"entrada": entrada})
    rng = np.random.default_rng(0)
    terminado = False
    while not terminado:
        accion = seleccionar_accion(red, obs, info["mascara_acciones"], 0.0, rng)
        obs, _, terminado, _, info = env.step(accion)
    return env.costo_acumulado


def evaluar_red(env: TerrainLevelingEnv, red: QNetwork, entradas: list[EntradaPool]) -> list[float]:
    """Costo del agente (sin exploración) en cada terreno de la lista."""
    return [correr_agente(env, red, e) for e in entradas]


def guardar_modelo(red: QNetwork, ruta: str, cfg: ConfigEntrenamiento, costo_validacion: float) -> None:
    """Guarda pesos + lo necesario para reconstruir y trazar el modelo."""
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": red.state_dict(),
            "obs_dim": red.red[0].in_features,
            "n_acciones": red.red[-1].out_features,
            "oculta": cfg.oculta,
            "config": asdict(cfg),
            "costo_validacion": costo_validacion,
        },
        ruta,
    )


def cargar_modelo(ruta: str) -> tuple[QNetwork, dict]:
    """Reconstruye la red guardada. Devuelve (red, metadatos)."""
    datos = torch.load(ruta, map_location="cpu", weights_only=False)
    red = QNetwork(datos["obs_dim"], datos["n_acciones"], datos["oculta"])
    red.load_state_dict(datos["state_dict"])
    red.eval()
    return red, datos


def entrenar(
    env: TerrainLevelingEnv, cfg: ConfigEntrenamiento = ConfigEntrenamiento()
) -> tuple[QNetwork, dict[str, list]]:
    """Entrena el DQN; se queda con la mejor red según validación. Devuelve (red, historial)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    obs_dim = env.observation_space.shape[0]
    n_acciones = env.action_space.n

    red = QNetwork(obs_dim, n_acciones, cfg.oculta)
    red_objetivo = QNetwork(obs_dim, n_acciones, cfg.oculta)
    red_objetivo.load_state_dict(red.state_dict())
    optimizador = torch.optim.Adam(red.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.capacidad_buffer, obs_dim, n_acciones, seed=cfg.seed)

    historial: dict[str, list] = {"recompensa": [], "costo": [], "perdida": [], "validacion": []}
    mejor_costo = float("inf")
    mejores_pesos = {k: v.clone() for k, v in red.state_dict().items()}
    pasos = 0

    for episodio in range(cfg.episodios):
        obs, info = env.reset(seed=cfg.seed if episodio == 0 else None)
        epsilon = epsilon_en(episodio, cfg)
        terminado, recompensa, perdidas = False, 0.0, []

        while not terminado:
            accion = seleccionar_accion(red, obs, info["mascara_acciones"], epsilon, rng)
            obs_sig, reward, terminado, _, info_sig = env.step(accion)
            buffer.agregar(obs, accion, reward, obs_sig, terminado, info_sig["mascara_acciones"])
            obs, info = obs_sig, info_sig
            recompensa += reward
            pasos += 1

            if len(buffer) >= max(cfg.batch, cfg.empezar_a_aprender):
                perdidas.append(
                    paso_de_aprendizaje(red, red_objetivo, optimizador, buffer.muestrear(cfg.batch), cfg.gamma)
                )
            if pasos % cfg.sincronizar_cada == 0:
                red_objetivo.load_state_dict(red.state_dict())

        historial["recompensa"].append(recompensa)
        historial["costo"].append(env.costo_acumulado)
        historial["perdida"].append(float(np.mean(perdidas)) if perdidas else float("nan"))

        if (episodio + 1) % cfg.evaluar_cada == 0:
            costo_val = float(np.mean(evaluar_red(env, red, env.pool_validacion)))
            historial["validacion"].append((episodio + 1, costo_val))
            logger.info(
                "episodio=%d epsilon=%.3f costo_validacion=%.2f recompensa_media=%.3f",
                episodio + 1, epsilon, costo_val, float(np.mean(historial["recompensa"][-cfg.evaluar_cada:])),
            )
            if costo_val < mejor_costo:
                mejor_costo = costo_val
                mejores_pesos = {k: v.clone() for k, v in red.state_dict().items()}
                if cfg.ruta_modelo:
                    guardar_modelo(red, cfg.ruta_modelo, cfg, costo_val)

    red.load_state_dict(mejores_pesos)
    red.eval()
    historial["mejor_costo_validacion"] = [mejor_costo]
    return red, historial


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    entorno = TerrainLevelingEnv()
    red_final, hist = entrenar(entorno)
    costos_test = evaluar_red(entorno, red_final, entorno.pool_test)
    logger.info("costo medio en test: %.2f", float(np.mean(costos_test)))