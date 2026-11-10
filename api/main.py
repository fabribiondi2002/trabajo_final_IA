"""
API REST del proyecto: dado un terreno (por su semilla), devuelve el orden de viajes que elige el agente
DQN entrenado, comparado con el del greedy.

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Variable de entorno:
    MODELO_PATH   ruta del modelo entrenado (por defecto modelo/modelo_final.pt)
"""
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Path, Request
from pydantic import BaseModel, Field

from src.env import crear_entorno
from src.inferencia import TerrenoNoResoluble, describir_terreno, resolver_terreno
from src.train import cargar_modelo

RUTA_MODELO_POR_DEFECTO = "modelo/modelo_final.pt"
SEED_MAXIMA = 999_999

logger = logging.getLogger("api")


def registrar(evento: str, **datos) -> None:
    """Escribe un evento como una línea JSON."""
    logger.info(json.dumps({"evento": evento, **datos}, ensure_ascii=False))


class SolicitudResolver(BaseModel):
    seed: int = Field(ge=0, le=SEED_MAXIMA, description="Semilla del terreno a resolver", examples=[7])


class Paso(BaseModel):
    paso: int
    viaje: int
    origen: list[int]
    destino: list[int]
    volumen: int
    distancia: int


class Recorrido(BaseModel):
    costo: float
    orden: list[Paso]


class Terreno(BaseModel):
    tamano: int
    alturas: list[list[int]] = Field(description="Altura actual de cada celda (fila = x, columna = y)")
    objetivo: list[list[int]] = Field(description="Altura objetivo de cada celda; alturas > objetivo se excavan, < objetivo se rellenan")


class RespuestaTerreno(BaseModel):
    seed: int
    terreno: Terreno
    n_viajes: int | None = Field(description="Cantidad de viajes que genera el terreno; null si excede el máximo del modelo")
    resoluble: bool = Field(description="False si el terreno necesita más viajes de los que soporta el modelo")


class RespuestaResolver(BaseModel):
    seed: int
    terreno: Terreno
    n_viajes: int
    agente: Recorrido
    greedy: Recorrido
    mejora_vs_greedy_pct: float


def create_app(ruta_modelo: str | None = None) -> FastAPI:
    """Crea la app. El modelo se carga al arrancar. Si no se puede cargar, la app no arranca."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        ruta = ruta_modelo or os.environ.get("MODELO_PATH", RUTA_MODELO_POR_DEFECTO)
        if not os.path.exists(ruta):
            raise RuntimeError(f"No se encontró el modelo en '{ruta}'. Definí MODELO_PATH con una ruta válida.")
        red, meta = cargar_modelo(ruta)
        app.state.red = red
        app.state.meta = meta
        app.state.ruta = ruta
        app.state.env = crear_entorno(meta.get("entorno"), tamano_pool=2)
        registrar("modelo_cargado", ruta=ruta, arquitectura=meta["config"].get("arquitectura", "plana"))
        yield

    app = FastAPI(
        title="Nivelación de terreno con DQN",
        description="Orden de viajes del agente DQN comparado con el greedy.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        """Responde si la API está viva y con el modelo cargado."""
        return {"estado": "ok"}

    @app.get("/modelo")
    def modelo(request: Request):
        """Datos del modelo cargado: configuración de entrenamiento y entorno."""
        meta = request.app.state.meta
        return {
            "ruta": request.app.state.ruta,
            "config": meta["config"],
            "entorno": meta.get("entorno"),
            "costo_validacion": meta.get("costo_validacion"),
        }

    @app.get("/terreno/{seed}", response_model=RespuestaTerreno)
    def terreno(seed: int = Path(ge=0, le=SEED_MAXIMA, description="Semilla del terreno", examples=[7])):
        """Devuelve el terreno de la semilla (alturas y objetivo) sin resolverlo, para elegir qué resolver."""
        return describir_terreno(app.state.env, seed)

    @app.post("/resolver", response_model=RespuestaResolver)
    def resolver(solicitud: SolicitudResolver, request: Request):
        """Resuelve el terreno de la semilla y devuelve el terreno, el orden del agente y del greedy con sus costos."""
        inicio = time.perf_counter()
        try:
            resultado = resolver_terreno(request.app.state.env, request.app.state.red, solicitud.seed)
        except TerrenoNoResoluble as error:
            registrar("terreno_no_resoluble", seed=solicitud.seed)
            raise HTTPException(status_code=422, detail=str(error)) from error
        registrar(
            "terreno_resuelto",
            seed=solicitud.seed,
            costo_agente=resultado["agente"]["costo"],
            costo_greedy=resultado["greedy"]["costo"],
            milisegundos=round(1000 * (time.perf_counter() - inicio), 1),
        )
        return resultado

    return app


app = create_app()