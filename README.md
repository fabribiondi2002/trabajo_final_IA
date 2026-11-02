# Nivelación de terreno con DQN

Agente de aprendizaje por refuerzo (DQN) que decide en qué orden hacer los viajes de extracción y relleno
para nivelar un terreno, minimizando la distancia total recorrida por la máquina. Proyecto final de
Inteligencia Artificial (IUA).

## Qué hace el proyecto

1. Genera terrenos sintéticos y resuelve qué zonas altas rellenan qué zonas bajas (problema de transporte).
2. Convierte esos flujos en una lista de viajes según la capacidad de la máquina.
3. Un agente DQN elige el siguiente viaje a realizar. Se compara contra:
   - **Greedy**: siempre va al viaje cuyo origen está más cerca de la máquina.
   - **Referencia**: el mejor orden que encuentra una búsqueda local.

La métrica principal es el costo total (distancia recorrida, menor es mejor) y la **brecha cerrada**:
`(costo greedy − costo agente) / (costo greedy − costo referencia)`. 0% es igualar al greedy y 100% es llegar a la referencia.

## Requisitos

- Python 3.10 o superior (se usa la sintaxis `int | None`). Probado con Python 3.12.
- Dependencias principales: `torch`, `numpy`, `gymnasium`, `networkx`, `noise`, `matplotlib` y `pytest`.
- El paquete `noise` compila una extensión en C: en Linux y macOS puede hacer falta un compilador
  (`build-essential` / Xcode Command Line Tools), y en Windows las Build Tools de Visual Studio.

## Instalación

```bash
git clone https://github.com/fabribiondi2002/trabajo_final_IA.git
cd trabajo_final_IA

python -m venv venv
source venv/bin/activate          # En Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Para verificar que todo quedó bien:

```bash
python -m pytest tests -q
```

## Variables de entorno

El proyecto actualmente no requiere variables de entorno. Toda la configuración se pasa por argumentos de línea de comandos
(ver más abajo) o por la clase `ConfigEntrenamiento` en `src/train.py`.

## Cómo ejecutarlo

Todos los comandos se corren desde la raíz del repositorio con el entorno virtual activado.

### Entrenar con varias seeds y medir en test

```bash
python -m src.experiment --carpeta models/base
```

Entrena con 5 seeds (0 a 4), guarda en la carpeta el mejor modelo y el historial de cada seed, y
muestra el costo en test comparado con el greedy y la referencia.

Argumentos principales:

| Argumento | Descripción | Por defecto |
|---|---|---|
| `--seeds` | seeds a entrenar | `0 1 2 3 4` |
| `--carpeta` | dónde se guardan modelos e historiales | `models/base` |
| `--arquitectura` | `plana` o `por_viaje` | `plana` |
| `--episodios` | episodios de entrenamiento | `3000` |
| `--oculta` | neuronas de las capas ocultas | `128` |
| `--lr` | learning rate | `1e-3` |
| `--evaluar-cada` | episodios entre evaluaciones en validación | `100` |
| `--peso-distancia` | escala del reward (parámetro del entorno) | `1.0` |
| `--recompensa` | `total` o `vacio` (solo castiga el tramo en vacío) | `total` |
| `--paciencia` | corta si la validación no mejora en N evaluaciones seguidas | sin corte |

### Las 4 configuraciones comparadas

```bash
# 1. Base: red plana, reward total
python -m src.experiment --carpeta models/base

# 2. Red por viaje, reward total x10
python -m src.experiment --arquitectura por_viaje --oculta 32 --lr 3e-4 --peso-distancia 10 \
  --evaluar-cada 50 --episodios 1000 --carpeta models/por_viaje_x10

# 3. Red por viaje + reward solo del tramo en vacío
python -m src.experiment --arquitectura por_viaje --oculta 32 --lr 3e-4 --peso-distancia 10 \
  --recompensa vacio --evaluar-cada 50 --episodios 1000 --carpeta models/por_viaje_vacio_completo

# 4. Lo mismo que la 3, con parada anticipada
python -m src.experiment --arquitectura por_viaje --oculta 32 --lr 3e-4 --peso-distancia 10 \
  --recompensa vacio --paciencia 4 --evaluar-cada 50 --episodios 1000 --carpeta models/por_viaje_vacio
```

### Evaluar un modelo guardado

```bash
python -m src.evaluation --modelo models/por_viaje_vacio/modelo_seed_0.pt
```

### Comparar las variantes en un gráfico

```bash
python -m src.comparison "base=models/base" "por viaje=models/por_viaje_x10" \
  "vacío sin paciencia=models/por_viaje_vacio_completo" "vacío + paciencia=models/por_viaje_vacio" \
  --salida graficos/comparacion.png
```

Con `--sin-referencia` se omite el cálculo de la referencia y es más rápido.

### Un solo entrenamiento con la configuración por defecto

```bash
python -m src.train
```

## Resultados

Medidos en 100 terrenos de test, 5 seeds por variante. Greedy: 355.44. Referencia: 348.15.

| Variante | Costo en test | Brecha cerrada |
|---|---|---|
| Base (red plana) | 385.20 ± 3.16 | −408% |
| Red por viaje (reward x10) | 356.52 ± 0.70 | −15% |
| Red por viaje + reward de vacío | 355.28 ± 0.38 | +2.2% ± 5.2% |
| Anterior + parada anticipada | 355.28 ± 0.38 | +2.2% ± 5.2% |

El agente empata con el greedy pero no lo supera: el greedy ya está a ~2% del mejor orden conocido, así que
el margen de mejora es chico. Ver `graficos/comparacion.png`.

## Estructura del repositorio

```
src/
  terrain_generator.py   generación de terrenos
  allocation.py          problema de transporte (qué zona alta rellena a cuál baja)
  trip_builder.py        construcción de los viajes
  env.py                 entorno Gymnasium
  agent.py               redes Q, replay buffer y paso de aprendizaje
  train.py               entrenamiento DQN
  baseline.py            heurística greedy
  reference.py           referencia por búsqueda local
  evaluation.py          evaluación contra greedy y referencia
  experiment.py          entrenamiento con varias seeds
  comparison.py          gráfico comparativo de variantes
tests/                   tests con pytest
models/                  modelos e historiales guardados
graficos/                gráficos generados
```

## API y Docker

Pendiente. Se documentará acá cuando esté implementada.