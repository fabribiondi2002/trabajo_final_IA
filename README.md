# Nivelación de terreno con DQN

Agente de aprendizaje por refuerzo (DQN) que decide en qué orden hacer los viajes de extracción y relleno
para nivelar un terreno agrícola, minimizando la distancia total recorrida por la máquina. Proyecto final de
Inteligencia Artificial (IUA).

## Contenido

1. [Demo](#demo)
2. [Autores y contexto académico](#autores-y-contexto-académico)
3. [Qué hace el proyecto](#qué-hace-el-proyecto)
4. [Definición PEAS](#definición-peas)
5. [Requisitos e instalación](#requisitos-e-instalación)
6. [Cómo ejecutarlo](#cómo-ejecutarlo)
7. [Resultados](#resultados)
8. [Análisis: el costo de validación empeora al entrenar](#análisis-el-costo-de-validación-empeora-al-entrenar)
9. [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
10. [Estructura del repositorio](#estructura-del-repositorio)
11. [API y Docker](#api-y-docker)
12. [Licencia](#licencia)

## Demo

`notebooks/demo.ipynb` es una demo interactiva que usa la API. Permite elegir una seed, ver el terreno que genera
y enviarlo a resolver para comparar al agente DQN contra el greedy.

1. Levantar la API (ver [API y Docker](#api-y-docker)).
2. Instalar lo que necesita el notebook (además de `requirements.txt`):

   ```bash
   pip install -r requirements-notebook.txt
   ```

3. Abrir `notebooks/demo.ipynb` (Jupyter o VS Code) y ejecutar las celdas en orden.

Qué se puede hacer en el notebook:

- **Elegir una seed y ver el terreno** en 2D (mapa de calor de alturas actuales, objetivo y diferencia) o en 3D
  (relieve actual y objetivo lado a lado, con sliders de elevación y giro).
- **Enviarlo a resolver** y ver el costo del greedy, el del agente y cuánto mejora o empeora el agente.
- **Comparación:** costo total separado en tramo con carga (igual en ambos, porque los viajes son los mismos)
  y traslado en vacío (lo único que cambia con el orden), más la curva de costo acumulado.
- **Animación paso a paso:** la máquina, el viaje actual en naranja, los anteriores en gris y el terreno
  poniéndose en cero, con la curva de costo acumulado.
- **Tabla de viajes** con origen, destino, volumen y distancia de cada paso.
- **Varias seeds:** tabla y gráfico de las seeds 0 a 9.

Si la API no está en `localhost:8000`, definir la variable de entorno `API_URL` antes de abrir el notebook.

Además, `graficos/comparacion.png` muestra las curvas de validación y el costo en test de las variantes
entrenadas.

## Autores y contexto académico

- **Autor:** Leandro Fabrizio Biondi (modalidad individual).
- **Materia:** Inteligencia Artificial, Instituto Universitario Aeronáutico (IUA), Córdoba, Argentina. Año 2026.
- **Docente:** Ing. Hernando Alexis González.
- **Instancia:** Proyecto Funcional. Ejes elegidos: *Arquitectura y Despliegue* (API REST con FastAPI en Docker) y
  *MLOps y Calidad de Código* (tests con pytest, logging y trazabilidad de experimentos).

## Qué hace el proyecto

1. Genera terrenos sintéticos y resuelve qué zonas altas rellenan qué zonas bajas (problema de transporte).
2. Convierte esos flujos en una lista de viajes según la capacidad de la máquina.
3. Un agente DQN elige el siguiente viaje a realizar. Se compara contra:
   - **Greedy**: siempre va al viaje cuyo origen está más cerca de la máquina (baseline del proyecto, sin aprendizaje).
   - **Referencia**: el mejor orden que encuentra una búsqueda local.

La métrica principal es el costo total (distancia recorrida, menor es mejor) y la **brecha cerrada**:

```
brecha cerrada = (costo greedy − costo agente) / (costo greedy − costo referencia)
```

0% es igualar al greedy y 100% es llegar a la referencia. Se mide siempre sobre terrenos de test que el agente
nunca vio.

### Metodología

- Los terrenos se generan a partir de rangos de semillas **sin superposición**: test (0–99), validación (100–149)
  y entrenamiento (150 en adelante).
- Durante el entrenamiento se evalúa en validación y se guarda el mejor modelo según ese costo. El test se mide una
  sola vez, con el modelo guardado.
- Cada configuración se entrena con 5 seeds y se reporta media y desvío estándar.
- El gamma final se eligió por costo de validación, no por test (ver la tabla de resultados).
- El entorno con el que se entrenó cada modelo queda guardado junto al modelo y al historial, para poder
  reproducir y evaluar cada experimento.

## Definición PEAS

| Elemento | Descripción |
|---|---|
| **Performance** | Minimizar la distancia total recorrida por la máquina para completar todos los viajes. El volumen total de tierra a mover lo fija el problema de transporte y es el mismo para cualquier orden, por lo que lo único que el agente puede optimizar es la distancia. Se mide con el costo en test y la brecha cerrada respecto del greedy y la referencia. |
| **Environment** | Grilla de N×N celdas (8×8 por defecto) con altura actual y objetivo, y lista de viajes pendientes (cada flujo origen-destino-volumen partido en cargas de tamaño `capacidad_maquina`). Se modela la capa fértil de cada celda (in situ / extraída / repuesta) y la posición de la máquina. Es totalmente observable, determinístico dado el terreno, secuencial, estático, discreto y con un único agente. |
| **Actuators** | Elegir el próximo viaje pendiente (acción discreta, hasta 50 viajes). El desplazamiento de la máquina y la extracción y reposición de la capa fértil se calculan automáticamente por el entorno. |
| **Sensors** | Lo que ve el agente es un vector: la posición de la máquina y, por cada viaje, su origen (x, y), su destino (x, y), su volumen y la distancia de la máquina al origen. Con la opción `--lookahead` se agregan 2 datos por viaje: la distancia de su destino al origen más cercano entre los otros viajes pendientes y al centroide de esos orígenes. Los viajes ya completados se enmascaran. |

## Requisitos e instalación

- Python 3.10 o superior (se usa la sintaxis `int | None`). Probado con Python 3.12.
- Dependencias principales: `torch`, `numpy`, `scipy`, `gymnasium`, `networkx`, `matplotlib`, `pytest`, y para la API
  `fastapi`, `uvicorn` y `pydantic` (ver `requirements.txt`, con versiones fijadas).
- El ruido Perlin que genera los terrenos está implementado en `src/ruido.py`, sin dependencias externas ni
  compilación de código C. Da los mismos terrenos que el paquete `noise`.
- Solo para la demo: `requirements-notebook.txt` (ver [Demo](#demo)).

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

La configuración de entrenamiento se pasa por argumentos de línea de comandos (ver abajo) o por la clase
`ConfigEntrenamiento` en `src/train.py`. Las variables de entorno de la API y la demo están en
[Variables de entorno](#variables-de-entorno).

## Cómo ejecutarlo

Todos los comandos se corren desde la raíz del repositorio con el entorno virtual activado.

### Entrenar con varias seeds y medir en test

```bash
python -m src.experiment --arquitectura por_viaje --recompensa vacio --lookahead --lr 3e-4 --peso-distancia 10 --gamma 0.39 --carpeta models/final
```

Entrena con 5 seeds (0 a 4), guarda en la carpeta el mejor modelo y el historial de cada seed, y muestra el costo
en test comparado con el greedy y la referencia.

| Argumento | Descripción | Por defecto |
|---|---|---|
| `--seeds` | seeds a entrenar | `0 1 2 3 4` |
| `--carpeta` | dónde se guardan modelos e historiales | `models/base` |
| `--arquitectura` | `plana` o `por_viaje` | `plana` |
| `--episodios` | episodios de entrenamiento | `1000` |
| `--oculta` | neuronas de las capas ocultas | `32` |
| `--lr` | learning rate | `3e-4` |
| `--gamma` | factor de descuento | `0.39` |
| `--evaluar-cada` | episodios entre evaluaciones en validación | `50` |
| `--peso-distancia` | escala del reward (parámetro del entorno) | `1.0` |
| `--recompensa` | `total` o `vacio` (solo castiga el tramo en vacío) | `total` |
| `--lookahead` | agrega 2 datos por viaje sobre lo que queda después de hacerlo | desactivado |
| `--doble` | usa Double DQN | desactivado |
| `--paciencia` | corta si la validación no mejora en N evaluaciones seguidas | sin corte |

### Reproducir la tabla de resultados

```bash
# 1. Base: red plana
python -m src.experiment --lr 3e-4 --peso-distancia 10 --gamma 0.99 --carpeta models/base

# 2. Red por viaje
python -m src.experiment --lr 3e-4 --peso-distancia 10 --gamma 0.99 \
  --arquitectura por_viaje --carpeta models/base_por_viaje

# 3. + reward solo del tramo en vacío
python -m src.experiment --lr 3e-4 --peso-distancia 10 --gamma 0.99 \
  --arquitectura por_viaje --recompensa vacio --carpeta models/base_vacio

# 4. + lookahead (gamma 0.99)
python -m src.experiment --lr 3e-4 --peso-distancia 10 --gamma 0.99 \
  --arquitectura por_viaje --recompensa vacio --lookahead --carpeta models/lookahead

# 5. Control: gamma 0.5 sin lookahead (con paciencia 4)
python -m src.experiment --lr 3e-4 --peso-distancia 10 --gamma 0.5 \
  --arquitectura por_viaje --recompensa vacio --paciencia 4 --carpeta models/g05_paciencia

# 6. Lookahead + gamma 0.5
python -m src.experiment --lr 3e-4 --peso-distancia 10 --gamma 0.5 \
  --arquitectura por_viaje --recompensa vacio --lookahead --carpeta models/lookahead_g05

# 7. Lookahead + gamma 0.39 (con paciencia 4)
python -m src.experiment --lr 3e-4 --peso-distancia 10 --gamma 0.39 \
  --arquitectura por_viaje --recompensa vacio --lookahead --paciencia 4 --carpeta models/lookahead_g039_paciencia

# 8. Lookahead + gamma 0.39 + Double DQN
python -m src.experiment --lr 3e-4 --peso-distancia 10 --gamma 0.39 \
  --arquitectura por_viaje --recompensa vacio --lookahead --doble --carpeta models/lookahead_g039_doble
```

Cada corrida de 5 seeds tarda entre 3 y 25 minutos según la configuración y la computadora.

### Evaluar un modelo guardado

```bash
python -m src.evaluation --modelo models/lookahead_g039_paciencia/modelo_seed_0.pt
```

### Comparar variantes en un gráfico

```bash
python -m src.comparison "por viaje=models/base_por_viaje" "vacío=models/base_vacio" \
  "lookahead γ=0.99=models/lookahead" "lookahead γ=0.5=models/lookahead_g05" \
  "lookahead γ=0.39=models/lookahead_g039_paciencia" --salida graficos/comparacion.png
```

Con `--sin-referencia` se omite el cálculo de la referencia y es más rápido.

## Resultados

Medidos en 100 terrenos de test, 5 seeds por variante. **Greedy: 355.44. Referencia: 348.15.**
Todas las filas usan `--lr 3e-4 --peso-distancia 10` y 1000 episodios.

| # | Variante | Arquitectura | Reward | Lookahead | Gamma | Costo en test | Brecha cerrada |
|---|---|---|---|---|---|---|---|
| 1 | Base | plana | total | No | 0.99 | 386.74 ± 1.89 | −429.4% ± 25.9% |
| 2 | Red por viaje | por viaje | total | No | 0.99 | 356.52 ± 0.70 | −14.8% ± 9.6% |
| 3 | Reward en vacío | por viaje | vacío | No | 0.99 | 355.28 ± 0.38 | 2.2% ± 5.2% |
| 4 | Lookahead | por viaje | vacío | Sí | 0.99 | 356.22 ± 0.57 | −10.7% ± 7.9% |
| 5 | Gamma 0.5, sin lookahead | por viaje | vacío | No | 0.5 | 355.29 ± 0.23 | 2.1% ± 3.1% |
| 6 | Lookahead + gamma 0.5 | por viaje | vacío | Sí | 0.5 | 353.66 ± 0.14 | 24.4% ± 1.9% |
| 7 | Lookahead + gamma 0.39 | por viaje | vacío | Sí | 0.39 | **353.31 ± 0.13** | **29.2% ± 1.7%** |
| 8 | Lookahead + gamma 0.39 + Double DQN | por viaje | vacío | Sí | 0.39 | 353.32 ± 0.10 | 29.1% ± 1.3% |

Notas sobre la tabla:

- La fila 5 se corrió con `--paciencia 4` y la 7 también. La parada anticipada no cambia el modelo guardado
  (el mejor checkpoint ya estaba guardado antes de cortar) y solo ahorra tiempo.
- **Gamma elegido por validación.** El costo de validación medio fue 361.31 (γ=0.5), 361.02 (γ=0.35),
  360.99 (γ=0.4) y 360.98 (γ=0.39). Entre 0.35 y 0.4 la diferencia es menor que el ruido: hay una meseta, no un
  óptimo preciso.
- **Ni el lookahead ni el gamma bajo alcanzan por separado.** Con gamma 0.99 el lookahead empeora (fila 4) y
  con gamma 0.5 sin lookahead el agente queda igual que el greedy (fila 5). Solo la combinación supera al greedy.
- **Double DQN no cambia el resultado** (fila 8 contra fila 7: diferencia de 0.01, muy por debajo del desvío).

### Meta del Project Charter

El Charter fijó como meta una reducción de costo de al menos 20% respecto del greedy, con la aclaración de que el
valor se ajustaría tras las primeras corridas exploratorias. Esas corridas mostraron que la meta no era alcanzable:
la mejor solución conocida (la referencia) reduce el costo apenas un **2.05%** respecto del greedy, de modo que ni
siquiera el óptimo conocido llega al 20%. El agente reduce el costo un **0.60%** (355.44 → 353.31). Por eso la
meta se redefinió en términos de la brecha cerrada, que mide qué parte de lo mejorable se logra: el agente cierra
el **29.2%** de la distancia entre el greedy y la referencia.

## Análisis: el costo de validación empeora al entrenar

**Qué se observa.** Con gamma 0.99, en todas las variantes el costo de validación *empeora* a medida que avanza el
entrenamiento, y el mejor modelo aparece alrededor del episodio 100. Por ejemplo, para la red por viaje con reward
en vacío (seed 0) la validación pasa de 362.9 en el episodio 100 a 372.6 en el 1000, mientras que el greedy
obtiene 362.86 en el mismo conjunto. Esto explica por qué la parada anticipada da exactamente el mismo resultado
que el entrenamiento completo: el modelo guardado ya es el del episodio ~100, y cortar solo ahorra tiempo
(por ejemplo, de 12.4 a 5.1 minutos en la variante con lookahead).

**Qué se probó.**

| Hipótesis | Experimento | Resultado |
|---|---|---|
| El reward es ruidoso | Reward solo del tramo en vacío (el tramo cargado suma lo mismo para cualquier orden) | Mejora el desvío entre seeds (0.70 → 0.38) pero la validación sigue empeorando |
| Sobreestimación del máximo | Double DQN | Sin efecto (353.32 contra 353.31) |
| Horizonte demasiado largo | Gamma 0.5 y menores | **Estabiliza la validación**, queda plana en torno a 361 durante todo el entrenamiento |
| Falta de información sobre lo que queda por hacer | Lookahead | Solo ayuda combinado con gamma bajo |

**Interpretación.** Con gamma 0.99 el objetivo de cada paso depende del valor de los ~50 viajes siguientes, una
estimación que se apoya en sí misma y acumula error. En este problema la diferencia entre un buen y un mal orden es
de apenas ~2% del costo, así que ese ruido tapa la señal y el entrenamiento se aleja de la política inicial, que ya
era razonable porque la feature de distancia al origen hace que el greedy sea un buen punto de partida. Con gamma
bajo el objetivo mira pocos pasos adelante y el aprendizaje es estable. El lookahead aporta la información de lo que
queda después de cada viaje, y el gamma bajo permite aprovecharla sin inestabilidad, por eso solo la combinación
supera al greedy. Esta explicación es consistente con todos los experimentos de la tabla, pero es una interpretación,
no se probó de forma aislada qué parte del efecto se debe al horizonte y cuál al ruido del objetivo.

## Limitaciones

**Limitaciones**

- **El agente supera al greedy del proyecto, pero el margen es chico.** La referencia reduce el costo solo ~2%
  respecto del greedy, y el agente cierra algo menos de un tercio de ese margen.
- **Instancias chicas y sintéticas:** terrenos de 8×8 con hasta 50 viajes.
- **Reproducibilidad entre computadoras:** los resultados pueden variar levemente entre máquinas por diferencias
  numéricas (por ejemplo, la fila 1 dio 386.31 en una computadora y 386.74 en otra), sin cambiar las conclusiones.

## Estructura del repositorio

```
api/
  main.py                API REST (FastAPI)
src/
  terrain_generator.py   generación de terrenos
  ruido.py               ruido Perlin (reemplaza al paquete noise)
  allocation.py          problema de transporte (qué zona alta rellena a cuál baja)
  trip_builder.py        construcción de los viajes
  env.py                 entorno Gymnasium
  agent.py               redes Q, replay buffer y paso de aprendizaje (DQN y Double DQN)
  train.py               entrenamiento DQN
  baseline.py            heurística greedy
  reference.py           referencia por búsqueda local
  evaluation.py          evaluación contra greedy y referencia
  inferencia.py          resolver un terreno con un modelo entrenado (lo usa la API)
  experiment.py          entrenamiento con varias seeds
  comparison.py          gráfico comparativo de variantes
tests/                   tests con pytest
notebooks/demo.ipynb     demo interactiva (usa la API)
modelo/modelo_final.pt   modelo que carga la API (y la imagen de Docker)
models/                  modelos e historiales de los experimentos (no se versionan)
graficos/                gráficos generados
Dockerfile               imagen de la API
requirements.txt         dependencias (versiones fijadas)
requirements-notebook.txt  dependencias extra de la demo
```

## API y Docker

La API (FastAPI) carga un modelo entrenado al arrancar y expone el agente por HTTP. El código está en `api/main.py`
y la lógica de inferencia en `src/inferencia.py`.

### Endpoints

| Método y ruta | Qué hace |
|---|---|
| `GET /health` | Indica si la API está viva. |
| `GET /modelo` | Datos del modelo cargado: ruta, configuración de entrenamiento, entorno y costo de validación. |
| `GET /terreno/{seed}` | Terreno de la seed (`alturas` y `objetivo`), cantidad de viajes y si el modelo puede resolverlo. No lo resuelve. |
| `POST /resolver` | Recibe `{"seed": 7}` y devuelve el terreno, el orden de viajes del agente y del greedy con sus costos, y la mejora del agente sobre el greedy. |

La seed va de 0 a 999999. Una seed inválida devuelve 422, y también un terreno que necesita más viajes
que el máximo del modelo (`GET /terreno/{seed}` lo avisa antes con `resoluble: false`).
FastAPI genera además una página interactiva en `http://localhost:8000/docs`.

### Con Docker

El modelo que usa la imagen está en `modelo/modelo_final.pt`.

```bash
docker build -t nivelacion-api .
docker run -p 8000:8000 nivelacion-api
```

Probar:

```bash
curl localhost:8000/health
curl localhost:8000/terreno/7
curl -X POST localhost:8000/resolver -H "Content-Type: application/json" -d '{"seed": 7}'
```

Para usar otro modelo sin reconstruir la imagen, montar la carpeta y apuntar `MODELO_PATH` a él:

```bash
docker run -p 8000:8000 -v "$(pwd)/models/mi_experimento:/app/otro" \
  -e MODELO_PATH=/app/otro/modelo_seed_0.pt nivelacion-api
```

### Sin Docker

```bash
uvicorn api.main:app --port 8000
```

### Logs

Cada pedido deja una línea JSON en la salida estándar (`modelo_cargado`, `terreno_resuelto` con costos y
milisegundos, `terreno_no_resoluble`).

### Variables de entorno

| Variable | Qué hace | Por defecto |
|---|---|---|
| `MODELO_PATH` | Ruta del modelo entrenado que carga la API al arrancar. Si el archivo no existe, la API no arranca. | `modelo/modelo_final.pt` (`/app/modelo/modelo_final.pt` en Docker) |
| `API_URL` | Dirección de la API que usa `notebooks/demo.ipynb`. | `http://localhost:8000` |

El entrenamiento y los experimentos no usan variables de entorno: se configuran por argumentos de línea de comandos
o por `ConfigEntrenamiento` en `src/train.py`.

## Licencia

Distribuido bajo la licencia MIT. Ver el archivo [LICENSE](LICENSE).