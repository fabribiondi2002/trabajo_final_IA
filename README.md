# Agente de RL para Ruteo de Nivelación de Terreno

Trabajo Final — Inteligencia Artificial — IUA (2026)
Modalidad: Individual — Vía Proyecto Funcional
Categoría tecnológica: Aprendizaje por Refuerzos (RL)

## Setup del entorno

```bash
python3 -m venv venv
source venv/bin/activate   # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Correr los tests

```bash
pytest tests/ -v
```

## Estructura del repo

```
/data        -> terrenos generados (gitignored, no versionado)
/notebooks   -> exploración y análisis
/src         -> código fuente modularizado (generador, asignación, entorno, agente)
/api         -> API REST (FastAPI) para servir el modelo entrenado
/models      -> checkpoints de agentes entrenados (gitignored, no versionado)
/tests       -> tests automatizados (pytest)
```

## Estado actual

- [x] Estructura base del repo
- [x] Módulo de asignación de volúmenes (transportation problem / min-cost flow) — `src/allocation.py`
- [ ] Generador de terrenos sintéticos
- [ ] Entorno Gymnasium custom
- [ ] Agente RL (Q-learning tabular)
- [ ] Baseline heurístico
- [ ] Evaluación y métricas
- [ ] API REST + Docker
- [ ] Publicación en GitHub Pages
