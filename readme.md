# Proyecto 1: Machine Learning I - UNAB

Este repositorio contiene el desarrollo del Proyecto 1 para la asignatura de Machine Learning I, el cual está dividido en dos fases principales que abordan distintos desafíos utilizando técnicas de Inteligencia Artificial.

## Estructura del Proyecto

### Fase 1: Predicción de Grupos Relacionados por el Diagnóstico (GRD)
En esta fase, desarrollamos un pipeline de machine learning supervisado para procesar y clasificar datos de codificación clínica.
- **Modelos utilizados:** Random Forest y XGBoost.
- **Técnicas aplicadas:** Limpieza de datos estructurados, normalización y análisis de explicabilidad del modelo utilizando **SHAP** (TreeSHAP) para identificar las características y predictores más relevantes en la clasificación de códigos GRD.

### Fase 2: Asistente RAG - Reglas de Juego de la IFAB (Fútbol)
En esta fase, implementamos una aplicación de Inteligencia Artificial Generativa basada en la arquitectura Retrieval-Augmented Generation (RAG).
- **Objetivo:** Crear un chatbot asistente (experto de la FIFA) que responda dudas basándose estrictamente en el reglamento oficial de fútbol de la IFAB 2025/2026.
- **Stack Tecnológico:** - **Framework de LLM:** LangChain.
  - **Base de Datos Vectorial:** ChromaDB (para almacenar los embeddings del reglamento en PDF).
  - **Modelo:** Llama 3 (ejecutado localmente mediante Ollama).
  - **Interfaz de Usuario:** Streamlit para proveer una plataforma web interactiva (`app.py`).

## Equipo de Desarrollo
Proyecto desarrollado de forma colaborativa por un equipo de 5 estudiantes de la Universidad Andrés Bello.
