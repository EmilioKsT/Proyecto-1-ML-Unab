# 🏥 Predicción de GRD y Chatbot Clínico con IA

**CINF104 Machine Learning — 2026**  
Universidad Andrés Bello · Ingeniería Civil Informática · Santiago, Chile

> Pipeline completo de ML supervisado para predicción de Grupos Relacionados por el Diagnóstico (GRD) + Chatbot RAG con Llama 3 para consulta del reglamento clínico.

---

## 👥 Autores

| Nombre | GitHub |
|---|---|
| Emilio Castillo | [@EmilioKsT](https://github.com/EmilioKsT) |
| Alfonso Gysling | [@gyslinga-debug](https://github.com/gyslinga-debug) |
| Maximiliano Vargas | [@ACSIM24](https://github.com/ACSIM24) |
| Benjamín Zúñiga | [@zCoveFPS](https://github.com/zCoveFPS) |
| Ignacio Amar | [@jiac1](https://github.com/jiac1) |

---

## 📁 Estructura del Repositorio

```
Proyecto-1-ML-Unab/
├── Fase 1/
│   ├── ProyectoML_1_.ipynb        # Pipeline ML completo (preprocesamiento, modelos, SHAP)
│   └── Fase1_proyecto_ML.pdf      # Informe técnico formato IEEE
└── Fase 2/
    ├── app.py                     # Chatbot RAG con Streamlit + LangChain + Llama 3
    └── reglas.pdf                 # Base de conocimiento (reglamento clínico)
```

---

## 🔵 Fase 1 — Predicción Automatizada de GRD

### Descripción

Pipeline de machine learning supervisado que predice el código GRD de un episodio hospitalario a partir de variables clínicas estructuradas. Los GRD determinan el reembolso hospitalario en el sistema público FONASA, por lo que su asignación precisa es crítica.

**Dataset:** 14.561 episodios anonimizados del Hospital El Pino (Santiago, Chile).  
**Datos:** Códigos ICD-10 (diagnósticos), ICD-9-CM (procedimientos), edad y sexo.

### Pipeline

```
CSV Hospital El Pino
        │
        ▼
  Preprocesamiento
  ├── Extracción de prefijos ICD-10 / ICD-9-CM
  ├── Binarización de sexo (Hombre → 0, Mujer → 1)
  └── Filtro clases con < 5 muestras (526 → 327 clases, 97.1% registros retenidos)
        │
        ▼
  Feature Engineering
  ├── Multi-hot encoding: Top-100 diagnósticos + Top-50 procedimientos
  ├── Features numéricas: edad, total_diag, total_proc, sexo
  └── Matriz final: X ∈ ℝ^(N×154)
        │
        ▼
  Modelos
  ├── Random Forest  — T=300, class_weight='balanced'
  └── XGBoost        — T=300, max_depth=6, early stopping en iteración 75
        │
        ▼
  Evaluación + SHAP (TreeSHAP, 200 muestras de test)
```

### Resultados

| Modelo | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|
| Random Forest | **0.1661** | 0.3912 | 0.4100 |
| XGBoost | 0.1544 | **0.4285** | **0.4500** |

- **RF** obtiene mejor Macro F1 gracias a `class_weight='balanced'`, priorizando clases minoritarias.
- **XGBoost** logra mayor accuracy y Weighted F1 optimizando log-loss sin reponderación.

### Configuraciones Experimentales

| Config | K_diag | K_proc | D | Min. muestras | Clases | Árboles | RF Macro F1 | XGB Macro F1 |
|---|---|---|---|---|---|---|---|---|
| Run 1 (baseline) | 50 | 30 | 84 | 2 | 450 | 100 | 0.079 | 0.100 |
| **Run 2 (final)** | **100** | **50** | **154** | **5** | **327** | **300** | **0.166** | **0.154** |

### Predictores más relevantes (SHAP)

| Ranking | Feature | ϕ̄ |
|---|---|---|
| 1 | Edad en años | ≈ 0.22 |
| 2 | Total_Procedimientos | — |
| 3 | Total_Diagnosticos | — |
| 4–5 | PROC_89.26, PROC_99.21 | — |

### Comparación con Literatura

| Trabajo | Tarea | Clases | Método | Macro F1 |
|---|---|---|---|---|
| Mullenbach et al. (2018) | ICD-9 (texto) | 8.921 | Conv-Attn | 0.088 |
| Perotte et al. (2014) | ICD-9 (texto) | 5.031 | SVM-hier | 0.101 |
| **Este trabajo (RF)** | GRD (estructurado) | 327 | Random Forest | **0.166** |
| **Este trabajo (XGB)** | GRD (estructurado) | 327 | XGBoost | **0.154** |

### Instalación — Fase 1

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter
```

```bash
cd "Fase 1"
jupyter notebook ProyectoML_1_.ipynb
```

> ⚠️ El dataset no está incluido por acuerdo de confidencialidad con Hospital El Pino. Contactar a los autores para acceso académico.

---

## 🟢 Fase 2 — Chatbot RAG con Llama 3

### Descripción

Chatbot conversacional que responde preguntas sobre el reglamento clínico usando arquitectura **RAG (Retrieval-Augmented Generation)**. El sistema recupera fragmentos relevantes del PDF de reglas y los inyecta como contexto al modelo Llama 3, ejecutado localmente vía Ollama.

**Interfaz:** Streamlit  
**LLM:** Llama 3 (local via Ollama)  
**Base de conocimiento:** `reglas.pdf`  
**Stack:** LangChain · ChromaDB · Ollama Embeddings

### Arquitectura RAG

```
Usuario (pregunta)
        │
        ▼
  Streamlit UI (app.py)
        │
        ▼
  Retriever — ChromaDB
  ├── reglas.pdf → PyPDFLoader
  ├── RecursiveCharacterTextSplitter (chunk=1000, overlap=200)
  └── OllamaEmbeddings (llama3) → vectorstore
        │
        ▼
  ChatPromptTemplate
  ├── System: responde SOLO con el contexto del PDF
  └── Human: {pregunta_usuario}
        │
        ▼
  ChatOllama (llama3)
        │
        ▼
  StrOutputParser → respuesta en pantalla
```

### Instalación — Fase 2

**1. Instalar Ollama y descargar el modelo**

```bash
# Instalar Ollama desde: https://ollama.com
ollama pull llama3
```

**2. Instalar dependencias Python**

```bash
pip install streamlit langchain langchain-community langchain-chroma langchain-ollama pypdf chromadb
```

**3. Ubicar los archivos**

Asegurarse de que `reglas.pdf` esté en el mismo directorio que `app.py`.

**4. Ejecutar la aplicación**

```bash
cd "Fase 2"
streamlit run app.py
```

La app quedará disponible en `http://localhost:8501`.

### Uso

1. Abre el navegador en `http://localhost:8501`
2. Escribe tu pregunta en el campo de texto
3. Haz clic en **Consultar Regla**
4. El chatbot responde basándose únicamente en `reglas.pdf`

---

## 🛠️ Stack Tecnológico

| Tecnología | Uso |
|---|---|
| Python 3.10+ | Lenguaje base |
| scikit-learn | Random Forest |
| XGBoost | Gradient Boosting |
| SHAP | Explicabilidad (TreeSHAP) |
| Streamlit | Interfaz web chatbot |
| LangChain | Pipeline RAG |
| ChromaDB | Base vectorial |
| Ollama + Llama 3 | LLM local |
| Jupyter Notebook | Experimentación ML |

---

## ⚠️ Limitaciones

- Multi-hot encoding descarta el orden de los códigos clínicos.
- Vocabulario Top-K excluye el 75% de los códigos únicos del dataset.
- El 38% de clases GRD filtradas por insuficiencia de muestras.
- Split único 80/20 sin validación cruzada estratificada.
- El chatbot requiere Ollama corriendo localmente con Llama 3 descargado.

---

## 📄 Licencia

Uso académico. Dataset bajo acuerdo de investigación con Hospital El Pino, Santiago, Chile.

---

*Proyecto desarrollado para el curso CINF104 Machine Learning, Primer Semestre 2026 — Universidad Andrés Bello.*
