import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuración visual de la página
st.title("⚽ Árbitro Asistente IA - Reglas IFAB")
st.write("Pregúntame cualquier duda sobre el reglamento del fútbol oficial 25/26.")

# 2. Carga y procesamiento del PDF
@st.cache_resource
def iniciar_base_de_conocimiento():
    # Asegúrate de que el archivo se llame exactamente 'reglas.pdf'
    loader = PyPDFLoader("reglas.pdf")
    docs = loader.load()
    
    # Fragmentamos el texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Creamos la base de datos vectorial usando el modelo local
    vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="llama3"))
    return vectorstore.as_retriever()

retriever = iniciar_base_de_conocimiento()

# 3. Configuración del Modelo Llama 3
llm = ChatOllama(model="llama3")

# Instrucciones estrictas para el bot
system_prompt = (
    "Eres un árbitro experto de la FIFA y un asistente especializado en las Reglas de Juego de la IFAB. "
    "Tu tarea es responder preguntas sobre el reglamento del fútbol basándote ÚNICAMENTE en el contexto proporcionado. "
    "Si la respuesta no está en el reglamento, di amablemente que tu conocimiento se limita a las reglas oficiales.\n\n"
    "Contexto:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Conexión del sistema RAG (Arquitectura moderna LCEL)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Interfaz interactiva
pregunta_usuario = st.text_input("Ingresa tu pregunta (ej: ¿Cuáles son las medidas de la cancha?):")

if st.button("Consultar Regla"):
    if pregunta_usuario:
        with st.spinner("Leyendo el reglamento..."):
            # Se invoca la cadena directamente con la pregunta
            respuesta = rag_chain.invoke(pregunta_usuario)
            st.success("¡Respuesta encontrada!")
            st.write(respuesta)
    else:
        st.warning("Por favor, escribe una pregunta antes de consultar.")