# app.py
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# ===== Config =====
PERSIST_DIR = "data/faiss_index"  # carpeta con index.faiss, index.pkl, docstore.pkl, meta.txt
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # DEBE ser el MISMO que usaste al construir

st.set_page_config(page_title="Agentic RAG", page_icon="🧠")
st.title("🧠 Agentic RAG")
#st.caption("Ingresa una pregunta")

@st.cache_resource
def load_vectorstore():
    # Verifica que existan los archivos del índice (formato actual)
    needed = ["index.faiss", "index.pkl"]
    missing = [f for f in needed if not os.path.exists(os.path.join(PERSIST_DIR, f))]
    if missing:
        st.error(
            "❌ Faltan archivos del índice FAISS en "
            f"`{PERSIST_DIR}`: {', '.join(missing)}\n\n"
            "¿Ejecutaste `python build_index.py` en esta carpeta y subiste `data/faiss_index/`?"
        )
        # (Opcional) muestra lo que hay en la carpeta para depurar:
        if os.path.isdir(PERSIST_DIR):
            st.write("📂 Contenido de la carpeta:", os.listdir(PERSIST_DIR))
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Carga del índice (pickle requiere este flag)
    vs = FAISS.load_local(
        PERSIST_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vs


def answer(question: str, vectorstore: FAISS) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})  # antes era 2 o 3 quizás
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return "No encuentro pasajes relevantes en los documentos cargados."

    context = "\n\n".join(d.page_content for d in docs)

    # 🔹 Prompt actualizado: fuerza al modelo a responder SOLO con contexto
    prompt = f"""
Eres un asistente especializado en responder sobre procedimientos y políticas internas de una empresa.
Responde ÚNICAMENTE con base en el siguiente contexto.
Si el contexto no contiene la información necesaria, responde "No tengo información suficiente en los documentos".

Pregunta: {question}

Contexto:
{context}
"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.predict(prompt)
    return response.strip()


# ===== UI =====
vs = load_vectorstore()

if vs is None:
    st.stop()

# (Opcional) mostrar cuántos documentos hay en el docstore
try:
    n_docs = len(vs.docstore._dict)
    st.info(f"📚 Índice cargado. Documentos en el store: {n_docs}")
except Exception:
    pass

q = st.text_input("Escribe tu consulta:", placeholder="¿Qué es RAG?")
if st.button("Enviar") and q.strip():
    with st.spinner("Pensando..."):
        resp = answer(q, vs)
    if resp and resp.strip():
        st.markdown(f"**Respuesta:**\n\n{resp}")
