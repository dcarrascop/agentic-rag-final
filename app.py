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

st.set_page_config(page_title="Agentic RAG (FAISS pre-cargado)", page_icon="🧠")
st.title("🧠 Agentic RAG – Índice FAISS precalculado")
st.caption("Usa un índice FAISS ya generado para evitar recalcular embeddings.")

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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return "No encuentro pasajes relevantes en el índice cargado."

    context = "\n\n".join(d.page_content for d in docs[:10])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = f"""
Responde de forma clara y concisa la siguiente pregunta.

Pregunta: {question}

Contexto:
{context}

Si el contexto no es suficiente, responde con conocimientos generales.
"""
    return llm.predict(prompt).strip()

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