# app.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

# === CONFIGURACIÓN ===
DATA_DIR = "data"
INDEX_DIR = "data/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="Agentic RAG con recarga de índice", page_icon="🧠")
st.title("🧠 Agentic RAG — Documentación interna")
st.caption("Busca respuestas en los documentos de políticas y procedimientos de tu empresa.")

# --- Función para construir el índice ---
def build_index():
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        st.error("No se encontraron archivos PDF en la carpeta `data/`.")
        return None

    docs = []
    for pdf in pdf_files:
        st.write(f"📄 Cargando {pdf}...")
        loader = PyPDFLoader(os.path.join(DATA_DIR, pdf))
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    st.write(f"✅ Total de fragmentos: {len(splits)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(splits, embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    db.save_local(INDEX_DIR)
    st.success("💾 Índice FAISS reconstruido correctamente.")
    return db


# --- Carga del índice desde disco ---
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(INDEX_DIR):
        st.warning("No se encontró el índice FAISS. Presiona el botón para generarlo.")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception as e:
        st.error(f"No se pudo cargar el índice: {e}")
        return None


# --- Interfaz ---
st.sidebar.header("⚙️ Configuración")

if st.sidebar.button("🔁 Reconstruir índice"):
    with st.spinner("Reconstruyendo índice..."):
        vs = build_index()
else:
    vs = load_vectorstore()

if vs is None:
    st.stop()

try:
    n_docs = len(vs.docstore._dict)
    st.info(f"📚 Índice cargado correctamente. Fragmentos almacenados: {n_docs}")
except Exception:
    pass


# --- Función para responder preguntas ---
def answer(question: str, vectorstore: FAISS) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return "No encuentro pasajes relevantes en los documentos cargados."

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Eres un asistente que responde exclusivamente según las políticas y procedimientos de la empresa.
Responde solo con base en el siguiente contexto. Si no hay información suficiente, di "No tengo información suficiente en los documentos".

Pregunta: {question}

Contexto:
{context}
"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.predict(prompt)
    return response.strip()


# --- Entrada de usuario ---
q = st.text_input("✏️ Escribe tu consulta:", placeholder="Ejemplo: ¿Cómo funciona el seguro complementario?")
if st.button("Enviar") and q.strip():
    with st.spinner("Buscando información..."):
        resp = answer(q, vs)
    st.markdown(f"**Respuesta:**\n\n{resp}")