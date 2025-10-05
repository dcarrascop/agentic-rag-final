# app.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

# === CONFIGURACIÓN ===
DATA_DIR = "data"
INDEX_DIR = "data/faiss_index"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

st.set_page_config(page_title="Agentic RAG con MultiQuery", page_icon="🧠")
st.title("🧠 Agentic RAG — Políticas y Procedimientos")
st.caption("Consulta los documentos internos de tu empresa usando recuperación semántica (RAG).")

# --- Construcción del índice ---
def build_index():
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        st.error("No se encontraron archivos PDF en la carpeta 'data/'.")
        return None

    docs = []
    for pdf in pdf_files:
        st.write(f"📄 Cargando {pdf}...")
        loader = PyPDFLoader(os.path.join(DATA_DIR, pdf))
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=350)
    splits = splitter.split_documents(docs)
    st.write(f"✅ Total de fragmentos: {len(splits)}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = FAISS.from_documents(splits, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    db.save_local(INDEX_DIR)
    st.success("💾 Índice FAISS reconstruido correctamente.")
    return db


# --- Carga del índice ---
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(INDEX_DIR):
        st.warning("No se encontró un índice FAISS. Presiona el botón para generarlo.")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return vs
    except Exception as e:
        st.error(f"❌ No se pudo cargar el índice: {e}")
        return None


# --- Botones de control ---
st.sidebar.header("⚙️ Configuración")
if st.sidebar.button("🔁 Reconstruir índice"):
    with st.spinner("Reconstruyendo índice..."):
        vs = build_index()
else:
    vs = load_vectorstore()

if st.sidebar.button("🗑️ Limpiar caché y recargar índice"):
    st.cache_resource.clear()
    st.success("🧹 Caché limpiada. Vuelve a preguntar para usar el índice actualizado.")

if vs is None:
    st.stop()

try:
    n_docs = len(vs.docstore._dict)
    st.info(f"📚 Índice cargado con {n_docs} fragmentos de texto.")
except Exception:
    pass


# --- Configuración del retriever mejorado ---
MQ_PROMPT_ES = PromptTemplate.from_template(
    "Genera 4 reformulaciones breves y diferentes, en español, de la siguiente consulta del usuario.\n"
    "Usa sinónimos y expresiones cercanas, sin traducir al inglés.\n"
    "Solo devuelve una lista con una reformulación por línea.\n\n"
    "Consulta: {question}"
)

def get_retriever(vectorstore):
    base = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 40})
    llm_q = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return MultiQueryRetriever.from_llm(retriever=base, llm=llm_q, prompt=MQ_PROMPT_ES)


# --- Generación de respuesta ---
def answer(question: str, vectorstore: FAISS) -> str:
    retriever = get_retriever(vectorstore)
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return "No encuentro pasajes relevantes en los documentos cargados."

    context = "\n\n".join(d.page_content for d in docs)
    st.write("🔍 **Vista previa del contexto recuperado:**")
    st.text(context[:1000] + "..." if len(context) > 1000 else context)

    prompt = f"""
Eres un asistente que responde exclusivamente según las políticas y procedimientos de la empresa.
Responde solo con base en el siguiente contexto. Si no hay información suficiente, responde "No tengo información suficiente en los documentos".

Pregunta: {question}

Contexto:
{context}
"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm.predict(prompt).strip()


# --- Interfaz principal ---
q = st.text_input("✏️ Escribe tu consulta:", placeholder="Ejemplo: ¿Cuál es el procedimiento del seguro complementario?")
if st.button("Enviar") and q.strip():
    with st.spinner("Buscando información..."):
        resp = answer(q, vs)
    st.markdown(f"**Respuesta:**\n\n{resp}")