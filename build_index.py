# build_index.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# === CONFIGURACIÓN ===
DATA_DIR = "data"
INDEX_DIR = "data/faiss_index"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# === CARGA DE DOCUMENTOS ===
pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
if not pdf_files:
    raise ValueError("❌ No se encontraron archivos PDF en la carpeta 'data/'.")

docs = []
for pdf in pdf_files:
    path = os.path.join(DATA_DIR, pdf)
    print(f"📄 Cargando {pdf}...")
    loader = PyPDFLoader(path)
    docs.extend(loader.load())

# === DIVISIÓN EN CHUNKS ===
splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=350)
splits = splitter.split_documents(docs)
print(f"✅ Total de fragmentos generados: {len(splits)}")

# === EMBEDDINGS Y VECTORIZACIÓN ===
print("🔍 Creando embeddings...")
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = FAISS.from_documents(splits, embeddings)

# === GUARDADO DEL ÍNDICE ===
os.makedirs(INDEX_DIR, exist_ok=True)
db.save_local(INDEX_DIR)
print(f"💾 Índice FAISS guardado en '{INDEX_DIR}'.")