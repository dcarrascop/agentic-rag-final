# build_index.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# === CONFIG ===
DATA_DIR = "data"
INDEX_DIR = "data/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# === CARGA DE DOCUMENTOS ===
pdf_files = [
    "Financiamiento_Capacitacion.pdf",
    "Seguros_Complementarios.pdf"
]

docs = []
for pdf in pdf_files:
    path = os.path.join(DATA_DIR, pdf)
    print(f"📄 Cargando {pdf}...")
    loader = PyPDFLoader(path)
    docs.extend(loader.load())

# === DIVISIÓN EN CHUNKS ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)
print(f"✅ Total de fragmentos: {len(splits)}")

# === EMBEDDINGS Y VECTORIZACIÓN ===
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = FAISS.from_documents(splits, embeddings)

# === GUARDADO ===
os.makedirs(INDEX_DIR, exist_ok=True)
db.save_local(INDEX_DIR)
print(f"💾 Índice guardado en {INDEX_DIR}")