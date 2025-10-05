# build_index.py
import os, hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_PATH = "genai-principles.pdf"             # tu PDF fuente
PERSIST_DIR = "data/faiss_index"              # carpeta que subiremos al repo
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"No se encontró {PDF_PATH}")

    print("📄 Cargando PDF…")
    docs = PyPDFLoader(PDF_PATH).load()
    if not docs:
        raise ValueError("El PDF parece no tener texto (¿escaneado?).")

    print("🔪 Splitting…")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    print("🔢 Embeddings…")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("🧱 Construyendo FAISS…")
    vs = FAISS.from_documents(splits, embeddings)

    os.makedirs(PERSIST_DIR, exist_ok=True)
    vs.save_local(PERSIST_DIR)

    with open(os.path.join(PERSIST_DIR, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(file_hash(PDF_PATH))

    print(f"✅ Guardado en {PERSIST_DIR}. Sube esta carpeta a tu repo.")

if __name__ == "__main__":
    main()