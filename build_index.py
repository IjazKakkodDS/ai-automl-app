import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

DOCUMENTS_DIR = "data_1/documents"
INDEX_FILE = "data_1/faiss_index.index"
METADATA_FILE = "data_1/metadata.json"

model = SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def main():
    documents = []
    metadata_list = []
    for fname in os.listdir(DOCUMENTS_DIR):
        file_path = os.path.join(DOCUMENTS_DIR, fname)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    continue
                doc_chunks = chunk_text(content)
                for i, chunk in enumerate(doc_chunks):
                    documents.append(chunk)
                    metadata_list.append({"filename": fname, "chunk_idx": i})
    if not documents:
        raise ValueError(f"No documents found in {DOCUMENTS_DIR}.")
    print(f"Found {len(documents)} total text chunks. Computing embeddings...")
    embeddings = model.encode(documents).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)
    print("FAISS index built and saved.")
    print(f"Index file: {INDEX_FILE}")
    print(f"Metadata file: {METADATA_FILE}")

if __name__ == "__main__":
    main()
