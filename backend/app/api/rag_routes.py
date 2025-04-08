from fastapi import APIRouter, HTTPException, Form
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os
from typing import Optional
from backend.app.agents.ai_insights_agent import generate_ai_insights

router = APIRouter()

INDEX_FILE = "data_1/faiss_index.index"
METADATA_FILE = "data_1/metadata.json"
DOCS_FOLDER = "data_1/documents"

if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
    raise FileNotFoundError("FAISS index or metadata file not found. Please run build_index.py first.")

index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

CHUNK_SIZE = 500
OVERLAP = 100

def get_chunk_text(full_text: str, chunk_idx: int, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> str:
    start = max(0, chunk_idx * chunk_size - chunk_idx * overlap)
    end = start + chunk_size
    return full_text[start:end]

@router.post("/query-summarize")
async def query_summarize(
    query: str = Form(...), 
    top_k: int = Form(3),
    enable_cot: bool = Form(False)
):
    try:
        query_embedding = model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        distances, indices = index.search(query_embedding, top_k)
        retrieved_snippets = []
        combined_text = ""
        for idx in indices[0]:
            if idx < len(metadata):
                meta = metadata[idx]
                filename = meta["filename"]
                c_idx = meta["chunk_idx"]
                file_path = os.path.join(DOCS_FOLDER, filename)
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        doc_text = f.read()
                    snippet = get_chunk_text(doc_text, c_idx)
                    snippet_info = {"filename": filename, "chunk_idx": c_idx, "text": snippet}
                    retrieved_snippets.append(snippet_info)
                    combined_text += f"Snippet from {filename} (chunk {c_idx}):\n{snippet}\n\n"
                else:
                    snippet_info = {"filename": filename, "chunk_idx": c_idx, "text": f"[File not found: {filename}]"}
                    retrieved_snippets.append(snippet_info)
                    combined_text += f"[File not found: {filename}]\n\n"
        final_answer = generate_ai_insights(
            eda_summary=combined_text,
            model_summary=f"User query: {query}",
            model_choice="mistral",
            chunk_threshold=2000,
            force_regenerate=True,
            enable_cot=enable_cot
        )
        return {
            "query": query,
            "top_k": top_k,
            "retrieved_snippets": retrieved_snippets,
            "answer": final_answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
