#!/usr/bin/env python
"""
Enhanced Automated Academic Paper Scraper (Free Version)

Features:
- Asynchronously fetches research paper data from arXiv (extendable to additional sources)
- Allows configuration of number of documents to scrape (default: 50, configurable up to 500)
- Truncates documents to a maximum size (default: 10,000 characters)
- Removes duplicate documents based on title similarity
- Ranks documents by semantic similarity to the query using SentenceTransformer embeddings
- Summarizes top documents using the free summarization model facebook/bart-large-cnn
- Outputs a structured JSON with title, source, URL, and summary (with actionable insights)
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import logging
import hashlib
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MAX_DOCS = 50
DEFAULT_SIZE_LIMIT = 10000

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_arxiv_url(query: str, start: int, max_results: int) -> str:
    base_url = "http://export.arxiv.org/api/query"
    return f"{base_url}?search_query=all:{query}&start={start}&max_results={max_results}"

def parse_arxiv_response(xml_text: str) -> list:
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    docs = []
    for entry in root.findall("atom:entry", ns):
        title_elem = entry.find("atom:title", ns)
        summary_elem = entry.find("atom:summary", ns)
        title = title_elem.text.strip() if title_elem is not None else "No Title"
        summary = summary_elem.text.strip() if summary_elem is not None else ""
        url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("rel") == "alternate":
                url = link.attrib.get("href")
                break
        docs.append({
            "title": title,
            "abstract": summary,
            "url": url,
            "source": "arXiv"
        })
    return docs

async def fetch_arxiv_documents(query: str, total_docs: int, results_per_page: int = 25) -> list:
    docs = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for start in range(0, total_docs, results_per_page):
            url = generate_arxiv_url(query, start, results_per_page)
            logger.info(f"Fetching URL: {url}")
            tasks.append(fetch_text(session, url))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        for response in responses:
            if isinstance(response, Exception):
                logger.error(f"Error fetching a page: {response}")
                continue
            try:
                page_docs = parse_arxiv_response(response)
                docs.extend(page_docs)
            except Exception as e:
                logger.error(f"Error parsing XML: {e}")
    return docs

async def fetch_text(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as resp:
        return await resp.text()

def truncate_document(text: str, size_limit: int = DEFAULT_SIZE_LIMIT) -> str:
    if len(text) > size_limit:
        return text[:size_limit] + "\n\n[TRUNCATED]"
    return text

def remove_duplicates(documents: list) -> list:
    seen = set()
    unique_docs = []
    for doc in documents:
        norm_title = doc["title"].lower().strip()
        if norm_title in seen:
            continue
        seen.add(norm_title)
        unique_docs.append(doc)
    return unique_docs

def rank_documents(documents: list, query: str, model: SentenceTransformer) -> list:
    texts = [doc["title"] + " " + doc.get("abstract", "") for doc in documents]
    query_embedding = model.encode(query, convert_to_tensor=True)
    doc_embeddings = model.encode(texts, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
    scored_docs = list(zip(documents, cosine_scores.tolist()))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    ranked_docs = [doc for doc, score in scored_docs]
    return ranked_docs

def summarize_document(text: str, query: str, max_length: int = 150, min_length: int = 40) -> str:
    prompt = f"Summarize the following research paper with actionable insights for the query '{query}':\n\n{text}"
    try:
        summary_list = summarizer(prompt, max_length=max_length, min_length=min_length, do_sample=False)
        summary = summary_list[0]['summary_text'].strip()
        return summary
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return "Summarization failed."

async def enhanced_scraper(query: str, total_docs: int = DEFAULT_MAX_DOCS, size_limit: int = DEFAULT_SIZE_LIMIT) -> dict:
    logger.info(f"Starting search for query: '{query}' with total_docs={total_docs}")
    raw_docs = await fetch_arxiv_documents(query, total_docs)
    for doc in raw_docs:
        doc_text = doc.get("abstract", "")
        doc["content"] = truncate_document(doc_text, size_limit)
    unique_docs = remove_duplicates(raw_docs)
    logger.info(f"Retrieved {len(raw_docs)} documents; {len(unique_docs)} unique documents remain after duplicate removal.")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ranked_docs = rank_documents(unique_docs, query, model)
    logger.info("Documents ranked by relevance.")
    top_n = min(5, len(ranked_docs))
    summarized_results = []
    for doc in ranked_docs[:top_n]:
        text_to_summarize = f"Title: {doc['title']}\n\nAbstract: {doc.get('content', '')}"
        summary = summarize_document(text_to_summarize, query)
        summarized_results.append({
            "title": doc["title"],
            "source": doc["source"],
            "url": doc.get("url"),
            "summary": summary
        })
    output = {
        "query": query,
        "total_documents_retrieved": len(ranked_docs),
        "top_results": summarized_results
    }
    return output

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Automated Academic Paper Scraper (Free Version)")
    parser.add_argument("--query", type=str, required=True, help="Search query (e.g., 'model overfitting recommendations')")
    parser.add_argument("--total_docs", type=int, default=DEFAULT_MAX_DOCS,
                        help=f"Number of documents to retrieve (default {DEFAULT_MAX_DOCS}, max 500)")
    parser.add_argument("--size_limit", type=int, default=DEFAULT_SIZE_LIMIT,
                        help=f"Max characters per document (default {DEFAULT_SIZE_LIMIT})")
    args = parser.parse_args()
    if args.total_docs > 500:
        logger.warning("Limiting total_docs to 500.")
        args.total_docs = 500
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(enhanced_scraper(args.query, args.total_docs, args.size_limit))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
