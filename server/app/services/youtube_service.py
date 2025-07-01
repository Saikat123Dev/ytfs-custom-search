import os
import re
import uuid
import numpy as np
import pyarrow as pa
from dotenv import load_dotenv
from typing import List, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
import voyageai
import lancedb

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

embedding_dim = 768
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db = lancedb.connect(os.path.join(BASE_DIR, "lancedb_data"))
print('db',db)

schema = pa.schema([
    ("id", pa.string()),
    ("video_id", pa.string()),
    ("text", pa.string()),
    ("start", pa.float32()),
    ("duration", pa.float32()),
    ("vector", pa.list_(pa.float32(), embedding_dim))
])

if "transcripts" in db.table_names():
    table = db.open_table("transcripts")
else:
    table = db.create_table("transcripts", schema=schema)


class TranscriptSegment:
    def __init__(self, text: str, start: float, duration: float):
        self.text = text
        self.start = start
        self.duration = duration

class YouTubeTranscriptService:
    def extract_video_id(self, url: str) -> str:
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
        if not match:
            raise ValueError("Invalid YouTube URL")
        return match.group(1)

    def get_transcript(self, youtube_url: str) -> List[TranscriptSegment]:
        vid = self.extract_video_id(youtube_url)
        try:
            raw = YouTubeTranscriptApi.get_transcript(vid)
        except (VideoUnavailable, TranscriptsDisabled, NoTranscriptFound) as e:
            raise ValueError(f"Transcript error: {e}")
        return [TranscriptSegment(**seg) for seg in raw]

    def chunk_transcript(self, segments: List[TranscriptSegment]) -> List[Document]:
        full_text = "\n".join(f"{i}|{seg.start}|{seg.duration}|{seg.text}" for i, seg in enumerate(segments))
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        raw_chunks = splitter.split_text(full_text)

        documents = []
        for chunk in raw_chunks:
            parts = chunk.split("\n")
            idxs = [int(p.split("|")[0]) for p in parts if "|" in p]
            if not idxs:
                continue
            first = segments[min(idxs)]
            doc_text = "\n".join(p.split("|", 3)[-1] for p in parts)
            documents.append(Document(page_content=doc_text, metadata={
                "start": first.start,
                "duration": sum(segments[i].duration for i in idxs)
            }))
        return documents


def embed_text(text: str, task_type: str = "retrieval_document") -> List[float]:
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type=task_type
    )
    embedding = result["embedding"]
    assert len(embedding) == embedding_dim, "Embedding dimension mismatch"
    return embedding

def embed_texts_multithreaded(texts: List[str], max_workers: int = 6) -> List[List[float]]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed_text, text) for text in texts]
        return [f.result() for f in as_completed(futures)]


def is_video_already_indexed(video_id: str) -> bool:
    return table.count_rows(filter=f"video_id = '{video_id}'") > 0

def index_documents(chunks: List[Document], video_id: str):
    if is_video_already_indexed(video_id):
        print("✅ Video already indexed in LanceDB.")
        return

    print("⚡ Generating Gemini embeddings...")
    texts = [doc.page_content for doc in chunks]
    vectors = embed_texts_multithreaded(texts)

    data = []
    for i, vec in enumerate(vectors):
        data.append({
            "id": str(uuid.uuid4()),
            "video_id": video_id,
            "text": chunks[i].page_content,
            "start": chunks[i].metadata["start"],
            "duration": chunks[i].metadata["duration"],
            "vector": vec
        })

    table.add(data)
    print("✅ Indexed and stored in LanceDB.")

def search_with_rerank_multiquery(query: str, video_id: str, top_k: int = 2) -> List[Dict[str, Any]]:
    queries = [q.strip() for q in re.split(r"[;,]", query) if q.strip()]
    final_results = []

    def is_far_enough(ts, used_timestamps):
        return all(abs(ts - t) > 300 for t in used_timestamps)

    for q in queries:
        query_vec = embed_text(q, task_type="retrieval_query")
        results = (
            table.search(query_vec, vector_column_name="vector")
            .where(f"video_id = '{video_id}'")
            .limit(20)
            .to_list()
        )
        if not results:
            continue

        texts = [r["text"] for r in results]
        reranked = voyage.rerank(query=q, documents=texts, model="rerank-2", top_k=top_k * 3)

        used_timestamps = set()
        selected = []

        for res in reranked.results:
            doc = results[res.index]
            ts = int(doc["start"])
            if is_far_enough(ts, used_timestamps):
                selected.append({
                    "query": q,
                    "text": doc["text"],
                    "start": doc["start"],
                    "score": res.relevance_score
                })
                used_timestamps.add(ts)
            if len(selected) >= top_k:
                break

        final_results.extend(selected)

    return final_results


def make_url(video_id: str, start: float) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={int(start)}s"

def print_results(results: List[Dict[str, Any]], video_id: str):
    print(f"\n🔎 Top {len(results)} results:\n")
    for r in results:
        snippet = " ".join(r['text'].split()[:25]) + "..."
        print(f"- 🔑 Query: \"{r.get('query', 'N/A')}\"")
        print(f"  ⏱️ Timestamp: [{int(r['start'])}s]")
        print(f"  📄 Snippet: \"{snippet}\"")
        print(f"  👉 {make_url(video_id, r['start'])} (score: {r['score']:.4f})\n")


def main():
    try:
        yt_url = input("🎥 YouTube URL: ").strip()
        query = input("🧠 Search query (comma or semicolon separated): ").strip()

        svc = YouTubeTranscriptService()
        video_id = svc.extract_video_id(yt_url)

        segments = svc.get_transcript(yt_url)
        chunks = svc.chunk_transcript(segments)

        index_documents(chunks, video_id)

        print("🔍 Performing semantic search and reranking...")
        results = search_with_rerank_multiquery(query, video_id)

        if results:
            print_results(results, video_id)
        else:
            print("⚠️ No relevant matches found.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
