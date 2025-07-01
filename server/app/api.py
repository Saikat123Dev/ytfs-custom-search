from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.youtube_service import (
    YouTubeTranscriptService,
    index_documents,
    search_with_rerank_multiquery,
    make_url,
    is_video_already_indexed
)
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = [
        "http://localhost:3000",  # Example: Your frontend development server
        "http://localhost:5173", # Example: Your deployed frontend
        # Add other allowed origins as needed
    ]
app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,  # Allow cookies and authorization headers
        allow_methods=["*"],     # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
        allow_headers=["*"],     # Allow all headers
    )





class SearchRequest(BaseModel):
    youtube_url: str
    query: str


@app.post("/search")
async def search_transcript(data: SearchRequest):
    try:
        svc = YouTubeTranscriptService()
        video_id = svc.extract_video_id(data.youtube_url)

        # If not indexed, synchronously process transcript and index
        if not is_video_already_indexed(video_id):
            print("⚙️ Indexing new video transcript before search...")
            segments = svc.get_transcript(data.youtube_url)
            chunks = svc.chunk_transcript(segments)
            index_documents(chunks, video_id)
            print("✅ Indexing done. Proceeding to search.")

        # Search after ensuring indexing is complete
        results = search_with_rerank_multiquery(data.query, video_id)

        formatted = []
        for r in results:
            snippet = " ".join(r['text'].split()[:25]) + "..."
            formatted.append({
                "query": r.get("query", "N/A"),
                "timestamp": int(r["start"]),
                "snippet": snippet,
                "url": make_url(video_id, r["start"]),
                "score": round(r["score"], 4)
            })

        return {
            "video_id": video_id,
            "results": formatted
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
