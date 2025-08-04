import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn
from services.youtube_service import YouTubeWorkflowService

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{rest_of_path:path}")
async def preflight_handler(request: Request, rest_of_path: str = ""):
    return JSONResponse(content={"message": "Preflight OK"}, status_code=200)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request model
class SearchRequest(BaseModel):
    youtube_url: str
    query: str
    suggestions: bool = False
    top_k: int = 5
    max_related_videos: int = 5

# YouTube service init
youtube_service = YouTubeWorkflowService()

def get_youtube_service():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if api_key:
        return build('youtube', 'v3', developerKey=api_key)
    return None

def search_youtube_videos(query: str, max_results: int = 5):
    youtube = get_youtube_service()
    if not youtube:
        return []

    try:
        search_response = youtube.search().list(
            part='id,snippet',
            q=query,
            type='video',
            maxResults=max_results,
            order='relevance'
        ).execute()

        return [
            {
                'video_id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'channel_title': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            }
            for item in search_response['items']
        ]
    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return []

def process_related_video_segments(video_info: dict, query: str, top_k: int = 1):
    try:
        video_id = video_info['video_id']
        video_url = video_info['url']

        if not youtube_service.is_video_indexed(video_id):
            youtube_service.index_video(video_url, video_id)

        results = youtube_service.search_video_content(query, video_id, top_k=1)

        if results:
            r = results[0]
            snippet = " ".join(r['text'].split()[:25]) + "..."
            return {
                "timestamp": int(r["start"]),
                "snippet": snippet,
                "url": r["url"],
                "score": round(r["score"], 4)
            }
        return None

    except Exception as e:
        logger.error(f"Error processing related video {video_info.get('video_id')}: {e}")
        return None

@app.post("/search")
async def search_transcript(data: SearchRequest):
    try:
        video_id = youtube_service.extract_video_id(data.youtube_url)
        input_video_title = youtube_service.get_video_title(video_id)

        if not youtube_service.is_video_indexed(video_id):
            logger.info("⚙️ Indexing new video transcript before search...")
            if not youtube_service.index_video(data.youtube_url, video_id):
                raise HTTPException(status_code=500, detail="Failed to index video transcript")
            logger.info("✅ Indexing done. Proceeding to search.")

        results = youtube_service.search_video_content(data.query, video_id, top_k=data.top_k)

        input_video_segments = [
            {
                "timestamp": int(r["start"]),
                "snippet": r['text'],
                "url": r["url"],
                "score": round(r["score"], 4)
            }
            for r in results
        ]

        response = {
            "input_video": {
                "video_id": video_id,
                "title": input_video_title,
                "url": data.youtube_url,
                "segments": input_video_segments
            },
            "suggestions_enabled": data.suggestions,
            "total_input_segments": len(input_video_segments)
        }

        if data.suggestions:
            logger.info("🌟 Suggestions enabled - searching for related videos...")
            related_videos = youtube_service.search_youtube_videos(data.query, data.max_related_videos)

            related_videos_with_segments = []
            for video in related_videos:
                best_segment = process_related_video_segments(video, data.query, top_k=1)
                if best_segment:
                    related_videos_with_segments.append({
                        "video_id": video['video_id'],
                        "title": video['title'],
                        "channel_title": video['channel_title'],
                        "description": video['description'][:200] + "..." if len(video['description']) > 200 else video['description'],
                        "published_at": video['published_at'],
                        "url": video['url'],
                        "best_segment": best_segment,
                        "has_relevant_content": True
                    })

            response["related_videos"] = related_videos_with_segments
            response["total_related_videos"] = len(related_videos_with_segments)

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "YouTube Semantic Search API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", 1))

    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Reload: {reload}, Workers: {workers}")

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_config=None,
        access_log=False
    )
