import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

# Import the correct class name from your service
from services.youtube_service import YouTubeTranscriptService  # Updated import to relative

app = FastAPI()

origins = [
    "http://localhost:3000",  # Example: Your frontend development server
    "http://localhost:5173",  # Example: Your deployed frontend
    # Add other allowed origins as needed
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,  # Allow cookies and authorization headers
    allow_methods=["*"],     # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],     # Allow all headers
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchRequest(BaseModel):
    youtube_url: str
    query: str
    suggestions: bool = False
    top_k: int = 5
    max_related_videos: int = 5

# Initialize YouTube service with correct class name
youtube_service = YouTubeTranscriptService()

def process_related_video_segments(video_info: dict, query: str, top_k: int = 1):
    """Process segments for a related video - returns exactly one best match"""
    try:
        video_id = video_info['video_id']
        video_url = video_info['url']
        
        # Index if not already indexed
        if not youtube_service.is_video_indexed(video_id):
            youtube_service.index_video_enhanced(video_url, video_id)  # Updated method name
        
        # Search segments - we only want the top 1 result
        results = youtube_service.search_video_content_enhanced(query, video_id, top_k=1)  # Updated method name
        
        # Return only the best match if found
        if results:
            r = results[0]  # Take only the first (best) result
            snippet = " ".join(r['text'].split()[:25]) + "..."
            return {
                "timestamp": int(r["start"]),
                "snippet": snippet,
                "url": r["url"],
                "score": round(r.get("relevance_score", r.get("score", 0)), 4)  # Updated score field
            }
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error processing related video {video_id}: {e}")
        return None

@app.post("/search")
async def search_transcript(data: SearchRequest):
    try:
        # Extract video ID from URL
        video_id = youtube_service.extract_video_id(data.youtube_url)
        
        # Get input video metadata (updated method name)
        metadata = youtube_service.get_video_metadata(video_id)
        input_video_title = metadata.get('title', f'Video {video_id}')
        
        # Index video if not already indexed
        if not youtube_service.is_video_indexed(video_id):
            print("⚙️ Indexing new video transcript before search...")
            success = youtube_service.index_video_enhanced(data.youtube_url, video_id)  # Updated method name
            if not success:
                raise HTTPException(status_code=500, detail="Failed to index video transcript")
            print("✅ Indexing done. Proceeding to search.")
        
        # Search input video after ensuring indexing is complete
        results = youtube_service.search_video_content_enhanced(data.query, video_id, top_k=data.top_k)  # Updated method name
        
        # Format input video results
        input_video_segments = []
        for r in results:
            snippet = " ".join(r['text'].split()[:25]) + "..."
            input_video_segments.append({
                "timestamp": int(r["start"]),
                "snippet": snippet,
                "url": r["url"],
                "score": round(r.get("relevance_score", r.get("score", 0)), 4)  # Updated score field
            })
        
        # Prepare response
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
        
        # If suggestions enabled, search for related videos
        if data.suggestions:
            print("🌟 Suggestions enabled - searching for related videos...")
            
            # Use the YouTube service method for searching videos (updated method name)
            related_videos = youtube_service.search_youtube_videos_enhanced(data.query, data.max_related_videos)
            
            related_videos_with_segments = []
            for video in related_videos:
                # Process segments for each related video - get only the best match
                best_segment = process_related_video_segments(video, data.query, top_k=1)
                
                # Only include videos where we found at least one relevant segment
                if best_segment:
                    related_videos_with_segments.append({
                        "video_id": video['video_id'],
                        "title": video['title'],
                        "channel_title": video['channel_title'],
                        "description": video['description'][:200] + "..." if len(video['description']) > 200 else video['description'],
                        "published_at": video['published_at'],
                        "url": video['url'],
                        "best_segment": best_segment,  # Changed from "segments" to "best_segment"
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
    # Add service health check
    try:
        health_status = youtube_service.get_health_status()
        return {
            "status": "healthy",
            "service_health": health_status
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Additional endpoint to get service statistics
@app.get("/stats")
async def get_statistics():
    try:
        stats = youtube_service.get_indexing_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoint to manually reset circuit breaker
@app.post("/admin/reset-circuit-breaker")
async def reset_circuit_breaker():
    try:
        youtube_service.reset_circuit_breaker()
        return {"status": "success", "message": "Circuit breaker reset"}
    except Exception as e:
        logger.error(f"Reset circuit breaker error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoint to cleanup failed videos
@app.post("/admin/cleanup")
async def cleanup_failed_videos():
    try:
        youtube_service.cleanup_failed_videos()
        return {"status": "success", "message": "Failed videos cleaned up"}
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get configuration from environment variables with defaults
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