import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
import uvicorn

# Import your YouTube service
from services.youtube_service import YouTubeWorkflowService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class SearchRequest(BaseModel):
    youtube_url: str
    query: str
    suggestions: bool = False
    top_k: int = 5
    max_related_videos: int = 5

# Initialize YouTube service
youtube_service = YouTubeWorkflowService()

def get_youtube_service():
    """Initialize YouTube API service"""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if api_key:
        return build('youtube', 'v3', developerKey=api_key)
    return None

def search_youtube_videos(query: str, max_results: int = 5):
    """Search for related videos on YouTube"""
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

        videos = []
        for item in search_response['items']:
            videos.append({
                'video_id': item['id']['videoId'],
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'channel_title': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            })
        return videos
    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return []

def process_related_video_segments(video_info: dict, query: str, top_k: int = 3):
    """Process segments for a related video"""
    try:
        video_id = video_info['video_id']
        video_url = video_info['url']
        
        # Index if not already indexed
        if not youtube_service.is_video_indexed(video_id):
            youtube_service.index_video(video_url, video_id)
        
        # Search segments
        results = youtube_service.search_video_content(query, video_id, top_k=top_k)
        
        formatted_segments = []
        for r in results:
            snippet = " ".join(r['text'].split()[:25]) + "..."
            formatted_segments.append({
                "timestamp": int(r["start"]),
                "snippet": snippet,
                "url": r["url"],
                "score": round(r["score"], 4)
            })
        
        return formatted_segments
    except Exception as e:
        logger.error(f"Error processing related video {video_id}: {e}")
        return []

@app.post("/search")
async def search_transcript(data: SearchRequest):
    try:
        # Extract video ID from URL
        video_id = youtube_service.extract_video_id(data.youtube_url)
        
        # Get input video title
        input_video_title = youtube_service.get_video_title(video_id)
        
        # Index video if not already indexed
        if not youtube_service.is_video_indexed(video_id):
            print("⚙️ Indexing new video transcript before search...")
            success = youtube_service.index_video(data.youtube_url, video_id)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to index video transcript")
            print("✅ Indexing done. Proceeding to search.")
        
        # Search input video after ensuring indexing is complete
        results = youtube_service.search_video_content(data.query, video_id, top_k=data.top_k)
        
        # Format input video results
        input_video_segments = []
        for r in results:
            snippet = " ".join(r['text'].split()[:25]) + "..."
            input_video_segments.append({
                "timestamp": int(r["start"]),
                "snippet": snippet,
                "url": r["url"],
                "score": round(r["score"], 4)
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
            
            # Use the YouTube service method for searching videos
            related_videos = youtube_service.search_youtube_videos(data.query, data.max_related_videos)
            
            related_videos_with_segments = []
            for video in related_videos:
                # Process segments for each related video
                segments = process_related_video_segments(video, data.query, top_k=3)
                
                related_videos_with_segments.append({
                    "video_id": video['video_id'],
                    "title": video['title'],
                    "channel_title": video['channel_title'],
                    "description": video['description'][:200] + "..." if len(video['description']) > 200 else video['description'],
                    "published_at": video['published_at'],
                    "url": video['url'],
                    "segments": segments,
                    "segments_found": len(segments)
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
    # Get configuration from environment variables with defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", 1))
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Reload: {reload}, Workers: {workers}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_config=None,
        access_log=False
    )