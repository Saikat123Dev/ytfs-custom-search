import os
import re
import uuid
import logging
import numpy as np
import pyarrow as pa
import json
import subprocess
import tempfile
import urllib.request
import urllib.error
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from supadata import Supadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
import voyageai
import lancedb

# Load environment variables
load_dotenv()

# Configure APIs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure AI services
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# Initialize Supadata
supadata = Supadata(api_key=os.getenv("SUPADATA_API_KEY"))

# LanceDB setup
embedding_dim = 768
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "lancedb_data")
db = lancedb.connect(db_path)

schema = pa.schema([
    ("id", pa.string()),
    ("video_id", pa.string()),
    ("text", pa.string()),
    ("start", pa.float32()),
    ("duration", pa.float32()),
    ("vector", pa.list_(pa.float32(), embedding_dim))
])

# Ensure clean state before creating table
try:
    db.drop_table("transcripts", ignore_missing=True)
    logger.info("Dropped old transcripts table.")
except Exception as e:
    logger.warning(f"Drop failed or not needed: {e}")

# Create new table
table = db.create_table("transcripts", schema=schema)
logger.info("Created new transcripts table.")


class TranscriptSegment:
    def __init__(self, text: str, start: float, duration: float):
        self.text = text
        self.start = start
        self.duration = duration


class YouTubeWorkflowService:
    def __init__(self):
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if self.youtube_api_key:
            self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
        else:
            self.youtube = None
            logger.warning("YouTube API key not found. Suggestions feature will be limited.")
        
        # Initialize Supadata client
        supadata_api_key = os.getenv("SUPADATA_API_KEY")
        if not supadata_api_key:
            raise ValueError("SUPADATA_API_KEY environment variable is required")
        
        self.supadata = Supadata(api_key=supadata_api_key)
        logger.info("Supadata client initialized successfully")

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11})",
            r"youtu\.be\/([0-9A-Za-z_-]{11})",
            r"embed\/([0-9A-Za-z_-]{11})",
            r"watch\?v=([0-9A-Za-z_-]{11})",
            r"shorts\/([0-9A-Za-z_-]{11})"
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        raise ValueError("Invalid YouTube URL. Please provide a valid YouTube video URL.")

    def _parse_supadata_transcript(self, content: List[Any]) -> List[TranscriptSegment]:
     segments = []

     for item in content:
        try:
            # Accessing attributes via dot notation because it's a TranscriptChunk object
            text = item.text
            start = float(item.offset) / 1000
            duration = float(item.duration) / 1000

            segments.append(TranscriptSegment(
                text=text,
                start=start,
                duration=duration,
            ))

        except Exception as e:
            logger.warning(f"Skipping invalid segment: {e}")

     if not segments:
        raise ValueError("No valid transcript segments found after parsing")

     return segments


    
    def _create_segments_from_text(self, text: str) -> List[TranscriptSegment]:
        """Create segments from plain text by splitting into sentences"""
        segments = []
        
        # Clean up the text
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        if not text:
            return segments
        
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        
        current_time = 0.0
        segment_duration = 3.0  # Default 3 seconds per segment
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Only include substantial sentences
                segments.append(TranscriptSegment(
                    text=sentence,
                    start=current_time,
                    duration=segment_duration
                ))
                current_time += segment_duration
        
        # If no sentences found, create one segment with the full text
        if not segments and text:
            segments.append(TranscriptSegment(
                text=text,
                start=0.0,
                duration=60.0  # Default 1 minute
            ))
        
        return segments

    def get_transcript(self, youtube_url: str) -> List[TranscriptSegment]:
        video_id = self.extract_video_id(youtube_url)
        logger.info(f"Extracting transcript for video {video_id} using Supadata AI...")

        try:
            # <-- use video_id, not url
            transcript = self.supadata.youtube.transcript(
                video_id=video_id,
                text=False
            )
            print('transcript',type(transcript))

            if not transcript or not hasattr(transcript, 'content'):
                raise ValueError("No transcript content returned from Supadata")

            transcript_content = transcript.content
            if not transcript_content:
                raise ValueError("Empty transcript content returned from Supadata")

            logger.info("Successfully retrieved transcript content from Supadata")
            logger.debug(f"Transcript content preview: {transcript_content[:200]}...")

            segments = self._parse_supadata_transcript(transcript_content)
            if not segments:
                raise ValueError("No transcript segments found after parsing")

            logger.info(f"Successfully extracted {len(segments)} transcript segments")
            return segments

        except Exception as e:
            error_msg = f"Failed to get transcript for video {video_id} using Supadata: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)



    def get_video_info(self, youtube_url: str) -> Dict[str, Any]:
        """Get video metadata using YouTube API (fallback to basic info if not available)"""
        video_id = self.extract_video_id(youtube_url)
        
        # Try YouTube API first
        if self.youtube:
            try:
                response = self.youtube.videos().list(
                    part='snippet,statistics,contentDetails',
                    id=video_id
                ).execute()
                
                if response['items']:
                    item = response['items'][0]
                    snippet = item['snippet']
                    statistics = item.get('statistics', {})
                    content_details = item.get('contentDetails', {})
                    
                    # Parse duration from ISO 8601 format
                    duration_str = content_details.get('duration', 'PT0S')
                    duration_seconds = self._parse_iso_duration(duration_str)
                    
                    return {
                        'video_id': video_id,
                        'title': snippet.get('title'),
                        'description': snippet.get('description'),
                        'uploader': snippet.get('channelTitle'),
                        'upload_date': snippet.get('publishedAt'),
                        'duration': duration_seconds,
                        'view_count': int(statistics.get('viewCount', 0)) if statistics.get('viewCount') else 0,
                        'like_count': int(statistics.get('likeCount', 0)) if statistics.get('likeCount') else 0,
                        'webpage_url': f"https://www.youtube.com/watch?v={video_id}"
                    }
            except Exception as e:
                logger.warning(f"YouTube API failed: {e}")
        
        # Fallback to basic info
        logger.info("Using fallback video info")
        return {
            'video_id': video_id,
            'title': f"Video {video_id}",
            'description': '',
            'uploader': '',
            'upload_date': '',
            'duration': 0,
            'view_count': 0,
            'like_count': 0,
            'webpage_url': f"https://www.youtube.com/watch?v={video_id}"
        }
    
    def _parse_iso_duration(self, duration_str: str) -> int:
        """Parse ISO 8601 duration to seconds"""
        try:
            # Remove PT prefix
            duration_str = duration_str.replace('PT', '')
            
            total_seconds = 0
            
            # Parse hours
            if 'H' in duration_str:
                hours = int(duration_str.split('H')[0])
                total_seconds += hours * 3600
                duration_str = duration_str.split('H')[1]
            
            # Parse minutes
            if 'M' in duration_str:
                minutes = int(duration_str.split('M')[0])
                total_seconds += minutes * 60
                duration_str = duration_str.split('M')[1]
            
            # Parse seconds
            if 'S' in duration_str:
                seconds = int(duration_str.split('S')[0])
                total_seconds += seconds
            
            return total_seconds
        except Exception:
            return 0

    def chunk_transcript(self, segments: List[TranscriptSegment]) -> List[Document]:
        """Chunk transcript into manageable pieces for embedding"""
        if not segments:
            return []

        # Create indexed text for chunking
        full_text = "\n".join(f"{i}|{seg.start}|{seg.duration}|{seg.text}" 
                             for i, seg in enumerate(segments))

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        raw_chunks = splitter.split_text(full_text)

        documents = []
        for chunk in raw_chunks:
            parts = chunk.split("\n")
            indices = []

            for part in parts:
                if "|" in part:
                    try:
                        idx = int(part.split("|")[0])
                        indices.append(idx)
                    except (ValueError, IndexError):
                        continue

            if not indices:
                continue

            # Get the first segment for metadata
            first_segment = segments[min(indices)]

            # Extract clean text content
            clean_text = "\n".join(part.split("|", 3)[-1] for part in parts if "|" in part and len(part.split("|")) >= 4)

            if not clean_text.strip():
                continue

            # Calculate total duration for this chunk
            total_duration = sum(segments[i].duration for i in indices if i < len(segments))

            documents.append(Document(
                page_content=clean_text,
                metadata={
                    "start": first_segment.start,
                    "duration": total_duration
                }
            ))

        return documents

    def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embeddings for text using Google's embedding model"""
        try:
            if not text.strip():
                logger.warning("Empty text provided for embedding")
                return [0.0] * embedding_dim

            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type=task_type
            )
            embedding = result["embedding"]

            # Ensure correct dimension
            if len(embedding) != embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {embedding_dim}, got {len(embedding)}")

            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * embedding_dim

    def embed_texts_batch(self, texts: List[str], max_workers: int = 5) -> List[List[float]]:
        """Generate embeddings for multiple texts using threading"""
        if not texts:
            return []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.embed_text, text): i for i, text in enumerate(texts)}
            embeddings = [None] * len(texts)

            for future in as_completed(futures):
                index = futures[future]
                try:
                    embeddings[index] = future.result()
                except Exception as e:
                    logger.error(f"Failed to embed text at index {index}: {e}")
                    # Use zero vector as fallback
                    embeddings[index] = [0.0] * embedding_dim

            return embeddings

    def is_video_indexed(self, video_id: str) -> bool:
        """Check if video is already indexed in the database"""
        try:
            count = table.count_rows(filter=f"video_id = '{video_id}'")
            return count > 0
        except Exception as e:
            logger.error(f"Error checking if video is indexed: {e}")
            return False

    def index_video(self, youtube_url: str, video_id: str) -> bool:
        """Index a video's transcript in the database"""
        if self.is_video_indexed(video_id):
            logger.info(f"Video {video_id} is already indexed.")
            return True

        try:
            logger.info(f"Getting transcript for video {video_id} using Supadata AI...")
            
            segments = self.get_transcript(youtube_url)

            if not segments:
                logger.warning(f"No transcript segments found for video {video_id}")
                return False

            logger.info(f"Chunking transcript into documents...")
            chunks = self.chunk_transcript(segments)

            if not chunks:
                logger.warning(f"No chunks created for video {video_id}")
                return False

            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            texts = [doc.page_content for doc in chunks]
            embeddings = self.embed_texts_batch(texts)

            # Prepare data for insertion
            data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if embedding and any(x != 0.0 for x in embedding):  # Skip zero embeddings
                    data.append({
                        "id": str(uuid.uuid4()),
                        "video_id": video_id,
                        "text": chunk.page_content,
                        "start": chunk.metadata["start"],
                        "duration": chunk.metadata["duration"],
                        "vector": embedding
                    })

            if not data:
                logger.warning(f"No valid embeddings generated for video {video_id}")
                return False

            logger.info(f"Inserting {len(data)} records into database...")
            table.add(data)
            logger.info(f"Successfully indexed video {video_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to index video {video_id}: {e}")
            return False

    def search_video_content(self, query: str, video_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search within a video's indexed content"""
        try:
            # Generate query embedding
            query_embedding = self.embed_text(query, task_type="retrieval_query")

            if not query_embedding or all(x == 0.0 for x in query_embedding):
                logger.error("Failed to generate valid query embedding")
                return []

            # Search in vector database
            results = (
                table.search(query_embedding, vector_column_name="vector")
                .where(f"video_id = '{video_id}'")
                .limit(top_k * 2)  # Get more results for reranking
                .to_list()
            )

            if not results:
                return []

            # Rerank results if Voyage API is available
            try:
                texts = [r["text"] for r in results]
                reranked = voyage.rerank(
                    query=query, 
                    documents=texts, 
                    model="rerank-2", 
                    top_k=top_k
                )

                final_results = []
                for res in reranked.results:
                    doc = results[res.index]
                    final_results.append({
                        "text": doc["text"],
                        "start": doc["start"],
                        "duration": doc["duration"],
                        "score": res.relevance_score,
                        "url": f"https://www.youtube.com/watch?v={video_id}&t={int(doc['start'])}s"
                    })

                return final_results

            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")
                # Fallback to original results without reranking
                return [{
                    "text": doc["text"],
                    "start": doc["start"],
                    "duration": doc["duration"],
                    "score": 1.0 - (i * 0.1),  # Simple fallback scoring
                    "url": f"https://www.youtube.com/watch?v={video_id}&t={int(doc['start'])}s"
                } for i, doc in enumerate(results[:top_k])]

        except Exception as e:
            logger.error(f"Error searching video content: {e}")
            return []

    def search_youtube_videos(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for videos on YouTube using the API"""
        if not self.youtube:
            logger.error("YouTube API not available")
            return []

        try:
            search_response = self.youtube.search().list(
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

        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching YouTube: {e}")
            return []

    def get_video_title(self, video_id: str) -> str:
        """Get video title from YouTube API"""
        if self.youtube:
            try:
                response = self.youtube.videos().list(
                    part='snippet',
                    id=video_id
                ).execute()

                if response['items']:
                    return response['items'][0]['snippet']['title']
            except Exception as e:
                logger.warning(f"YouTube API failed: {e}")
        
        # Fallback
        return f"Video {video_id}"


def print_video_segments(segments: List[Dict[str, Any]], video_title: str = ""):
    """Print formatted video segments"""
    if not segments:
        print("❌ No relevant segments found.")
        return

    if video_title:
        print(f"\n🎥 Video: {video_title}")

    print(f"🎯 Found {len(segments)} relevant segments:")
    print("-" * 60)

    for i, segment in enumerate(segments, 1):
        # Truncate text for display
        text_preview = " ".join(segment['text'].split()[:30])
        if len(segment['text'].split()) > 30:
            text_preview += "..."

        print(f"\n{i}. ⏱️  Timestamp: {int(segment['start'])}s")
        print(f"   📊 Score: {segment['score']:.4f}")
        print(f"   📝 Content: {text_preview}")
        print(f"   🔗 Direct link: {segment['url']}")


def print_youtube_search_results(videos: List[Dict[str, Any]]):
    """Print formatted YouTube search results"""
    if not videos:
        print("❌ No videos found.")
        return

    print(f"\n🔍 Found {len(videos)} videos:")
    print("-" * 60)

    for i, video in enumerate(videos, 1):
        # Truncate description
        desc_preview = video['description'][:100]
        if len(video['description']) > 100:
            desc_preview += "..."

        print(f"\n{i}. 📺 {video['title']}")
        print(f"   👤 Channel: {video['channel_title']}")
        print(f"   📅 Published: {video['published_at'][:10]}")
        print(f"   📝 {desc_preview}")
        print(f"   🔗 {video['url']}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize the service
    try:
        service = YouTubeWorkflowService()
        
        # Test video URL
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with actual URL
        video_id = service.extract_video_id(url)
        
        # Get video info
        info = service.get_video_info(url)
        print(f"Video Info: {info}")
        
        # Index the video
        success = service.index_video(url, video_id)
        if success:
            print(f"Successfully indexed video {video_id}")
            
            # Search within the video
            results = service.search_video_content("your search query", video_id, top_k=3)
            print_video_segments(results, info.get('title', ''))
        else:
            print(f"Failed to index video {video_id}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set the SUPADATA_API_KEY environment variable")