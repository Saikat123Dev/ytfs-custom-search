import os
import re
import uuid
import logging
import asyncio
import aiohttp
import numpy as np
import pyarrow as pa
import json
import tempfile
from functools import lru_cache
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Union
from supadata import Supadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
import voyageai
import lancedb
import time
from dataclasses import dataclass
import hashlib
from threading import Lock
import pickle

# Load environment variables
load_dotenv()

# Configure logging for better performance monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure AI services
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# Connection pooling and caching
EMBEDDING_CACHE = {}
CACHE_LOCK = Lock()
MAX_CACHE_SIZE = 10000

# Optimized embedding dimension
embedding_dim = 768

# Database setup with optimizations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "lancedb_data")
db = lancedb.connect(db_path)

# Optimized schema with indexes
schema = pa.schema([
    ("id", pa.string()),
    ("video_id", pa.string()),
    ("text", pa.string()),
    ("start", pa.float32()),
    ("duration", pa.float32()),
    ("vector", pa.list_(pa.float32(), embedding_dim)),
    ("text_hash", pa.string()),  # For deduplication
    ("created_at", pa.timestamp('us'))  # For cleanup
])

@dataclass
class TranscriptSegment:
    text: str
    start: float
    duration: float
    
    def __post_init__(self):
        self.text = self.text.strip()

class OptimizedEmbeddingCache:
    """Thread-safe LRU cache for embeddings with persistence"""
    
    def __init__(self, max_size: int = 10000, cache_file: str = "embedding_cache.pkl"):
        self.max_size = max_size
        self.cache_file = cache_file
        self.cache = {}
        self.access_times = {}
        self.lock = Lock()
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache = data.get('cache', {})
                    logger.info(f"Loaded {len(self.cache)} embeddings from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump({'cache': self.cache}, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_hash(self, text: str) -> str:
        """Get hash for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        text_hash = self._get_hash(text)
        with self.lock:
            if text_hash in self.cache:
                self.access_times[text_hash] = time.time()
                return self.cache[text_hash]
        return None
    
    def put(self, text: str, embedding: List[float]):
        """Put embedding in cache"""
        text_hash = self._get_hash(text)
        with self.lock:
            # Remove oldest item if cache is full
            if len(self.cache) >= self.max_size:
                oldest_hash = min(self.access_times.keys(), key=lambda k: self.access_times[k])
                del self.cache[oldest_hash]
                del self.access_times[oldest_hash]
            
            self.cache[text_hash] = embedding
            self.access_times[text_hash] = time.time()
            
            # Periodically save cache
            if len(self.cache) % 100 == 0:
                self._save_cache()

# Global embedding cache
embedding_cache = OptimizedEmbeddingCache()

class OptimizedYouTubeWorkflowService:
    def __init__(self):
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if self.youtube_api_key:
            self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
        else:
            self.youtube = None
            logger.warning("YouTube API key not found. Suggestions feature will be limited.")
        
        # Initialize Supadata client with connection pooling
        supadata_api_key = os.getenv("SUPADATA_API_KEY")
        if not supadata_api_key:
            raise ValueError("SUPADATA_API_KEY environment variable is required")
        
        self.supadata = Supadata(api_key=supadata_api_key)
        
        # Thread pools for different operations
        self.embedding_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="embedding")
        self.io_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="io")
        
        # Initialize or get existing table
        self._setup_database()
        
        logger.info("Optimized YouTube service initialized successfully")

    def _setup_database(self):
        """Setup database with optimizations"""
        try:
            # Try to open existing table
            self.table = db.open_table("transcripts")
            logger.info("Opened existing transcripts table")
        except:
            # Create new table if doesn't exist
            self.table = db.create_table("transcripts", schema=schema)
            logger.info("Created new transcripts table")
            
        # Create indexes for better query performance
        try:
            # These may fail if indexes already exist, which is fine
            self.table.create_index("video_id")
            self.table.create_index("text_hash")
        except:
            pass

    @lru_cache(maxsize=1000)
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL with caching"""
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

    def _parse_supadata_transcript_optimized(self, content: List[Any]) -> List[TranscriptSegment]:
        """Optimized transcript parsing with batch processing"""
        segments = []
        
        # Pre-allocate list for better performance
        segments_data = []
        
        for item in content:
            try:
                text = item.text.strip()
                if not text or len(text) < 3:  # Skip very short segments
                    continue
                    
                start = float(item.offset) / 1000
                duration = float(item.duration) / 1000
                
                segments_data.append((text, start, duration))
                
            except Exception as e:
                logger.debug(f"Skipping invalid segment: {e}")
                continue
        
        # Batch create segments
        segments = [TranscriptSegment(text=text, start=start, duration=duration) 
                   for text, start, duration in segments_data]
        
        if not segments:
            raise ValueError("No valid transcript segments found after parsing")
        
        logger.info(f"Parsed {len(segments)} transcript segments")
        return segments

    async def get_transcript_async(self, youtube_url: str) -> List[TranscriptSegment]:
        """Async transcript retrieval"""
        video_id = self.extract_video_id(youtube_url)
        logger.info(f"Extracting transcript for video {video_id} using Supadata AI...")

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            transcript = await loop.run_in_executor(
                self.io_executor,
                lambda: self.supadata.youtube.transcript(video_id=video_id, text=False)
            )

            if not transcript or not hasattr(transcript, 'content'):
                raise ValueError("No transcript content returned from Supadata")

            transcript_content = transcript.content
            if not transcript_content:
                raise ValueError("Empty transcript content returned from Supadata")

            logger.info("Successfully retrieved transcript content from Supadata")
            
            # Parse in thread pool
            segments = await loop.run_in_executor(
                self.io_executor,
                self._parse_supadata_transcript_optimized,
                transcript_content
            )
            
            return segments

        except Exception as e:
            error_msg = f"Failed to get transcript for video {video_id} using Supadata: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_transcript(self, youtube_url: str) -> List[TranscriptSegment]:
        """Synchronous wrapper for transcript retrieval"""
        return asyncio.run(self.get_transcript_async(youtube_url))

    def chunk_transcript_optimized(self, segments: List[TranscriptSegment]) -> List[Document]:
        """Optimized transcript chunking with intelligent segmentation"""
        if not segments:
            return []

        # Use more efficient chunking strategy
        documents = []
        current_chunk_text = []
        current_chunk_start = segments[0].start
        current_chunk_duration = 0
        chunk_size_limit = 800  # Smaller chunks for better performance
        
        for segment in segments:
            segment_text = segment.text.strip()
            if not segment_text:
                continue
                
            # Check if adding this segment would exceed limit
            potential_text = " ".join(current_chunk_text + [segment_text])
            
            if len(potential_text) > chunk_size_limit and current_chunk_text:
                # Create document from current chunk
                chunk_text = " ".join(current_chunk_text)
                if chunk_text.strip():
                    documents.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "start": current_chunk_start,
                            "duration": current_chunk_duration
                        }
                    ))
                
                # Start new chunk
                current_chunk_text = [segment_text]
                current_chunk_start = segment.start
                current_chunk_duration = segment.duration
            else:
                current_chunk_text.append(segment_text)
                current_chunk_duration += segment.duration
        
        # Add final chunk
        if current_chunk_text:
            chunk_text = " ".join(current_chunk_text)
            if chunk_text.strip():
                documents.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "start": current_chunk_start,
                        "duration": current_chunk_duration
                    }
                ))
        
        logger.info(f"Created {len(documents)} optimized chunks")
        return documents

    def embed_text_cached(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embeddings with caching"""
        if not text.strip():
            return [0.0] * embedding_dim
        
        # Check cache first
        cached_embedding = embedding_cache.get(text)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type=task_type
            )
            embedding = result["embedding"]

            if len(embedding) != embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {embedding_dim}, got {len(embedding)}")

            # Cache the result
            embedding_cache.put(text, embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * embedding_dim

    def embed_texts_batch_optimized(self, texts: List[str], max_workers: int = 8) -> List[List[float]]:
        """Optimized batch embedding with caching and deduplication"""
        if not texts:
            return []
        
        # Deduplicate texts while preserving order
        unique_texts = []
        text_to_indices = {}
        
        for i, text in enumerate(texts):
            if text not in text_to_indices:
                text_to_indices[text] = []
                unique_texts.append(text)
            text_to_indices[text].append(i)
        
        logger.info(f"Processing {len(unique_texts)} unique texts (reduced from {len(texts)})")
        
        # Process unique texts in parallel
        embeddings_dict = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_text = {
                executor.submit(self.embed_text_cached, text): text 
                for text in unique_texts
            }
            
            for future in as_completed(future_to_text):
                text = future_to_text[future]
                try:
                    embedding = future.result()
                    embeddings_dict[text] = embedding
                except Exception as e:
                    logger.error(f"Failed to embed text: {e}")
                    embeddings_dict[text] = [0.0] * embedding_dim
        
        # Reconstruct embeddings in original order
        result_embeddings = []
        for text in texts:
            result_embeddings.append(embeddings_dict[text])
        
        return result_embeddings

    def is_video_indexed_fast(self, video_id: str) -> bool:
        """Fast video indexing check using count with filter"""
        try:
            count = self.table.count_rows(filter=f"video_id = '{video_id}'")
            return count > 0
        except Exception as e:
            logger.error(f"Error checking if video is indexed: {e}")
            return False

    async def index_video_async(self, youtube_url: str, video_id: str) -> bool:
        """Async video indexing for better performance"""
        if self.is_video_indexed_fast(video_id):
            logger.info(f"Video {video_id} is already indexed.")
            return True

        try:
            start_time = time.time()
            
            # Get transcript asynchronously
            segments = await self.get_transcript_async(youtube_url)
            transcript_time = time.time() - start_time
            logger.info(f"Transcript retrieval took {transcript_time:.2f}s")

            if not segments:
                logger.warning(f"No transcript segments found for video {video_id}")
                return False

            # Chunk transcript
            chunk_start = time.time()
            chunks = self.chunk_transcript_optimized(segments)
            chunk_time = time.time() - chunk_start
            logger.info(f"Chunking took {chunk_time:.2f}s")

            if not chunks:
                logger.warning(f"No chunks created for video {video_id}")
                return False

            # Generate embeddings
            embed_start = time.time()
            texts = [doc.page_content for doc in chunks]
            embeddings = self.embed_texts_batch_optimized(texts, max_workers=10)
            embed_time = time.time() - embed_start
            logger.info(f"Embedding generation took {embed_time:.2f}s")

            # Prepare data for batch insertion
            data_prep_start = time.time()
            current_time = pa.scalar(int(time.time() * 1000000), type=pa.timestamp('us'))
            
            data = []
            for chunk, embedding in zip(chunks, embeddings):
                if embedding and any(x != 0.0 for x in embedding):
                    text_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
                    data.append({
                        "id": str(uuid.uuid4()),
                        "video_id": video_id,
                        "text": chunk.page_content,
                        "start": chunk.metadata["start"],
                        "duration": chunk.metadata["duration"],
                        "vector": embedding,
                        "text_hash": text_hash,
                        "created_at": current_time
                    })
            
            data_prep_time = time.time() - data_prep_start
            logger.info(f"Data preparation took {data_prep_time:.2f}s")

            if not data:
                logger.warning(f"No valid embeddings generated for video {video_id}")
                return False

            # Batch insert
            insert_start = time.time()
            self.table.add(data)
            insert_time = time.time() - insert_start
            logger.info(f"Database insertion took {insert_time:.2f}s")

            total_time = time.time() - start_time
            logger.info(f"Successfully indexed video {video_id} in {total_time:.2f}s total")
            return True

        except Exception as e:
            logger.error(f"Failed to index video {video_id}: {e}")
            return False

    def index_video(self, youtube_url: str, video_id: str) -> bool:
        """Synchronous wrapper for video indexing"""
        return asyncio.run(self.index_video_async(youtube_url, video_id))

    def search_video_content_optimized(self, query: str, video_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Optimized video content search with caching"""
        try:
            search_start = time.time()
            
            # Generate query embedding with caching
            query_embedding = self.embed_text_cached(query, task_type="retrieval_query")

            if not query_embedding or all(x == 0.0 for x in query_embedding):
                logger.error("Failed to generate valid query embedding")
                return []

            # Optimized vector search with pre-filtering
            results = (
                self.table.search(query_embedding, vector_column_name="vector")
                .where(f"video_id = '{video_id}'")
                .limit(min(top_k * 3, 50))  # Limit to reasonable number for reranking
                .to_list()
            )

            search_time = time.time() - search_start
            logger.info(f"Vector search took {search_time:.2f}s")

            if not results:
                return []

            # Fast reranking with timeout
            rerank_start = time.time()
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

                rerank_time = time.time() - rerank_start
                logger.info(f"Reranking took {rerank_time:.2f}s")
                return final_results

            except Exception as e:
                logger.warning(f"Reranking failed, using vector similarity: {e}")
                # Fast fallback without reranking
                return [{
                    "text": doc["text"],
                    "start": doc["start"],
                    "duration": doc["duration"],
                    "score": 1.0 - (i * 0.05),  # Simple similarity-based scoring
                    "url": f"https://www.youtube.com/watch?v={video_id}&t={int(doc['start'])}s"
                } for i, doc in enumerate(results[:top_k])]

        except Exception as e:
            logger.error(f"Error searching video content: {e}")
            return []

    @lru_cache(maxsize=500)
    def get_video_info_cached(self, youtube_url: str) -> Dict[str, Any]:
        """Cached video info retrieval"""
        video_id = self.extract_video_id(youtube_url)
        
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
        
        # Fallback
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
            duration_str = duration_str.replace('PT', '')
            total_seconds = 0
            
            if 'H' in duration_str:
                hours = int(duration_str.split('H')[0])
                total_seconds += hours * 3600
                duration_str = duration_str.split('H')[1]
            
            if 'M' in duration_str:
                minutes = int(duration_str.split('M')[0])
                total_seconds += minutes * 60
                duration_str = duration_str.split('M')[1]
            
            if 'S' in duration_str:
                seconds = int(duration_str.split('S')[0])
                total_seconds += seconds
            
            return total_seconds
        except Exception:
            return 0

    def cleanup_old_cache(self, days_old: int = 30):
        """Clean up old cached data"""
        try:
            cutoff_time = int((time.time() - (days_old * 24 * 3600)) * 1000000)
            deleted = self.table.delete(f"created_at < {cutoff_time}")
            logger.info(f"Cleaned up {deleted} old records")
        except Exception as e:
            logger.error(f"Failed to cleanup old cache: {e}")

    def __del__(self):
        """Cleanup resources"""
        try:
            self.embedding_executor.shutdown(wait=False)
            self.io_executor.shutdown(wait=False)
            embedding_cache._save_cache()
        except:
            pass


# Optimized utility functions
def print_video_segments_fast(segments: List[Dict[str, Any]], video_title: str = ""):
    """Optimized segment printing"""
    if not segments:
        print("❌ No relevant segments found.")
        return

    if video_title:
        print(f"\n🎥 Video: {video_title}")

    print(f"🎯 Found {len(segments)} relevant segments:")
    print("-" * 60)

    for i, segment in enumerate(segments, 1):
        # More efficient text truncation
        text_words = segment['text'].split()
        text_preview = " ".join(text_words[:25])
        if len(text_words) > 25:
            text_preview += "..."

        print(f"\n{i}. ⏱️  {int(segment['start'])}s | 📊 {segment['score']:.3f}")
        print(f"   📝 {text_preview}")
        print(f"   🔗 {segment['url']}")


# Example usage with performance monitoring
if __name__ == "__main__":
    import time
    
    try:
        start_time = time.time()
        service = OptimizedYouTubeWorkflowService()
        init_time = time.time() - start_time
        print(f"Service initialization took {init_time:.2f}s")
        
        # Test with a video URL
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = service.extract_video_id(url)
        
        # Get video info
        info_start = time.time()
        info = service.get_video_info_cached(url)
        info_time = time.time() - info_start
        print(f"Video info retrieval took {info_time:.2f}s")
        print(f"Video Info: {info['title']}")
        
        # Index the video
        index_start = time.time()
        success = service.index_video(url, video_id)
        index_time = time.time() - index_start
        
        if success:
            print(f"Successfully indexed video {video_id} in {index_time:.2f}s")
            
            # Search within the video
            search_start = time.time()
            results = service.search_video_content_optimized("your search query", video_id, top_k=3)
            search_time = time.time() - search_start
            print(f"Search took {search_time:.2f}s")
            
            print_video_segments_fast(results, info.get('title', ''))
        else:
            print(f"Failed to index video {video_id}")
            
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f}s")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have set the required environment variables")