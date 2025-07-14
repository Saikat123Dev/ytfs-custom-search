import os
import re
import uuid
import logging
import numpy as np
import pyarrow as pa
import time
import random
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.generativeai as genai
import voyageai
import lancedb
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# === Moved TranscriptSegment class here to resolve NameError ===
class TranscriptSegment:
    def __init__(self, text: str, start: float, duration: float):
        self.text = text
        self.start = start
        self.duration = duration

# Load environment variables
load_dotenv()

# Configure APIs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure AI services
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# LanceDB setup
embedding_dim = 768
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db = lancedb.connect(os.path.join(BASE_DIR, "lancedb_data"))
print('db', db)

schema = pa.schema([
    ("id", pa.string()),
    ("video_id", pa.string()),
    ("text", pa.string()),
    ("start", pa.float32()),
    ("duration", pa.float32()),
    ("vector", pa.list_(pa.float32(), embedding_dim))
])

try:
    table = db.open_table("transcripts")
    logger.info("Successfully opened existing 'transcripts' table.")
except ValueError:
    logger.info("Table 'transcripts' not found. Creating new table.")
    try:
        db.drop_table("transcripts", ignore_missing=True)
    except Exception as e:
        logger.debug(f"Error dropping table (this is expected if table doesn't exist): {e}")
    table = db.create_table("transcripts", schema=schema)
    logger.info("Successfully created 'transcripts' table.")

class RateLimitedSession:
    """Enhanced session with advanced rate limiting and anti-bot detection"""
    def __init__(self, requests_per_minute: int = 15):  # Reduced from 30 to 15
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0
        self.request_count = 0
        self.session_start_time = time.time()
        # Create session with enhanced retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=3,  # Increased from 2 to 3
            status_forcelist=[429, 500, 502, 503, 504, 403],  # Added 403
            allowed_methods=["GET", "POST"],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=1, pool_maxsize=1)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        # Enhanced headers to mimic real browser behavior
        self.update_headers()
        # Connection pooling settings
        self.session.keep_alive = False

    def update_headers(self):
        """Update headers with realistic browser simulation"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        accept_languages = [
            'en-US,en;q=0.9',
            'en-GB,en;q=0.9',
            'en-US,en;q=0.8,es;q=0.7',
            'en-GB,en-US;q=0.8,en;q=0.7'
        ]
        self.session.headers.update({
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': random.choice(accept_languages),
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Charset': 'UTF-8,*;q=0.7',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
            'Sec-GPC': '1'
        })

    def wait_if_needed(self):
        """Enhanced wait logic with progressive delays"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        # Progressive delay based on request count
        base_delay = self.min_interval
        if self.request_count > 50:
            base_delay *= 2
        elif self.request_count > 100:
            base_delay *= 3
        if time_since_last < base_delay:
            wait_time = base_delay - time_since_last
            # Add more jitter to avoid detection
            wait_time += random.uniform(0.5, 2.0)
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds (request #{self.request_count})")
            time.sleep(wait_time)
        # Additional random delay every 10 requests
        if self.request_count % 10 == 0 and self.request_count > 0:
            extra_delay = random.uniform(2.0, 5.0)
            logger.info(f"Extra delay: {extra_delay:.2f} seconds")
            time.sleep(extra_delay)
        # Update headers occasionally to simulate browser behavior
        if self.request_count % 25 == 0:
            self.update_headers()
            logger.info("Updated headers to simulate new browser session")
        self.last_request_time = time.time()
        self.request_count += 1

    def make_request(self, url: str, **kwargs) -> requests.Response:
        """Make a request with all anti-detection measures"""
        self.wait_if_needed()
        # Add random delay before actual request
        time.sleep(random.uniform(0.1, 0.5))
        try:
            response = self.session.get(url, timeout=30, **kwargs)
            # Check for rate limiting
            if response.status_code == 429:
                logger.warning("Rate limited, backing off significantly")
                time.sleep(random.uniform(30, 60))
                raise requests.exceptions.RetryError("Rate limited")
            return response
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

class YouTubeWorkflowService:
    def __init__(self):
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        if self.youtube_api_key:
            self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
        else:
            self.youtube = None
            logger.warning("YouTube API key not found. Suggestions feature will be limited.")
        # Initialize enhanced rate-limited session
        self.rate_limiter = RateLimitedSession(requests_per_minute=10)  # Very conservative
        # Track failed videos to avoid repeated attempts
        self.failed_videos = set()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3

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

    def exponential_backoff_retry(self, func, max_retries: int = 3, base_delay: float = 2.0):
        """Enhanced retry function with circuit breaker pattern"""
        # Circuit breaker: if too many consecutive failures, pause longer
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.warning(f"Circuit breaker activated. Pausing for 5 minutes after {self.consecutive_failures} failures")
            time.sleep(300)  # 5 minutes
            self.consecutive_failures = 0
        for attempt in range(max_retries):
            try:
                result = func()
                self.consecutive_failures = 0  # Reset on success
                return result
            except Exception as e:
                self.consecutive_failures += 1
                if attempt == max_retries - 1:
                    logger.error(f"All retry attempts failed. Consecutive failures: {self.consecutive_failures}")
                    raise e
                if any(phrase in str(e).lower() for phrase in ["429", "too many requests", "rate limit", "quota"]):
                    # Much longer delay for rate limit errors
                    delay = base_delay * (3 ** attempt) + random.uniform(5, 15)
                    logger.warning(f"Rate limited. Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    # Standard backoff for other errors
                    delay = base_delay * (2 ** attempt) + random.uniform(1, 3)
                    logger.warning(f"Error occurred. Retrying in {delay:.2f} seconds: {e}")
                    time.sleep(delay)

    def get_transcript_with_fallback(self, video_id: str) -> List[TranscriptSegment]:
        """Get transcript with enhanced anti-detection measures"""
        # Check if we've already failed on this video recently
        if video_id in self.failed_videos:
            logger.info(f"Skipping video {video_id} - previously failed")
            raise ValueError(f"Video {video_id} previously failed transcript extraction")
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()

        def attempt_transcript_fetch():
            # Language preference order
            language_preferences = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
            logger.info(f"Attempting to fetch transcript for video {video_id}")
            # Add random delay to simulate human behavior
            time.sleep(random.uniform(1, 3))
            # Method 1: Try preferred languages with retries
            for lang in language_preferences:
                try:
                    logger.info(f"Trying language: {lang}")
                    # Additional delay between language attempts
                    if lang != language_preferences[0]:
                        time.sleep(random.uniform(2, 4))
                    raw_transcript = YouTubeTranscriptApi.get_transcript(
                        video_id, 
                        languages=[lang],
                        preserve_formatting=True
                    )
                    logger.info(f"Successfully retrieved transcript in {lang}")
                    return raw_transcript, f"Language: {lang}"
                except (NoTranscriptFound, TranscriptsDisabled):
                    continue
                except Exception as e:
                    if any(phrase in str(e).lower() for phrase in ["429", "too many requests", "rate limit"]):
                        logger.warning(f"Rate limited on language {lang}, waiting longer...")
                        time.sleep(random.uniform(30, 60))
                        raise e
                    logger.warning(f"Error with language {lang}: {e}")
                    continue
            # Method 2: Auto-detect language with longer delay
            try:
                logger.info("Trying auto-detection")
                time.sleep(random.uniform(3, 6))
                raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
                logger.info("Successfully retrieved transcript with auto-detection")
                return raw_transcript, "Auto-detected"
            except Exception as e:
                if any(phrase in str(e).lower() for phrase in ["429", "too many requests", "rate limit"]):
                    logger.warning("Rate limited on auto-detection")
                    time.sleep(random.uniform(30, 60))
                    raise e
                logger.warning(f"Auto-detection failed: {e}")
            # Method 3: Get first available transcript
            try:
                logger.info("Trying first available transcript")
                time.sleep(random.uniform(2, 5))
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                for transcript in transcript_list:
                    try:
                        time.sleep(random.uniform(1, 2))
                        raw_transcript = transcript.fetch()
                        logger.info(f"Retrieved transcript in {transcript.language_code}")
                        return raw_transcript, f"Available: {transcript.language_code}"
                    except Exception as e:
                        if any(phrase in str(e).lower() for phrase in ["429", "too many requests", "rate limit"]):
                            raise e
                        continue
            except Exception as e:
                if any(phrase in str(e).lower() for phrase in ["429", "too many requests", "rate limit"]):
                    raise e
                logger.warning(f"Failed to get available transcripts: {e}")
            # If all methods fail
            raise ValueError(f"No transcript available for video {video_id}")

        try:
            # Use enhanced exponential backoff retry
            raw_transcript, source = self.exponential_backoff_retry(
                attempt_transcript_fetch,
                max_retries=2,  # Reduced retries
                base_delay=5.0  # Increased base delay
            )
            if not raw_transcript:
                raise ValueError(f"Empty transcript for video {video_id}")
            logger.info(f"Transcript source: {source}, segments: {len(raw_transcript)}")
            # Convert to TranscriptSegment objects
            segments = []
            for segment in raw_transcript:
                text = segment.get('text', '').strip()
                start = float(segment.get('start', 0))
                duration = float(segment.get('duration', 0))
                if text:
                    segments.append(TranscriptSegment(text=text, start=start, duration=duration))
            if not segments:
                raise ValueError(f"No valid segments found for video {video_id}")
            logger.info(f"Successfully processed {len(segments)} segments")
            return segments
        except Exception as e:
            # Add to failed videos to avoid retrying
            self.failed_videos.add(video_id)
            logger.error(f"Failed to get transcript for video {video_id}: {e}")
            raise

    def chunk_transcript(self, segments: List[TranscriptSegment]) -> List[Document]:
        """Chunk transcript into manageable pieces for embedding"""
        if not segments:
            return []
        # Create indexed text for chunking
        full_text = "\n".join(f"{i}|{seg.start}|{seg.duration}|{seg.text}" 
                             for i, seg in enumerate(segments))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n", ".", "!", "?", ",", " "]
        )
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
        """Generate embeddings for text using Google's embedding model with enhanced error handling"""
        try:
            if not text.strip():
                logger.warning("Empty text provided for embedding")
                return [0.0] * embedding_dim
            # Add retry logic for embedding API with longer delays
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Add delay before embedding requests
                    if attempt > 0:
                        delay = (2 ** attempt) * 3 + random.uniform(1, 3)
                        logger.info(f"Embedding retry delay: {delay:.2f} seconds")
                        time.sleep(delay)
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
                    if attempt == max_retries - 1:
                        logger.error(f"All embedding attempts failed: {e}")
                        raise e
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * embedding_dim

    def embed_texts_batch(self, texts: List[str], max_workers: int = 2) -> List[List[float]]:
        """Generate embeddings for multiple texts with very conservative concurrency"""
        if not texts:
            return []
        # Further reduced max_workers and added delays
        embeddings = []
        # Process sequentially to avoid overwhelming APIs
        for i, text in enumerate(texts):
            try:
                # Add delay between each embedding request
                if i > 0:
                    time.sleep(random.uniform(1, 3))
                embedding = self.embed_text(text)
                embeddings.append(embedding)
                # Progress logging
                if (i + 1) % 5 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} embeddings")
            except Exception as e:
                logger.error(f"Failed to embed text at index {i}: {e}")
                embeddings.append([0.0] * embedding_dim)
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
        """Index a video's transcript in the database with enhanced error handling"""
        if self.is_video_indexed(video_id):
            logger.info(f"Video {video_id} is already indexed.")
            return True
        try:
            logger.info(f"Starting indexing process for video {video_id}")
            # Add initial delay to avoid burst requests
            time.sleep(random.uniform(2, 5))
            # Get transcript with retries and rate limiting
            segments = self.get_transcript_with_fallback(youtube_url)
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
                if embedding and any(x != 0.0 for x in embedding):
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
                .limit(top_k * 2)
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
                return [{
                    "text": doc["text"],
                    "start": doc["start"],
                    "duration": doc["duration"],
                    "score": 1.0 - (i * 0.1),
                    "url": f" https://www.youtube.com/watch?v={video_id}&t={int(doc['start'])}s"
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
            # Add delay before API call
            time.sleep(random.uniform(1, 2))
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
                    'url': f" https://www.youtube.com/watch?v={item['id']['videoId']}"
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
        if not self.youtube:
            return f"Video {video_id}"
        try:
            # Add delay before API call
            time.sleep(random.uniform(0.5, 1.5))
            response = self.youtube.videos().list(
                part='snippet',
                id=video_id
            ).execute()
            if response['items']:
                return response['items'][0]['snippet']['title']
            else:
                return f"Video {video_id}"
        except Exception as e:
            logger.error(f"Error getting video title: {e}")
            return f"Video {video_id}"

    def clear_failed_videos(self):
        """Clear the failed videos cache"""
        self.failed_videos.clear()
        self.consecutive_failures = 0
        logger.info("Cleared failed videos cache and reset failure counter")

    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        return {
            "consecutive_failures": self.consecutive_failures,
            "failed_videos_count": len(self.failed_videos),
            "request_count": self.rate_limiter.request_count,
            "session_uptime": time.time() - self.rate_limiter.session_start_time
        }

def print_video_segments(segments: List[Dict[str, Any]], video_title: str = ""):
    """Print formatted video segments"""
    if not segments:
        print("❌ No relevant segments found.")
        return
    if video_title:
        print(f"\n✅ Video: {video_title}")
    print(f"📌 Found {len(segments)} relevant segments:")
    print("-" * 60)
    for i, segment in enumerate(segments, 1):
        text_preview = " ".join(segment['text'].split()[:30])
        if len(segment['text'].split()) > 30:
            text_preview += "..."
        print(f"\n{i}. ⏱ Timestamp: {int(segment['start'])}s")
        print(f"   🔍 Score: {segment['score']:.4f}")
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
        desc_preview = video['description'][:100]
        if len(video['description']) > 100:
            desc_preview += "..."
        print(f"\n{i}. 🎬 {video['title']}")
        print(f"   📺 Channel: {video['channel_title']}")
        print(f"   📅 Published: {video['published_at'][:10]}")
        print(f"   📄 {desc_preview}")
        print(f"   🔗 {video['url']}")