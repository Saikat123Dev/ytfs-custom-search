import os
import re
import uuid
import logging
import asyncio
import aiohttp
import numpy as np
import pyarrow as pa
import time
import random
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Tuple
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
from functools import wraps
from datetime import datetime, timedelta
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure AI services
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# Constants
EMBEDDING_DIM = 768
MAX_RETRIES = 5
BASE_DELAY = 2.0
REQUESTS_PER_MINUTE = 10  # Very conservative for production
EMBEDDING_BATCH_SIZE = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# LanceDB setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db = lancedb.connect(os.path.join(BASE_DIR, "lancedb_data"))

schema = pa.schema([
    ("id", pa.string()),
    ("video_id", pa.string()),
    ("text", pa.string()),
    ("start", pa.float32()),
    ("duration", pa.float32()),
    ("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
    ("created_at", pa.timestamp('ms')),
    ("chunk_index", pa.int32())
])

# Initialize table with better error handling
try:
    table = db.open_table("transcripts")
    logger.info("Successfully opened existing 'transcripts' table.")
except Exception:
    logger.info("Creating new 'transcripts' table.")
    try:
        db.drop_table("transcripts", ignore_missing=True)
        table = db.create_table("transcripts", schema=schema)
        logger.info("Successfully created 'transcripts' table.")
    except Exception as e:
        logger.error(f"Failed to create table: {e}")
        raise


class CircuitBreaker:
    """Circuit breaker pattern for handling consecutive failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and 
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on 429 responses"""
    
    def __init__(self, initial_requests_per_minute: int = 10):
        self.requests_per_minute = initial_requests_per_minute
        self.min_interval = 60.0 / self.requests_per_minute
        self.last_request_time = 0
        self.consecutive_429s = 0
        self.last_adjustment_time = time.time()
        
    def wait_if_needed(self):
        """Wait with adaptive delays based on recent 429 responses"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Adaptive interval based on recent 429 errors
        adaptive_interval = self.min_interval * (1.5 ** self.consecutive_429s)
        
        if time_since_last < adaptive_interval:
            wait_time = adaptive_interval - time_since_last
            # Add jitter to prevent thundering herd
            wait_time += random.uniform(0.5, 1.5)
            logger.info(f"Adaptive rate limiting: waiting {wait_time:.2f}s (429 count: {self.consecutive_429s})")
            time.sleep(wait_time)
        
        self.last_request_time = time.time()
    
    def on_429_response(self):
        """Called when a 429 response is received"""
        self.consecutive_429s += 1
        if self.consecutive_429s > 3:
            # Significantly slow down
            self.requests_per_minute = max(1, self.requests_per_minute // 2)
            self.min_interval = 60.0 / self.requests_per_minute
            logger.warning(f"Reduced rate limit to {self.requests_per_minute} requests/minute")
    
    def on_success_response(self):
        """Called when a successful response is received"""
        if self.consecutive_429s > 0:
            self.consecutive_429s = max(0, self.consecutive_429s - 1)
            # Gradually increase rate if no recent 429s
            if self.consecutive_429s == 0 and time.time() - self.last_adjustment_time > 300:
                self.requests_per_minute = min(20, self.requests_per_minute + 1)
                self.min_interval = 60.0 / self.requests_per_minute
                self.last_adjustment_time = time.time()


class ImprovedTranscriptSegment:
    """Enhanced transcript segment with additional metadata"""
    
    def __init__(self, text: str, start: float, duration: float, confidence: float = 1.0):
        self.text = text.strip()
        self.start = start
        self.duration = duration
        self.confidence = confidence
        self.word_count = len(self.text.split())
        self.char_count = len(self.text)
    
    def is_valid(self) -> bool:
        """Check if segment is valid for processing"""
        return (
            self.text and 
            len(self.text.strip()) > 3 and
            self.word_count > 1 and
            self.duration > 0
        )


class YouTubeTranscriptService:
    """Enhanced YouTube transcript service with improved error handling and rate limiting"""
    
    def __init__(self):
        self.youtube_api_key = os.getenv("YOUTUBE_API_KEY")
        self.youtube = None
        if self.youtube_api_key:
            try:
                self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
                logger.info("YouTube API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize YouTube API: {e}")
        
        # Initialize rate limiting and circuit breaker
        self.rate_limiter = AdaptiveRateLimiter(initial_requests_per_minute=REQUESTS_PER_MINUTE)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=600)
        
        # Track failed videos with expiration
        self.failed_videos = {}
        self.failed_video_expiry = 3600  # 1 hour
        
        # Initialize session with better retry configuration
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with optimized retry configuration"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            respect_retry_after_header=True
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Enhanced headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        return session
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL with enhanced patterns"""
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11})",
            r"youtu\.be\/([0-9A-Za-z_-]{11})",
            r"embed\/([0-9A-Za-z_-]{11})",
            r"watch\?v=([0-9A-Za-z_-]{11})",
            r"shorts\/([0-9A-Za-z_-]{11})",
            r"live\/([0-9A-Za-z_-]{11})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                video_id = match.group(1)
                logger.debug(f"Extracted video ID: {video_id}")
                return video_id
        
        # If it's already a video ID
        if re.match(r"^[0-9A-Za-z_-]{11}$", url):
            return url
        
        raise ValueError(f"Invalid YouTube URL or video ID: {url}")
    
    def _is_video_failed_recently(self, video_id: str) -> bool:
        """Check if video failed recently and shouldn't be retried"""
        if video_id not in self.failed_videos:
            return False
        
        failure_time = self.failed_videos[video_id]
        if time.time() - failure_time > self.failed_video_expiry:
            # Remove expired failure
            del self.failed_videos[video_id]
            return False
        
        return True
    
    def _mark_video_as_failed(self, video_id: str):
        """Mark video as failed with timestamp"""
        self.failed_videos[video_id] = time.time()
    
    def get_transcript_with_enhanced_fallback(self, video_id: str) -> List[ImprovedTranscriptSegment]:
        """Get transcript with comprehensive fallback strategy"""
        
        if self._is_video_failed_recently(video_id):
            logger.info(f"Skipping video {video_id} - recently failed")
            raise ValueError(f"Video {video_id} recently failed transcript extraction")
        
        def attempt_transcript_fetch():
            self.rate_limiter.wait_if_needed()
            
            # Language preference with expanded options
            language_preferences = [
                ['en'],
                ['en-US'],
                ['en-GB'],
                ['en-CA'],
                ['en-AU'],
                ['en-IN'],
                ['en-ZA']
            ]
            
            logger.info(f"Attempting transcript fetch for video {video_id}")
            
            # Method 1: Try specific languages
            for lang_list in language_preferences:
                try:
                    logger.debug(f"Trying languages: {lang_list}")
                    raw_transcript = YouTubeTranscriptApi.get_transcript(
                        video_id, 
                        languages=lang_list,
                        preserve_formatting=True
                    )
                    logger.info(f"Successfully retrieved transcript in {lang_list[0]}")
                    self.rate_limiter.on_success_response()
                    return raw_transcript, f"Language: {lang_list[0]}"
                except (NoTranscriptFound, TranscriptsDisabled):
                    continue
                except Exception as e:
                    if "429" in str(e) or "Too Many Requests" in str(e):
                        self.rate_limiter.on_429_response()
                        raise e
                    logger.debug(f"Error with language {lang_list}: {e}")
                    continue
            
            # Method 2: Auto-detect with manual transcript preference
            try:
                logger.debug("Trying auto-detection")
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                # Prefer manually created transcripts
                manual_transcripts = [t for t in transcript_list if not t.is_generated]
                if manual_transcripts:
                    transcript = manual_transcripts[0]
                    raw_transcript = transcript.fetch()
                    logger.info(f"Retrieved manual transcript in {transcript.language_code}")
                    self.rate_limiter.on_success_response()
                    return raw_transcript, f"Manual: {transcript.language_code}"
                
                # Fall back to auto-generated
                raw_transcript = YouTubeTranscriptApi.get_transcript(video_id)
                logger.info("Successfully retrieved auto-generated transcript")
                self.rate_limiter.on_success_response()
                return raw_transcript, "Auto-generated"
                
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    self.rate_limiter.on_429_response()
                    raise e
                logger.debug(f"Auto-detection failed: {e}")
            
            # Method 3: Try any available transcript
            try:
                logger.debug("Trying any available transcript")
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                for transcript in transcript_list:
                    try:
                        raw_transcript = transcript.fetch()
                        logger.info(f"Retrieved transcript in {transcript.language_code}")
                        self.rate_limiter.on_success_response()
                        return raw_transcript, f"Available: {transcript.language_code}"
                    except Exception as e:
                        if "429" in str(e) or "Too Many Requests" in str(e):
                            self.rate_limiter.on_429_response()
                            raise e
                        continue
                        
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    self.rate_limiter.on_429_response()
                    raise e
                logger.debug(f"Failed to get any transcript: {e}")
            
            raise ValueError(f"No transcript available for video {video_id}")
        
        try:
            # Use circuit breaker pattern
            raw_transcript, source = self.circuit_breaker.call(
                self._exponential_backoff_retry,
                attempt_transcript_fetch,
                max_retries=MAX_RETRIES,
                base_delay=BASE_DELAY
            )
            
            if not raw_transcript:
                raise ValueError(f"Empty transcript for video {video_id}")
            
            logger.info(f"Transcript source: {source}, segments: {len(raw_transcript)}")
            
            # Convert to enhanced segments with validation
            segments = []
            for i, segment in enumerate(raw_transcript):
                try:
                    text = segment.get('text', '').strip()
                    start = float(segment.get('start', 0))
                    duration = float(segment.get('duration', 0))
                    
                    # Enhanced segment with confidence scoring
                    enhanced_segment = ImprovedTranscriptSegment(
                        text=text,
                        start=start,
                        duration=duration,
                        confidence=1.0 if 'confidence' not in segment else float(segment.get('confidence', 1.0))
                    )
                    
                    if enhanced_segment.is_valid():
                        segments.append(enhanced_segment)
                    else:
                        logger.debug(f"Skipping invalid segment {i}: {text[:50]}...")
                        
                except (ValueError, TypeError) as e:
                    logger.debug(f"Error processing segment {i}: {e}")
                    continue
            
            if not segments:
                raise ValueError(f"No valid segments found for video {video_id}")
            
            logger.info(f"Successfully processed {len(segments)} valid segments")
            return segments
            
        except Exception as e:
            self._mark_video_as_failed(video_id)
            logger.error(f"Failed to get transcript for video {video_id}: {e}")
            raise
    
    def _exponential_backoff_retry(self, func, max_retries: int = MAX_RETRIES, base_delay: float = BASE_DELAY):
        """Enhanced exponential backoff with jitter"""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                
                error_str = str(e).lower()
                if "429" in error_str or "too many requests" in error_str:
                    # Exponential backoff with jitter for rate limiting
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
                    # Add extra delay for 429 errors
                    delay *= 2
                    logger.warning(f"Rate limited. Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                elif "unavailable" in error_str or "disabled" in error_str:
                    # Don't retry for permanently unavailable videos
                    raise e
                else:
                    # Regular exponential backoff for other errors
                    delay = base_delay * (1.5 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(delay)
    
    def create_enhanced_chunks(self, segments: List[ImprovedTranscriptSegment]) -> List[Document]:
        """Create enhanced chunks with better context preservation"""
        if not segments:
            return []
        
        # Filter out low-quality segments
        quality_segments = [s for s in segments if s.confidence > 0.5 and s.word_count >= 2]
        
        if not quality_segments:
            logger.warning("No quality segments found after filtering")
            return segments  # Fall back to original segments
        
        # Create text with metadata for better chunking
        segment_texts = []
        for i, seg in enumerate(quality_segments):
            # Add contextual markers
            timestamp_marker = f"[{int(seg.start)}s]"
            segment_texts.append(f"{timestamp_marker} {seg.text}")
        
        full_text = " ".join(segment_texts)
        
        # Enhanced text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        chunks = splitter.split_text(full_text)
        
        documents = []
        for chunk_idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            
            # Extract timestamp information
            timestamp_matches = re.findall(r'\[(\d+)s\]', chunk)
            if timestamp_matches:
                start_time = float(timestamp_matches[0])
                end_time = float(timestamp_matches[-1]) if len(timestamp_matches) > 1 else start_time
            else:
                start_time = 0
                end_time = 0
            
            # Clean the text by removing timestamp markers
            clean_text = re.sub(r'\[\d+s\]\s*', '', chunk).strip()
            
            if len(clean_text) < 20:  # Skip very short chunks
                continue
            
            # Calculate quality score
            word_count = len(clean_text.split())
            quality_score = min(1.0, word_count / 50)  # Normalize by expected word count
            
            duration = end_time - start_time if end_time > start_time else 30  # Default duration
            
            documents.append(Document(
                page_content=clean_text,
                metadata={
                    "start": start_time,
                    "duration": duration,
                    "chunk_index": chunk_idx,
                    "quality_score": quality_score,
                    "word_count": word_count,
                    "char_count": len(clean_text),
                    "timestamp_count": len(timestamp_matches)
                }
            ))
        
        logger.info(f"Created {len(documents)} enhanced chunks from {len(quality_segments)} segments")
        return documents
    
    def embed_text_with_retry(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """Generate embeddings with enhanced retry logic"""
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * EMBEDDING_DIM
        
        # Truncate very long text
        max_length = 8000  # Conservative limit for Gemini
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.debug(f"Truncated text to {max_length} characters")
        
        for attempt in range(MAX_RETRIES):
            try:
                result = genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type=task_type
                )
                
                embedding = result["embedding"]
                
                if len(embedding) != EMBEDDING_DIM:
                    raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}")
                
                return embedding
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to generate embedding after {MAX_RETRIES} attempts: {e}")
                    return [0.0] * EMBEDDING_DIM
                
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {wait_time:.2f}s: {e}")
                time.sleep(wait_time)
    
    def embed_texts_batch_improved(self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> List[List[float]]:
        """Improved batch embedding with better error handling"""
        if not texts:
            return []
        
        embeddings = []
        
        # Process in smaller batches to avoid overwhelming the API
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.embed_text_with_retry(text)
                batch_embeddings.append(embedding)
                
                # Small delay between requests
                time.sleep(0.2)
            
            embeddings.extend(batch_embeddings)
            
            # Longer delay between batches
            if i + batch_size < len(texts):
                time.sleep(1.0)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return embeddings
    
    def is_video_indexed(self, video_id: str) -> bool:
        """Check if video is already indexed"""
        try:
            count = table.count_rows(filter=f"video_id = '{video_id}'")
            is_indexed = count > 0
            logger.debug(f"Video {video_id} indexed: {is_indexed} ({count} records)")
            return is_indexed
        except Exception as e:
            logger.error(f"Error checking if video is indexed: {e}")
            return False
    
    def index_video_enhanced(self, youtube_url: str, video_id: str = None) -> bool:
        """Enhanced video indexing with comprehensive error handling"""
        if not video_id:
            video_id = self.extract_video_id(youtube_url)
        
        if self.is_video_indexed(video_id):
            logger.info(f"Video {video_id} is already indexed")
            return True
        
        try:
            logger.info(f"Starting enhanced indexing for video {video_id}")
            
            # Get transcript with enhanced fallback
            segments = self.get_transcript_with_enhanced_fallback(video_id)
            
            if not segments:
                logger.warning(f"No transcript segments found for video {video_id}")
                return False
            
            # Create enhanced chunks
            chunks = self.create_enhanced_chunks(segments)
            
            if not chunks:
                logger.warning(f"No chunks created for video {video_id}")
                return False
            
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            texts = [doc.page_content for doc in chunks]
            embeddings = self.embed_texts_batch_improved(texts)
            
            # Prepare enhanced data for insertion
            data = []
            current_time = datetime.now()
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if embedding and any(abs(x) > 1e-6 for x in embedding):  # Check for non-zero embedding
                    data.append({
                        "id": str(uuid.uuid4()),
                        "video_id": video_id,
                        "text": chunk.page_content,
                        "start": float(chunk.metadata["start"]),
                        "duration": float(chunk.metadata["duration"]),
                        "vector": embedding,
                        "created_at": current_time,
                        "chunk_index": i
                    })
                else:
                    logger.warning(f"Skipping chunk {i} due to invalid embedding")
            
            if not data:
                logger.error(f"No valid embeddings generated for video {video_id}")
                return False
            
            # Insert data in batches
            batch_size = 10
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                try:
                    table.add(batch)
                    logger.debug(f"Inserted batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
                    return False
            
            logger.info(f"Successfully indexed video {video_id} with {len(data)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index video {video_id}: {e}")
            return False
    
    def search_video_content_enhanced(self, query: str, video_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced search with better ranking and filtering"""
        try:
            # Generate query embedding
            query_embedding = self.embed_text_with_retry(query, task_type="retrieval_query")
            
            if not query_embedding or all(abs(x) < 1e-6 for x in query_embedding):
                logger.error("Failed to generate valid query embedding")
                return []
            
            # Search with larger initial results for better reranking
            search_limit = min(top_k * 3, 50)
            
            results = (
                table.search(query_embedding, vector_column_name="vector")
                .where(f"video_id = '{video_id}'")
                .limit(search_limit)
                .to_list()
            )
            
            if not results:
                logger.info(f"No results found for query in video {video_id}")
                return []
            
            # Enhanced reranking with Voyage AI
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
                        "start": float(doc["start"]),
                        "duration": float(doc["duration"]),
                        "relevance_score": float(res.relevance_score),
                        "chunk_index": int(doc.get("chunk_index", 0)),
                        "url": f"https://www.youtube.com/watch?v={video_id}&t={int(doc['start'])}s"
                    })
                
                logger.info(f"Reranked {len(final_results)} results")
                return final_results
                
            except Exception as e:
                logger.warning(f"Reranking failed, using vector similarity: {e}")
                
                # Fallback to vector similarity scores
                final_results = []
                for i, doc in enumerate(results[:top_k]):
                    final_results.append({
                        "text": doc["text"],
                        "start": float(doc["start"]),
                        "duration": float(doc["duration"]),
                        "relevance_score": 1.0 - (i * 0.1),  # Approximate relevance
                        "chunk_index": int(doc.get("chunk_index", 0)),
                        "url": f"https://www.youtube.com/watch?v={video_id}&t={int(doc['start'])}s"
                    })
                
                return final_results
                
        except Exception as e:
            logger.error(f"Error searching video content: {e}")
            return []
    
    def search_youtube_videos_enhanced(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Enhanced YouTube video search with better error handling"""
        if not self.youtube:
            logger.error("YouTube API not available")
            return []
        
        try:
            search_response = self.youtube.search().list(
                part='id,snippet',
                q=query,
                type='video',
                maxResults=max_results,
                order='relevance',
                videoDefinition='any',
                videoCaption='any'  # Prefer videos with captions
            ).execute()
            
            videos = []
            for item in search_response['items']:
                video_data = {
                    'video_id': item['id']['videoId'],
                    'title': item['snippet']['title'],
                    'description': item['snippet']['description'],
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails'].get('medium', {}).get('url', ''),
                    'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}"
                }
                videos.append(video_data)
            
            logger.info(f"Found {len(videos)} videos for query: {query}")
            return videos
            
        except HttpError as e:
            logger.error(f"YouTube API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error searching YouTube: {e}")
            return []
    
    def get_video_metadata(self, video_id: str) -> Dict[str, Any]:
        """Get comprehensive video metadata"""
        if not self.youtube:
            return {"title": f"Video {video_id}", "duration": None}
        
        try:
            response = self.youtube.videos().list(
                part='snippet,contentDetails,statistics',
                id=video_id
            ).execute()
            
            if not response['items']:
                return {"title": f"Video {video_id}", "duration": None}
            
            video = response['items'][0]
            snippet = video['snippet']
            content_details = video['contentDetails']
            statistics = video.get('statistics', {})
            
            # Parse duration from ISO 8601 format
            duration_str = content_details.get('duration', '')
            duration_seconds = self._parse_duration(duration_str)
            
            return {
                "title": snippet['title'],
                "description": snippet['description'],
                "channel_title": snippet['channelTitle'],
                "published_at": snippet['publishedAt'],
                "duration": duration_seconds,
                "view_count": int(statistics.get('viewCount', 0)),
                "like_count": int(statistics.get('likeCount', 0)),
                "comment_count": int(statistics.get('commentCount', 0)),
                "thumbnail_url": snippet['thumbnails'].get('medium', {}).get('url', ''),
                "tags": snippet.get('tags', []),
                "category_id": snippet.get('categoryId', ''),
                "language": snippet.get('defaultLanguage', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error getting video metadata: {e}")
            return {"title": f"Video {video_id}", "duration": None}
    
    def _parse_duration(self, duration_str: str) -> Optional[int]:
        """Parse ISO 8601 duration to seconds"""
        if not duration_str:
            return None
        
        # Simple regex for PT1H2M3S format
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
        if not match:
            return None
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    def get_indexing_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed videos"""
        try:
            total_records = table.count_rows()
            
            # Get unique video count
            unique_videos = table.search([0.0] * EMBEDDING_DIM, vector_column_name="vector").limit(10000).to_list()
            unique_video_ids = set(record['video_id'] for record in unique_videos)
            
            # Get recent indexing activity
            recent_records = table.search([0.0] * EMBEDDING_DIM, vector_column_name="vector").limit(100).to_list()
            recent_videos = set()
            for record in recent_records:
                if 'created_at' in record:
                    created_at = record['created_at']
                    if isinstance(created_at, (int, float)):
                        # Convert timestamp to datetime
                        created_date = datetime.fromtimestamp(created_at / 1000)  # Assuming milliseconds
                        if created_date > datetime.now() - timedelta(days=7):
                            recent_videos.add(record['video_id'])
            
            return {
                "total_records": total_records,
                "unique_videos": len(unique_video_ids),
                "recent_videos_week": len(recent_videos),
                "failed_videos": len(self.failed_videos),
                "circuit_breaker_state": self.circuit_breaker.state,
                "current_rate_limit": self.rate_limiter.requests_per_minute
            }
            
        except Exception as e:
            logger.error(f"Error getting indexing statistics: {e}")
            return {"error": str(e)}
    
    def cleanup_failed_videos(self):
        """Clean up expired failed video entries"""
        current_time = time.time()
        expired_videos = [
            video_id for video_id, failure_time in self.failed_videos.items()
            if current_time - failure_time > self.failed_video_expiry
        ]
        
        for video_id in expired_videos:
            del self.failed_videos[video_id]
        
        if expired_videos:
            logger.info(f"Cleaned up {len(expired_videos)} expired failed video entries")
    
    def reset_circuit_breaker(self):
        """Manually reset the circuit breaker"""
        self.circuit_breaker.state = "CLOSED"
        self.circuit_breaker.failure_count = 0
        logger.info("Circuit breaker reset")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            "circuit_breaker_state": self.circuit_breaker.state,
            "circuit_breaker_failures": self.circuit_breaker.failure_count,
            "rate_limiter_rpm": self.rate_limiter.requests_per_minute,
            "consecutive_429s": self.rate_limiter.consecutive_429s,
            "failed_videos_count": len(self.failed_videos),
            "youtube_api_available": self.youtube is not None,
            "database_connected": True,  # If we get here, DB is connected
            "last_request_time": self.rate_limiter.last_request_time
        }


# Utility functions for better output formatting
def format_video_segments(segments: List[Dict[str, Any]], video_title: str = "", max_segments: int = 10) -> str:
    """Format video segments for display with enhanced information"""
    if not segments:
        return "❌ No relevant segments found."
    
    output = []
    if video_title:
        output.append(f"\n🎥 Video: {video_title}")
    
    display_segments = segments[:max_segments]
    output.append(f"🎯 Found {len(segments)} relevant segments (showing top {len(display_segments)}):")
    output.append("-" * 80)
    
    for i, segment in enumerate(display_segments, 1):
        # Format timestamp
        start_time = int(segment['start'])
        minutes, seconds = divmod(start_time, 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        
        # Truncate text for display
        text_preview = segment['text']
        if len(text_preview) > 150:
            text_preview = text_preview[:150] + "..."
        
        # Format score
        score = segment.get('relevance_score', segment.get('score', 0))
        
        output.append(f"\n{i}. ⏱️  {timestamp} (Score: {score:.3f})")
        output.append(f"   📝 {text_preview}")
        output.append(f"   🔗 {segment['url']}")
    
    if len(segments) > max_segments:
        output.append(f"\n... and {len(segments) - max_segments} more segments")
    
    return "\n".join(output)


def format_youtube_search_results(videos: List[Dict[str, Any]], max_videos: int = 10) -> str:
    """Format YouTube search results for display"""
    if not videos:
        return "❌ No videos found."
    
    output = []
    display_videos = videos[:max_videos]
    output.append(f"\n🔍 Found {len(videos)} videos (showing top {len(display_videos)}):")
    output.append("-" * 80)
    
    for i, video in enumerate(display_videos, 1):
        # Format description
        description = video.get('description', '')
        if len(description) > 100:
            description = description[:100] + "..."
        
        # Format date
        published_date = video.get('published_at', '')[:10]
        
        output.append(f"\n{i}. 📺 {video['title']}")
        output.append(f"   👤 Channel: {video['channel_title']}")
        output.append(f"   📅 Published: {published_date}")
        if description:
            output.append(f"   📝 {description}")
        output.append(f"   🔗 {video['url']}")
    
    if len(videos) > max_videos:
        output.append(f"\n... and {len(videos) - max_videos} more videos")
    
    return "\n".join(output)


# Enhanced example usage with comprehensive error handling
def main():
    """Example usage of the enhanced YouTube service"""
    service = YouTubeTranscriptService()
    
    # Health check
    health = service.get_health_status()
    print("🏥 Service Health Status:")
    for key, value in health.items():
        print(f"   {key}: {value}")
    
    # Example: Index a video
    try:
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with actual video
        video_id = service.extract_video_id(video_url)
        
        print(f"\n📹 Processing video: {video_id}")
        
        # Get video metadata
        metadata = service.get_video_metadata(video_id)
        print(f"📊 Video metadata:")
        print(f"   Title: {metadata['title']}")
        print(f"   Duration: {metadata.get('duration', 'Unknown')} seconds")
        print(f"   Views: {metadata.get('view_count', 'Unknown')}")
        
        # Index the video
        print(f"\n🔄 Indexing video...")
        success = service.index_video_enhanced(video_url, video_id)
        
        if success:
            print(f"✅ Successfully indexed video {video_id}")
            
            # Search within the video
            query = "main topic"  # Replace with actual search query
            results = service.search_video_content_enhanced(query, video_id, top_k=5)
            
            print(format_video_segments(results, metadata['title']))
            
        else:
            print(f"❌ Failed to index video {video_id}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example: Search YouTube videos
    try:
        print(f"\n🔍 Searching YouTube videos...")
        search_query = "python tutorial"  # Replace with actual search query
        videos = service.search_youtube_videos_enhanced(search_query, max_results=5)
        
        print(format_youtube_search_results(videos))
        
    except Exception as e:
        print(f"❌ Search error: {e}")
    
    # Show indexing statistics
    try:
        stats = service.get_indexing_statistics()
        print(f"\n📈 Indexing Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Statistics error: {e}")
    
    # Cleanup
    service.cleanup_failed_videos()
    print(f"\n🧹 Cleanup completed")


if __name__ == "__main__":
    main()