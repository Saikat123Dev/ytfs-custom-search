import os
import re
import uuid
import logging
import numpy as np
import pyarrow as pa
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
        # Try to drop table if it exists (but ignore if it doesn't)
        db.drop_table("transcripts", ignore_missing=True)
    except Exception as e:
        logger.debug(f"Error dropping table (this is expected if table doesn't exist): {e}")

    # Create new table
    table = db.create_table("transcripts", schema=schema)
    logger.info("Successfully created 'transcripts' table.")


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
        
        # Configure proxy settings for transcript API
        self.proxy_config = None
        if os.getenv("USE_PROXY", "false").lower() == "true":
            proxy_url = os.getenv("PROXY_URL", "http://81af49f477b044f6872b853f907fe061:@api.zyte.com:8011")
            self.proxy_config = {
                "http": proxy_url,
                "https": proxy_url,
            }
            logger.info("Proxy configuration enabled")

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
   
    def get_available_transcripts(self, video_id: str) -> Dict[str, Any]:
        """Get information about available transcripts for debugging"""
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(
                video_id, 
                proxies=self.proxy_config
            )
            available = {}

            for transcript in transcript_list:
                available[transcript.language] = {
                    'language': transcript.language,
                    'language_code': transcript.language_code,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                }

            return available
        except Exception as e:
            logger.error(f"Error getting transcript list: {e}")
            return {}

    def get_transcript(self, youtube_url: str) -> List[TranscriptSegment]:
        """Get transcript segments from YouTube video with improved error handling"""
        video_id = self.extract_video_id(youtube_url)

        # Language preference order (add more as needed)
        language_preferences = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']

        try:
            # First, try to get transcript list to see what's available
            logger.info(f"Checking available transcripts for video {video_id}...")
            
            available_transcripts = self.get_available_transcripts(video_id)

            if available_transcripts:
                logger.info(f"Available transcripts: {list(available_transcripts.keys())}")
            else:
                logger.warning(f"No transcript information available for video {video_id}")

            # Try different approaches to get transcript
            raw_transcript = None
            transcript_source = None

            # Method 1: Try preferred languages
            for lang in language_preferences:
                try:
                    raw_transcript = YouTubeTranscriptApi.get_transcript(
                        video_id, 
                        languages=[lang], 
                        proxies=self.proxy_config
                    )
                    transcript_source = f"Language: {lang}"
                    logger.info(f"Successfully retrieved transcript in {lang}")
                    break
                except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
                    continue
                except Exception as e:
                    logger.warning(f"Error with language {lang}: {e}")
                    continue

            # Method 2: Try without language specification (auto-detect)
            if raw_transcript is None:
                try:
                    raw_transcript = YouTubeTranscriptApi.get_transcript(
                        video_id, 
                        proxies=self.proxy_config
                    )
                    transcript_source = "Auto-detected language"
                    logger.info("Successfully retrieved transcript with auto-detection")
                except Exception as e:
                    logger.warning(f"Auto-detection failed: {e}")

            # Method 3: Try to get any available transcript
            if raw_transcript is None:
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(
                        video_id, 
                        proxies=self.proxy_config
                    )
                    # Get the first available transcript
                    for transcript in transcript_list:
                        try:
                            raw_transcript = transcript.fetch(proxies=self.proxy_config)
                            transcript_source = f"First available: {transcript.language_code}"
                            logger.info(f"Retrieved transcript in {transcript.language_code}")
                            break
                        except Exception as e:
                            logger.warning(f"Failed to fetch {transcript.language_code}: {e}")
                            continue
                except Exception as e:
                    logger.error(f"Failed to list transcripts: {e}")

            # Method 4: Try to get generated transcript if manual not available
            if raw_transcript is None:
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(
                        video_id, 
                        proxies=self.proxy_config
                    )
                    for transcript in transcript_list:
                        if transcript.is_generated:
                            try:
                                raw_transcript = transcript.fetch(proxies=self.proxy_config)
                                transcript_source = f"Generated: {transcript.language_code}"
                                logger.info(f"Retrieved generated transcript in {transcript.language_code}")
                                break
                            except Exception as e:
                                logger.warning(f"Failed to fetch generated transcript {transcript.language_code}: {e}")
                                continue
                except Exception as e:
                    logger.error(f"Failed to get generated transcripts: {e}")

            # If we still don't have a transcript, try translated versions
            if raw_transcript is None:
                try:
                    transcript_list = YouTubeTranscriptApi.list_transcripts(
                        video_id, 
                        proxies=self.proxy_config
                    )
                    for transcript in transcript_list:
                        if transcript.is_translatable:
                            try:
                                # Try to translate to English
                                translated = transcript.translate('en')
                                raw_transcript = translated.fetch(proxies=self.proxy_config)
                                transcript_source = f"Translated to English from {transcript.language_code}"
                                logger.info(f"Retrieved translated transcript from {transcript.language_code}")
                                break
                            except Exception as e:
                                logger.warning(f"Failed to translate from {transcript.language_code}: {e}")
                                continue
                except Exception as e:
                    logger.error(f"Failed to get translatable transcripts: {e}")

            if raw_transcript is None:
                # Provide detailed error information
                error_msg = f"Could not retrieve transcript for video {video_id}. "
                if available_transcripts:
                    error_msg += f"Available languages: {list(available_transcripts.keys())}. "
                else:
                    error_msg += "No transcripts appear to be available. "
                error_msg += "This could be due to: 1) Transcripts disabled by uploader, 2) Video is too new, 3) Video is private/restricted, 4) Regional restrictions."
                raise ValueError(error_msg)

            logger.info(f"Transcript retrieved via: {transcript_source}")
            logger.info(f"Transcript contains {len(raw_transcript)} segments")

            # Convert to TranscriptSegment objects
            segments = []
            for segment in raw_transcript:
                # Handle potential missing keys
                text = segment.get('text', '').strip()
                start = float(segment.get('start', 0))
                duration = float(segment.get('duration', 0))

                if text:  # Only add segments with actual text
                    segments.append(TranscriptSegment(text=text, start=start, duration=duration))

            if not segments:
                raise ValueError(f"Transcript retrieved but contains no valid text segments for video {video_id}")

            logger.info(f"Successfully processed {len(segments)} transcript segments")
            return segments

        except (VideoUnavailable, TranscriptsDisabled, NoTranscriptFound) as e:
            error_msg = f"Transcript unavailable for video {video_id}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error getting transcript for video {video_id}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

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
            logger.info(f"Getting transcript for video {video_id}...")
            
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
        if not self.youtube:
            return f"Video {video_id}"

        try:
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