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
import yt_dlp
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
        
        # Proxy configuration - set these in your .env file
        self.proxy_servers = self._load_proxy_config()
        
    def _load_proxy_config(self) -> List[str]:
        """Load proxy configuration from environment variables"""
        proxies = []
        
        # Single proxy from env
        single_proxy = os.getenv("YOUTUBE_PROXY")
        if single_proxy:
            proxies.append(single_proxy)
        
        # Multiple proxies from env (comma-separated)
        proxy_list = os.getenv("YOUTUBE_PROXY_LIST")
        if proxy_list:
            proxies.extend([p.strip() for p in proxy_list.split(",") if p.strip()])
        
        # Default proxy servers (you can customize these)
        default_proxies = [
            # HTTP proxies
            "http://proxy1.example.com:8080",
            "http://proxy2.example.com:8080",
            # SOCKS proxies
            "socks5://proxy3.example.com:1080",
            "socks4://proxy4.example.com:1080",
        ]
        
        # Only use default proxies if no custom ones are provided
        if not proxies:
            logger.info("No custom proxies configured, using default proxy list")
            # Uncomment the line below if you want to use default proxies
            # proxies = default_proxies
        
        logger.info(f"Loaded {len(proxies)} proxy servers")
        return proxies
    
    def _get_ytdlp_options(self, proxy: Optional[str] = None) -> Dict[str, Any]:
        """Get yt-dlp options with proxy support"""
        options = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU'],
            'subtitlesformat': 'json3',
            'skip_download': True,
            'ignoreerrors': True,
            'no_warnings': True,
            'extract_flat': False,
            'quiet': True,
            'cookiefile': os.getenv("YOUTUBE_COOKIES_FILE"),  # Optional: cookies file path
        }
        
        # Add proxy if provided
        if proxy:
            options['proxy'] = proxy
            logger.info(f"Using proxy: {proxy}")
        
        # Add authentication if available
        username = os.getenv("YOUTUBE_USERNAME")
        password = os.getenv("YOUTUBE_PASSWORD")
        if username and password:
            options['username'] = username
            options['password'] = password
        
        return options

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

    def _try_with_proxies(self, func, *args, **kwargs):
        """Try a function with different proxies until one works"""
        # First try without proxy
        try:
            return func(None, *args, **kwargs)
        except Exception as e:
            logger.warning(f"Direct connection failed: {e}")
        
        # Try with each proxy
        for proxy in self.proxy_servers:
            try:
                logger.info(f"Trying with proxy: {proxy}")
                return func(proxy, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Proxy {proxy} failed: {e}")
                continue
        
        raise Exception("All proxy attempts failed")

    def _extract_info_with_proxy(self, proxy: Optional[str], url: str) -> Dict[str, Any]:
        """Extract video info using yt-dlp with proxy support"""
        options = self._get_ytdlp_options(proxy)
        
        with yt_dlp.YoutubeDL(options) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return info
            except Exception as e:
                logger.error(f"yt-dlp extraction failed: {e}")
                raise

    def _download_subtitle_content(self, url: str, video_id: str) -> Dict[str, Any]:
        """Download subtitle content from URL"""
        import urllib.request
        import urllib.error
        import json
        
        try:
            # Try direct download first
            with urllib.request.urlopen(url) as response:
                content = response.read().decode('utf-8')
                return json.loads(content)
        except Exception as e:
            logger.warning(f"Direct download failed: {e}")
            
            # Try with proxy if available
            for proxy in self.proxy_servers:
                try:
                    # Setup proxy handler
                    if proxy.startswith('http://') or proxy.startswith('https://'):
                        proxy_handler = urllib.request.ProxyHandler({'http': proxy, 'https': proxy})
                    elif proxy.startswith('socks'):
                        # For SOCKS proxies, you might need additional setup
                        logger.warning(f"SOCKS proxy {proxy} not supported for direct download")
                        continue
                    else:
                        continue
                    
                    opener = urllib.request.build_opener(proxy_handler)
                    with opener.open(url) as response:
                        content = response.read().decode('utf-8')
                        return json.loads(content)
                        
                except Exception as proxy_error:
                    logger.warning(f"Proxy {proxy} failed: {proxy_error}")
                    continue
            
            raise Exception(f"All download attempts failed for subtitle URL")

    def _parse_ytdlp_subtitles(self, info: Dict[str, Any]) -> List[TranscriptSegment]:
        """Parse subtitles from yt-dlp info"""
        segments = []
        
        # Try to get subtitles in order of preference
        subtitle_keys = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
        
        subtitles = info.get('subtitles', {})
        automatic_captions = info.get('automatic_captions', {})
        
        # Combine both subtitle sources (manual subtitles take precedence)
        all_subtitles = {**automatic_captions, **subtitles}
        
        selected_subtitles = None
        selected_lang = None
        
        # Find the best available subtitle language
        for lang in subtitle_keys:
            if lang in all_subtitles:
                selected_subtitles = all_subtitles[lang]
                selected_lang = lang
                break
        
        if not selected_subtitles:
            # Try any available language
            for lang, subs in all_subtitles.items():
                if subs:
                    selected_subtitles = subs
                    selected_lang = lang
                    break
        
        if not selected_subtitles:
            raise ValueError("No subtitles available for this video")
        
        logger.info(f"Using subtitles in language: {selected_lang}")
        
        # Find the best subtitle format
        subtitle_info = None
        format_priority = ['json3', 'vtt', 'srv3', 'srv2', 'srv1', 'ttml']
        
        for fmt in format_priority:
            for sub_info in selected_subtitles:
                if sub_info.get('ext') == fmt:
                    subtitle_info = sub_info
                    break
            if subtitle_info:
                break
        
        if not subtitle_info:
            # Fallback to first available format
            subtitle_info = selected_subtitles[0] if selected_subtitles else None
        
        if not subtitle_info:
            raise ValueError("No suitable subtitle format found")
        
        subtitle_url = subtitle_info.get('url')
        if not subtitle_url:
            raise ValueError("No subtitle URL found")
        
        logger.info(f"Downloading subtitles in format: {subtitle_info.get('ext', 'unknown')}")
        
        try:
            # Download and parse subtitle content
            subtitle_data = self._download_subtitle_content(subtitle_url, info['id'])
            
            # Parse based on format
            subtitle_format = subtitle_info.get('ext', 'unknown')
            
            if subtitle_format == 'json3' and 'events' in subtitle_data:
                # JSON3 format (YouTube's detailed format)
                for event in subtitle_data['events']:
                    if 'segs' in event:
                        start_time = float(event.get('tStartMs', 0)) / 1000.0
                        duration = float(event.get('dDurationMs', 0)) / 1000.0
                        
                        text_parts = []
                        for seg in event['segs']:
                            if 'utf8' in seg:
                                text_parts.append(seg['utf8'])
                        
                        text = ''.join(text_parts).strip()
                        if text and text != '\n':
                            # Clean up text
                            text = re.sub(r'\n+', ' ', text)
                            text = re.sub(r'\s+', ' ', text)
                            segments.append(TranscriptSegment(
                                text=text,
                                start=start_time,
                                duration=max(duration, 1.0)  # Ensure minimum duration
                            ))
            
            elif isinstance(subtitle_data, list):
                # Standard list format
                for item in subtitle_data:
                    text = item.get('text', '').strip()
                    start = float(item.get('start', 0))
                    duration = float(item.get('dur', item.get('duration', 1.0)))
                    
                    if text and text != '\n':
                        # Clean up text
                        text = re.sub(r'\n+', ' ', text)
                        text = re.sub(r'\s+', ' ', text)
                        segments.append(TranscriptSegment(
                            text=text,
                            start=start,
                            duration=max(duration, 1.0)
                        ))
            
            else:
                # Try to parse as VTT or other text-based formats
                if isinstance(subtitle_data, str):
                    # Parse VTT or SRT format
                    segments = self._parse_text_subtitles(subtitle_data)
                else:
                    raise ValueError(f"Unknown subtitle format: {subtitle_format}")
            
        except Exception as e:
            logger.error(f"Error parsing subtitle data: {e}")
            raise ValueError(f"Failed to parse subtitle data: {e}")
        
        if not segments:
            raise ValueError("No valid transcript segments found after parsing")
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.start)
        
        logger.info(f"Successfully parsed {len(segments)} subtitle segments")
        return segments
    
    def _parse_text_subtitles(self, content: str) -> List[TranscriptSegment]:
        """Parse VTT or SRT format subtitles"""
        segments = []
        lines = content.split('\n')
        
        current_start = None
        current_duration = None
        current_text = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or line.startswith('WEBVTT') or line.startswith('NOTE'):
                continue
            
            # Time code pattern (VTT: 00:00:01.000 --> 00:00:03.000)
            time_match = re.match(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})', line)
            if time_match:
                # Save previous segment
                if current_start is not None and current_text:
                    text = ' '.join(current_text).strip()
                    if text:
                        segments.append(TranscriptSegment(
                            text=text,
                            start=current_start,
                            duration=current_duration or 1.0
                        ))
                
                # Parse new timestamp
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, time_match.groups())
                start_time = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
                end_time = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
                
                current_start = start_time
                current_duration = end_time - start_time
                current_text = []
                continue
            
            # SRT time code pattern (00:00:01,000 --> 00:00:03,000)
            srt_time_match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', line)
            if srt_time_match:
                # Save previous segment
                if current_start is not None and current_text:
                    text = ' '.join(current_text).strip()
                    if text:
                        segments.append(TranscriptSegment(
                            text=text,
                            start=current_start,
                            duration=current_duration or 1.0
                        ))
                
                # Parse new timestamp
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, srt_time_match.groups())
                start_time = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
                end_time = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
                
                current_start = start_time
                current_duration = end_time - start_time
                current_text = []
                continue
            
            # Skip numeric lines (SRT sequence numbers)
            if line.isdigit():
                continue
            
            # Collect text content
            if current_start is not None:
                # Clean up text
                clean_line = re.sub(r'<[^>]+>', '', line)  # Remove HTML tags
                clean_line = re.sub(r'\{[^}]+\}', '', clean_line)  # Remove style tags
                clean_line = clean_line.strip()
                if clean_line:
                    current_text.append(clean_line)
        
        # Don't forget the last segment
        if current_start is not None and current_text:
            text = ' '.join(current_text).strip()
            if text:
                segments.append(TranscriptSegment(
                    text=text,
                    start=current_start,
                    duration=current_duration or 1.0
                ))
        
        return segments

    def get_transcript(self, youtube_url: str) -> List[TranscriptSegment]:
        """Get transcript segments from YouTube video using yt-dlp with proxy support"""
        video_id = self.extract_video_id(youtube_url)
        logger.info(f"Extracting transcript for video {video_id}...")
        
        try:
            # Try to extract info with proxy fallback
            info = self._try_with_proxies(self._extract_info_with_proxy, youtube_url)
            
            if not info:
                raise ValueError("Failed to extract video information")
            
            logger.info(f"Successfully extracted video info: {info.get('title', 'Unknown Title')}")
            
            # Parse subtitles
            segments = self._parse_ytdlp_subtitles(info)
            
            if not segments:
                raise ValueError("No transcript segments found")
            
            logger.info(f"Successfully extracted {len(segments)} transcript segments")
            return segments
            
        except Exception as e:
            error_msg = f"Failed to get transcript for video {video_id}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def get_video_info(self, youtube_url: str) -> Dict[str, Any]:
        """Get video metadata using yt-dlp"""
        try:
            info = self._try_with_proxies(self._extract_info_with_proxy, youtube_url)
            
            return {
                'video_id': info.get('id'),
                'title': info.get('title'),
                'description': info.get('description'),
                'uploader': info.get('uploader'),
                'upload_date': info.get('upload_date'),
                'duration': info.get('duration'),
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'webpage_url': info.get('webpage_url')
            }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}

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
        """Get video title from YouTube API or yt-dlp"""
        if self.youtube:
            try:
                response = self.youtube.videos().list(
                    part='snippet',
                    id=video_id
                ).execute()

                if response['items']:
                    return response['items'][0]['snippet']['title']
            except Exception as e:
                logger.warning(f"YouTube API failed, trying yt-dlp: {e}")
        
        # Fallback to yt-dlp
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            info = self._try_with_proxies(self._extract_info_with_proxy, url)
            return info.get('title', f"Video {video_id}")
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


# Example usage and testing
if __name__ == "__main__":
    # Initialize the service
    service = YouTubeWorkflowService()
    
    # Example usage
    try:
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