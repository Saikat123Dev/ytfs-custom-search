import google.generativeai as genai
import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
import time
import os
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import lancedb
import pyarrow as pa
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from dotenv import load_dotenv
from collections import defaultdict, Counter
import threading
import pickle
import json
from queue import Queue
import random

# Load environment variables
load_dotenv()

@dataclass
class TranscriptSegment:
    """Individual transcript segment - no merging"""
    text: str
    start: float
    duration: float
    keywords: Optional[List[str]] = None
    
    def clean_text(self) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', self.text)  # Remove annotations
        text = re.sub(r'[^\w\s\-.,;!?]', '', text)  # Keep only basic punctuation
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text
    
    def text_hash(self) -> str:
        """Generate hash for deduplication"""
        return hashlib.md5(f"{self.clean_text()}_{self.start}".encode()).hexdigest()[:12]

class SmartTextProcessor:
    """Advanced text processing with keyword optimization"""
    
    STOPWORDS = {
        'the', 'and', 'but', 'are', 'for', 'not', 'you', 'your', 'this', 'that', 
        'with', 'have', 'has', 'was', 'were', 'they', 'their', 'there', 'what', 
        'which', 'who', 'will', 'would', 'should', 'could', 'can', 'may', 'might',
        'must', 'shall', 'here', 'from', 'into', 'than', 'then', 'when', 'where',
        'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'only', 'own', 'same', 'very', 'just', 'now', 'even', 'also'
    }
    
    IMPORTANT_KEYWORDS = {
        'technology', 'ai', 'machine', 'learning', 'data', 'algorithm', 'model',
        'system', 'software', 'hardware', 'computer', 'programming', 'code',
        'business', 'market', 'strategy', 'growth', 'revenue', 'profit', 'customer',
        'science', 'research', 'study', 'analysis', 'method', 'result', 'conclusion'
    }
    
    @staticmethod
    def extract_smart_keywords(text: str, max_words: int = 8) -> List[str]:
        """Extract meaningful keywords with TF-IDF-like scoring"""
        text_lower = text.lower()
        words = re.findall(r'\b[a-z]{3,}\b', text_lower)
        
        if len(words) < 3:  # Skip very short segments
            return []
        
        total_words = len(words)
        
        # Count word frequencies with importance weighting
        word_freq = Counter()
        for word in words:
            if word not in SmartTextProcessor.STOPWORDS and len(word) > 2:
                weight = 3 if word in SmartTextProcessor.IMPORTANT_KEYWORDS else 1
                word_freq[word] += weight
        
        # Extract meaningful bigrams
        bigrams = []
        for i in range(len(words) - 1):
            if (words[i] not in SmartTextProcessor.STOPWORDS and 
                words[i+1] not in SmartTextProcessor.STOPWORDS and
                len(words[i]) > 2 and len(words[i+1]) > 2):
                bigram = f"{words[i]} {words[i+1]}"
                bigrams.append(bigram)
        
        # Add bigram frequencies with higher weight
        bigram_freq = Counter(bigrams)
        for bigram, count in bigram_freq.most_common(3):
            word_freq[bigram] += count * 2
        
        return [word for word, _ in word_freq.most_common(max_words)]

class MultithreadedEmbeddingService:
    """High-performance embedding service optimized for individual segments"""
    
    def __init__(self, max_workers: int = 8):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        genai.configure(api_key=api_key)
        self.model_name = "text-embedding-004"
        self.dimensions = 768
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.request_times = Queue()
        self.requests_per_minute = 55  # Slightly more aggressive
        self.rate_lock = threading.RLock()
        self.cache_file = "embedding_cache_v2.pkl"
        self.embedding_cache = self._load_cache()
        self.cache_lock = threading.RLock()
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'api_calls': 0,
            'avg_batch_time': 0.0,
            'successful_embeddings': 0,
            'failed_embeddings': 0
        }
        self.stats_lock = threading.Lock()
        print(f"🚀 Initialized embedding service with {max_workers} workers")
    
    def _load_cache(self) -> Dict:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    print(f"📦 Loaded {len(cache)} cached embeddings")
                    return cache
        except Exception as e:
            print(f"⚠️ Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        with self.cache_lock:
            try:
                # Create backup before saving
                if os.path.exists(self.cache_file):
                    backup_file = f"{self.cache_file}.backup"
                    os.rename(self.cache_file, backup_file)
                
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.embedding_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                # Remove backup if save successful
                if os.path.exists(f"{self.cache_file}.backup"):
                    os.remove(f"{self.cache_file}.backup")
                    
                print(f"💾 Saved {len(self.embedding_cache)} embeddings to cache")
            except Exception as e:
                print(f"⚠️ Could not save cache: {e}")
                # Restore backup if save failed
                backup_file = f"{self.cache_file}.backup"
                if os.path.exists(backup_file):
                    os.rename(backup_file, self.cache_file)
    
    def _get_cache_key(self, text: str, is_query: bool) -> str:
        """Generate cache key for text and query type"""
        content = f"{text[:500]}_{is_query}"  # Limit key length
        return hashlib.md5(content.encode()).hexdigest()
    
    def _wait_for_rate_limit(self):
        """Smart rate limiting with jitter"""
        with self.rate_lock:
            current_time = time.time()
            
            # Clean old requests from queue
            while not self.request_times.empty():
                try:
                    old_time = self.request_times.get_nowait()
                    if current_time - old_time < 60:  # Within last minute
                        self.request_times.put(old_time)
                        break
                except:
                    break
            
            # Check if we need to wait
            if self.request_times.qsize() >= self.requests_per_minute:
                wait_time = 60.0 / self.requests_per_minute
                jitter = random.uniform(0.1, 0.3)  # Add randomness
                time.sleep(wait_time + jitter)
            
            self.request_times.put(current_time)
    
    def _get_single_embedding(self, text: str, is_query: bool, max_retries: int = 3) -> List[float]:
        """Get embedding for single text with retry logic"""
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                task_type = "retrieval_query" if is_query else "retrieval_document"
                
                # Truncate text to avoid API limits
                truncated_text = text[:8000] if len(text) > 8000 else text
                
                result = genai.embed_content(
                    model=f"models/{self.model_name}",
                    content=truncated_text,
                    task_type=task_type
                )
                
                # Normalize embedding vector
                vector = np.array(result['embedding'], dtype=np.float32)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                
                with self.stats_lock:
                    self.stats['api_calls'] += 1
                    self.stats['successful_embeddings'] += 1
                
                return vector.tolist()
                
            except Exception as e:
                print(f"❌ Embedding error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    with self.stats_lock:
                        self.stats['failed_embeddings'] += 1
        
        # Return zero vector on failure
        return [0.0] * self.dimensions
    
    def _process_embedding_batch(self, batch_data: List[Tuple[int, str, bool]]) -> List[Tuple[int, List[float]]]:
        """Process a batch of embeddings efficiently"""
        results = []
        
        for idx, text, is_query in batch_data:
            cache_key = self._get_cache_key(text, is_query)
            
            # Check cache first
            with self.cache_lock:
                if cache_key in self.embedding_cache:
                    embedding = self.embedding_cache[cache_key]
                    with self.stats_lock:
                        self.stats['cache_hits'] += 1
                else:
                    # Get new embedding
                    embedding = self._get_single_embedding(text, is_query)
                    self.embedding_cache[cache_key] = embedding
            
            results.append((idx, embedding))
            
            with self.stats_lock:
                self.stats['total_requests'] += 1
        
        return results
    
    def get_embeddings_batch(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Get embeddings for multiple texts using multithreading"""
        if not texts:
            return []
        
        start_time = time.time()
        print(f"🧠 Processing {len(texts)} embeddings with {self.max_workers} threads...")
        
        # Prepare indexed data
        indexed_texts = [(i, text, is_query) for i, text in enumerate(texts)]
        
        # Calculate optimal batch size
        batch_size = max(1, len(texts) // (self.max_workers * 3))
        batch_size = min(batch_size, 5)  # Keep batches small for better parallelism
        
        batches = [indexed_texts[i:i + batch_size] for i in range(0, len(indexed_texts), batch_size)]
        print(f"📦 Created {len(batches)} batches (avg size: {len(texts)/len(batches):.1f})")
        
        # Initialize results array
        embeddings = [None] * len(texts)
        
        # Submit all batches to thread pool
        future_to_batch = {
            self.executor.submit(self._process_embedding_batch, batch): batch 
            for batch in batches
        }
        
        # Collect results as they complete
        completed_batches = 0
        for future in as_completed(future_to_batch):
            try:
                batch_results = future.result(timeout=60)  # 60 second timeout
                for idx, embedding in batch_results:
                    embeddings[idx] = embedding
                    
                completed_batches += 1
                if completed_batches % 3 == 0 or completed_batches == len(batches):
                    progress = (completed_batches / len(batches)) * 100
                    print(f"⏳ Progress: {completed_batches}/{len(batches)} batches ({progress:.1f}%)")
                    
            except Exception as e:
                print(f"❌ Batch processing error: {e}")
                # Fill failed batch with zero vectors
                batch = future_to_batch[future]
                for idx, _, _ in batch:
                    if embeddings[idx] is None:
                        embeddings[idx] = [0.0] * self.dimensions
        
        # Ensure no None values remain
        for i in range(len(embeddings)):
            if embeddings[i] is None:
                embeddings[i] = [0.0] * self.dimensions
        
        # Save cache periodically
        if len(texts) > 5:
            self._save_cache()
        
        # Update statistics
        total_time = time.time() - start_time
        with self.stats_lock:
            self.stats['avg_batch_time'] = (
                self.stats['avg_batch_time'] * 0.8 + total_time * 0.2
            )
        
        cache_hit_rate = (self.stats['cache_hits'] / max(self.stats['total_requests'], 1)) * 100
        print(f"✅ Completed in {total_time:.2f}s | Cache hit rate: {cache_hit_rate:.1f}%")
        
        return embeddings
    
    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics"""
        with self.stats_lock:
            total_requests = max(self.stats['total_requests'], 1)
            cache_hit_rate = (self.stats['cache_hits'] / total_requests) * 100
            success_rate = (self.stats['successful_embeddings'] / 
                           max(self.stats['successful_embeddings'] + self.stats['failed_embeddings'], 1)) * 100
            
            return {
                **self.stats,
                'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                'success_rate': f"{success_rate:.1f}%",
                'cache_size': len(self.embedding_cache),
                'avg_batch_time_seconds': round(self.stats['avg_batch_time'], 2)
            }
    
    def shutdown(self):
        """Gracefully shutdown the service"""
        print("🔄 Shutting down embedding service...")
        self.executor.shutdown(wait=True)
        self._save_cache()
        print("✅ Embedding service shutdown complete")

class OptimizedTranscriptDatabase:
    """Database optimized for individual transcript segments"""
    
    def __init__(self, max_workers: int = 8):
        self.db = lancedb.connect("transcript_segments_db")
        self.table_name = "individual_segments_v1"
        self.embedding_service = MultithreadedEmbeddingService(max_workers=max_workers)
        self._initialize_table()
        self.keyword_index = defaultdict(set)
        self.video_cache = set()
        self._build_indexes()
    
    def _initialize_table(self):
        """Initialize database table with optimized schema"""
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("video_id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("original_text", pa.string()),  # Keep original for display
            pa.field("keywords", pa.list_(pa.string())),
            pa.field("embedding", pa.list_(pa.float32(), self.embedding_service.dimensions)),
            pa.field("start_time", pa.float32()),
            pa.field("duration", pa.float32()),
            pa.field("text_length", pa.int32()),
            pa.field("keyword_count", pa.int32()),
            pa.field("created_at", pa.int64())
        ])
        
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
            print(f"📂 Opened existing table: {self.table_name}")
        else:
            self.table = self.db.create_table(self.table_name, schema=schema)
            print(f"🆕 Created new table: {self.table_name}")
    
    def _build_indexes(self):
        """Build in-memory indexes for fast keyword lookup"""
        print("🔍 Building keyword and video indexes...")
        try:
            total_rows = self.table.count_rows()
            if total_rows == 0:
                print("📊 No existing data to index")
                return
                
            print(f"📊 Processing {total_rows} existing segments...")
            
            # Process in chunks to avoid memory issues
            chunk_size = 2000
            processed = 0
            
            while processed < total_rows:
                try:
                    chunk = self.table.head(chunk_size, offset=processed).to_pylist()
                    if not chunk:
                        break
                    
                    for segment in chunk:
                        # Build keyword index
                        for keyword in segment.get('keywords', []):
                            if keyword:  # Skip empty keywords
                                self.keyword_index[keyword.lower()].add(segment['id'])
                        
                        # Build video cache
                        self.video_cache.add(segment['video_id'])
                    
                    processed += len(chunk)
                    if processed % 5000 == 0:
                        print(f"📈 Indexed {processed}/{total_rows} segments...")
                        
                except Exception as e:
                    print(f"⚠️ Error processing chunk at offset {processed}: {e}")
                    break
            
            print(f"✅ Built indexes: {len(self.keyword_index)} unique keywords, {len(self.video_cache)} videos")
            
        except Exception as e:
            print(f"⚠️ Could not build indexes: {e}")
    
    def video_exists(self, video_id: str) -> bool:
        """Check if video already exists in database"""
        if video_id in self.video_cache:
            return True
        
        try:
            # Double-check with database query
            result = self.table.search([0.0] * self.embedding_service.dimensions) \
                              .where(f"video_id = '{video_id}'") \
                              .limit(1) \
                              .to_list()
            
            exists = len(result) > 0
            if exists:
                self.video_cache.add(video_id)
            return exists
            
        except Exception as e:
            print(f"⚠️ Error checking video existence: {e}")
            return False
    
    def add_segments(self, segments: List[TranscriptSegment], video_id: str) -> int:
        """Add individual transcript segments to database"""
        if not segments:
            print(f"⚡ No segments provided for video {video_id}")
            return 0
            
        if self.video_exists(video_id):
            print(f"⚡ Video {video_id} already exists - skipping")
            return 0
        
        print(f"🚀 Processing {len(segments)} individual segments for video {video_id}...")
        start_time = time.time()
        
        # Filter and prepare segments
        valid_segments = []
        embedding_texts = []
        
        for seg in segments:
            clean_text = seg.clean_text()
            
            # Skip very short or meaningless segments
            if len(clean_text) < 10 or len(clean_text.split()) < 2:
                continue
            
            # Extract keywords for this segment
            seg.keywords = SmartTextProcessor.extract_smart_keywords(clean_text)
            
            valid_segments.append(seg)
            
            # Prepare text for embedding (combine keywords + text for better context)
            keyword_text = " ".join(seg.keywords[:5]) if seg.keywords else ""
            combined_text = f"{keyword_text} {clean_text}" if keyword_text else clean_text
            embedding_texts.append(combined_text)
        
        if not valid_segments:
            print(f"⚠️ No valid segments found for video {video_id}")
            return 0
        
        print(f"📝 Filtered to {len(valid_segments)} valid segments")
        
        # Get embeddings for all segments
        embeddings = self.embedding_service.get_embeddings_batch(embedding_texts, is_query=False)
        
        # Prepare data for database insertion
        data = []
        for seg, embedding in zip(valid_segments, embeddings):
            clean_text = seg.clean_text()
            segment_id = seg.text_hash()
            keywords = seg.keywords or []
            
            data.append({
                "id": segment_id,
                "video_id": video_id,
                "text": clean_text,
                "original_text": seg.text,  # Keep original for display
                "keywords": keywords,
                "embedding": embedding,
                "start_time": float(seg.start),
                "duration": float(seg.duration),
                "text_length": len(clean_text),
                "keyword_count": len(keywords),
                "created_at": int(datetime.now().timestamp())
            })
            
            # Update keyword index
            for keyword in keywords:
                self.keyword_index[keyword.lower()].add(segment_id)
        
        # Insert into database
        try:
            self.table.add(data)
            self.video_cache.add(video_id)
            
            total_time = time.time() - start_time
            print(f"✅ Added {len(data)} segments in {total_time:.2f}s")
            return len(data)
            
        except Exception as e:
            print(f"❌ Database insert error: {e}")
            return 0
    
    def search_hybrid(self, query: str, video_id: str, top_k: int = 10) -> List[Dict]:
        """Advanced hybrid search combining semantic and keyword matching"""
        try:
            print(f"🔍 Searching for: '{query}' in video: {video_id}")
            
            # Extract query keywords
            query_keywords = SmartTextProcessor.extract_smart_keywords(query, 6)
            print(f"🏷️ Query keywords: {query_keywords}")
            
            # Prepare enhanced query for embedding
            keyword_text = " ".join(query_keywords[:4])
            enhanced_query = f"{keyword_text} {query}" if query_keywords else query
            query_embedding = self.embedding_service.get_embeddings_batch([enhanced_query], is_query=True)[0]
            
            # Find relevant segments by keywords
            keyword_segment_ids = set()
            for kw in query_keywords:
                matching_ids = self.keyword_index.get(kw.lower(), set())
                keyword_segment_ids.update(matching_ids)
                print(f"🔑 Keyword '{kw}' matched {len(matching_ids)} segments")
            
            # Perform semantic search
            search_query = self.table.search(query_embedding, vector_column_name="embedding") \
                                   .where(f"video_id = '{video_id}'") \
                                   .limit(top_k * 3)  # Get more candidates for reranking
            
            results = search_query.to_list()
            print(f"🎯 Found {len(results)} semantic candidates")
            
            if not results:
                print("❌ No results found")
                return []
            
            # Enhanced scoring and ranking
            scored_results = []
            query_keywords_set = set(kw.lower() for kw in query_keywords)
            
            for result in results:
                # Base semantic similarity score
                base_score = 1.0 - result.get('_distance', 1.0)
                base_score = max(0.0, min(1.0, base_score))  # Clamp to [0,1]
                
                # Keyword overlap bonus
                result_keywords = set(kw.lower() for kw in result.get('keywords', []))
                keyword_overlap = len(result_keywords.intersection(query_keywords_set))
                keyword_bonus = min(keyword_overlap / max(len(query_keywords_set), 1), 1.0) * 0.4
                
                # Text matching bonus
                text_lower = result['text'].lower()
                text_matches = sum(1 for kw in query_keywords if kw.lower() in text_lower)
                text_bonus = min(text_matches / max(len(query_keywords), 1), 1.0) * 0.3
                
                # Segment quality bonus (longer, more keywords = better)
                quality_bonus = min(result.get('keyword_count', 0) / 8.0, 0.2)
                
                # Combine scores
                final_score = (base_score * 0.5) + keyword_bonus + text_bonus + quality_bonus
                
                result['combined_score'] = final_score
                result['keyword_overlap'] = keyword_overlap
                result['text_matches'] = text_matches
                result['base_semantic_score'] = base_score
                
                scored_results.append(result)
            
            # Sort by combined score and return top results
            scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
            return scored_results[:top_k]
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            total_rows = self.table.count_rows()
            return {
                'total_segments': total_rows,
                'unique_videos': len(self.video_cache),
                'keyword_index_size': len(self.keyword_index),
                'avg_keywords_per_segment': sum(len(ids) for ids in self.keyword_index.values()) / max(len(self.keyword_index), 1),
                'embedding_stats': self.embedding_service.get_performance_stats()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def shutdown(self):
        """Shutdown database service"""
        self.embedding_service.shutdown()

class YouTubeService:
    """YouTube transcript service - no merging, pure individual segments"""
    
    def __init__(self):
        self.cache = {}
        self.cache_file = "youtube_segments_cache.json"
        self._load_cache()
    
    def _load_cache(self):
        """Load cached transcript segments"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    print(f"📺 Loaded {len(cache_data)} cached video transcripts")
                    
                    for video_id, segments_data in cache_data.items():
                        segments = [
                            TranscriptSegment(
                                text=seg_data['text'],
                                start=seg_data['start'],
                                duration=seg_data['duration'],
                                keywords=seg_data.get('keywords')
                            ) for seg_data in segments_data
                        ]
                        self.cache[video_id] = segments
        except Exception as e:
            print(f"⚠️ Could not load YouTube cache: {e}")
    
    def _save_cache(self):
        """Save transcript cache"""
        try:
            cache_data = {}
            for video_id, segments in self.cache.items():
                cache_data[video_id] = [
                    {
                        'text': seg.text,
                        'start': seg.start,
                        'duration': seg.duration,
                        'keywords': seg.keywords
                    } for seg in segments
                ]
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Saved {len(cache_data)} video transcripts to cache")
        except Exception as e:
            print(f"⚠️ Could not save YouTube cache: {e}")
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r"(?:v=|\/)([0-9A-Za-z_-]{11})",
            r"youtu\.be\/([0-9A-Za-z_-]{11})",
            r"youtube\.com\/embed\/([0-9A-Za-z_-]{11})",
            r"youtube\.com\/watch\?v=([0-9A-Za-z_-]{11})"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # Check if it's already a video ID
        if len(url) == 11 and re.match(r'^[0-9A-Za-z_-]+$', url):
            return url
            
        raise ValueError(f"Could not extract video ID from: {url}")
    
    def get_transcript(self, video_id: str) -> List[TranscriptSegment]:
        """Get individual transcript segments (no merging)"""
        if video_id in self.cache:
            print(f"📺 Using cached transcript for {video_id}")
            return self.cache[video_id]
        
        try:
            print(f"🔄 Fetching fresh transcript for video: {video_id}")
            
            # Try to get transcript in preferred languages
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
            )
            
            if not transcript_list:
            # Convert to individual segments
              segments = []
            for entry in transcript_list:
                segment = TranscriptSegment(
                    text=entry['text'],
                    start=entry['start'],
                    duration=entry.get('duration', 0.0)
                )
                segments.append(segment)
            
            # Cache the segments
            self.cache[video_id] = segments
            self._save_cache()
            
            print(f"✅ Retrieved {len(segments)} individual segments")
            return segments
            
        except TranscriptsDisabled:
            print(f"❌ Transcripts disabled for video: {video_id}")
            return []
        except NoTranscriptFound:
            print(f"❌ No transcript found for video: {video_id}")
            return []
        except VideoUnavailable:
            print(f"❌ Video unavailable: {video_id}")
            return []
        except Exception as e:
            print(f"❌ Error fetching transcript for {video_id}: {e}")
            return []

class TranscriptProcessor:
    """Main processing orchestrator"""
    
    def __init__(self, max_workers: int = 8):
        self.youtube_service = YouTubeService()
        self.database = OptimizedTranscriptDatabase(max_workers=max_workers)
        self.processing_stats = {
            'videos_processed': 0,
            'segments_processed': 0,
            'start_time': time.time(),
            'errors': []
        }
    
    def process_video(self, video_url: str) -> Dict:
        """Process a single YouTube video"""
        try:
            # Extract video ID
            video_id = self.youtube_service.extract_video_id(video_url)
            print(f"\n🎥 Processing video: {video_id}")
            
            # Check if already processed
            if self.database.video_exists(video_id):
                print(f"⏭️ Video {video_id} already processed")
                return {
                    'video_id': video_id,
                    'status': 'already_exists',
                    'segments_added': 0
                }
            
            # Get transcript segments
            segments = self.youtube_service.get_transcript(video_id)
            if not segments:
                error_msg = f"No transcript available for video {video_id}"
                self.processing_stats['errors'].append(error_msg)
                return {
                    'video_id': video_id,
                    'status': 'no_transcript',
                    'error': error_msg,
                    'segments_added': 0
                }
            
            # Add segments to database
            segments_added = self.database.add_segments(segments, video_id)
            
            # Update stats
            self.processing_stats['videos_processed'] += 1
            self.processing_stats['segments_processed'] += segments_added
            
            return {
                'video_id': video_id,
                'status': 'success',
                'segments_added': segments_added,
                'total_segments': len(segments)
            }
            
        except Exception as e:
            error_msg = f"Error processing video {video_url}: {str(e)}"
            print(f"❌ {error_msg}")
            self.processing_stats['errors'].append(error_msg)
            return {
                'video_id': getattr(self, '_last_video_id', 'unknown'),
                'status': 'error',
                'error': error_msg,
                'segments_added': 0
            }
    
    def process_videos(self, video_urls: List[str]) -> Dict:
        """Process multiple YouTube videos"""
        print(f"🚀 Starting batch processing of {len(video_urls)} videos")
        start_time = time.time()
        
        results = []
        successful = 0
        failed = 0
        
        for i, url in enumerate(video_urls, 1):
            print(f"\n📊 Progress: {i}/{len(video_urls)} videos")
            result = self.process_video(url)
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
            elif result['status'] == 'error':
                failed += 1
            
            # Show progress every 5 videos
            if i % 5 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                eta = (len(video_urls) - i) * avg_time
                print(f"⏱️ Processed {i}/{len(video_urls)} | ETA: {eta/60:.1f} minutes")
        
        total_time = time.time() - start_time
        
        # Final summary
        summary = {
            'total_videos': len(video_urls),
            'successful': successful,
            'failed': failed,
            'already_existed': sum(1 for r in results if r['status'] == 'already_exists'),
            'no_transcript': sum(1 for r in results if r['status'] == 'no_transcript'),
            'total_segments_added': sum(r['segments_added'] for r in results),
            'processing_time_seconds': round(total_time, 2),
            'avg_time_per_video': round(total_time / len(video_urls), 2),
            'results': results,
            'errors': self.processing_stats['errors']
        }
        
        print(f"\n🎉 Batch processing complete!")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️ Already existed: {summary['already_existed']}")
        print(f"📝 Total segments added: {summary['total_segments_added']}")
        print(f"⏱️ Total time: {total_time/60:.1f} minutes")
        
        return summary
    
    def search_transcripts(self, query: str, video_id: str, top_k: int = 5) -> List[Dict]:
        """Search transcript segments"""
        return self.database.search_hybrid(query, video_id, top_k)
    
    def get_stats(self) -> Dict:
        """Get comprehensive processing statistics"""
        db_stats = self.database.get_stats()
        
        return {
            'processing_stats': self.processing_stats,
            'database_stats': db_stats,
            'runtime_minutes': (time.time() - self.processing_stats['start_time']) / 60
        }
    
    def shutdown(self):
        """Graceful shutdown"""
        print("\n🔄 Shutting down processor...")
        self.database.shutdown()
        print("✅ Processor shutdown complete")

# Example usage and testing functions
def test_single_video():
    """Test with a single video"""
    processor = TranscriptProcessor(max_workers=6)
    
    try:
        # Example video URL
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with actual video
        
        result = processor.process_video(video_url)
        print(f"\n🎯 Processing result: {result}")
        
        # Test search if video was processed
        if result['status'] == 'success':
            video_id = result['video_id']
            search_results = processor.search_transcripts("music", video_id, top_k=3)
            
            print(f"\n🔍 Search results for 'music':")
            for i, segment in enumerate(search_results, 1):
                print(f"{i}. Score: {segment['combined_score']:.3f}")
                print(f"   Text: {segment['text'][:100]}...")
                print(f"   Time: {segment['start_time']:.1f}s")
                print(f"   Keywords: {segment['keywords']}")
                print()
        
        # Show stats
        stats = processor.get_stats()
        print(f"\n📊 Final stats: {stats}")
        
    finally:
        processor.shutdown()

def test_batch_processing():
    """Test with multiple videos"""
    processor = TranscriptProcessor(max_workers=8)
    
    try:
        # Example video URLs - replace with actual URLs
        video_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=9bZkp7q19f0",
            "https://www.youtube.com/watch?v=kJQP7kiw5Fk"
        ]
        
        results = processor.process_videos(video_urls)
        
        print(f"\n📈 Batch processing results:")
        print(f"Total videos: {results['total_videos']}")
        print(f"Successful: {results['successful']}")
        print(f"Total segments: {results['total_segments_added']}")
        
        # Show detailed results
        for result in results['results']:
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_emoji} {result['video_id']}: {result['status']} ({result['segments_added']} segments)")
        
    finally:
        processor.shutdown()

def interactive_search():
    """Interactive search interface"""
    processor = TranscriptProcessor(max_workers=4)
    
    try:
        print("🔍 Interactive Transcript Search")
        print("Commands: 'quit' to exit, 'stats' for statistics")
        
        while True:
            print("\n" + "="*50)
            video_input = input("Enter YouTube URL or video ID: ").strip()
            
            if video_input.lower() == 'quit':
                break
            elif video_input.lower() == 'stats':
                stats = processor.get_stats()
                print(f"\n📊 Statistics: {stats}")
                continue
            
            try:
                video_id = processor.youtube_service.extract_video_id(video_input)
                
                # Process video if not already in database
                if not processor.database.video_exists(video_id):
                    print(f"🔄 Processing new video: {video_id}")
                    result = processor.process_video(video_input)
                    if result['status'] != 'success':
                        print(f"❌ Failed to process: {result.get('error', 'Unknown error')}")
                        continue
                
                # Search loop for this video
                while True:
                    query = input(f"\nSearch query for {video_id} (or 'back' to select new video): ").strip()
                    
                    if query.lower() == 'back':
                        break
                    
                    if not query:
                        continue
                    
                    results = processor.search_transcripts(query, video_id, top_k=5)
                    
                    if not results:
                        print("❌ No results found")
                        continue
                    
                    print(f"\n🎯 Found {len(results)} results:")
                    for i, segment in enumerate(results, 1):
                        print(f"\n{i}. Score: {segment['combined_score']:.3f} | Time: {segment['start_time']:.1f}s")
                        print(f"   Text: {segment['original_text']}")
                        if segment['keywords']:
                            print(f"   Keywords: {', '.join(segment['keywords'][:5])}")
                        print(f"   Matches: {segment['keyword_overlap']} keywords, {segment['text_matches']} text")
            
            except Exception as e:
                print(f"❌ Error: {e}")
    
    finally:
        processor.shutdown()

if __name__ == "__main__":
    print("🎬 YouTube Transcript Processor")
    print("1. Test single video")
    print("2. Test batch processing")
    print("3. Interactive search")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        test_single_video()
    elif choice == "2":
        test_batch_processing()
    elif choice == "3":
        interactive_search()
    else:
        print("Invalid choice. Running interactive search...")
        interactive_search()