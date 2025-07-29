"""
RAG and Semantic Search Integration

Enhances the routing system with semantic search capabilities and
intelligent context retrieval for better model responses.
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


@dataclass
class SemanticSearchResult:
    """Result from semantic search."""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    chunk_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "source": self.source,
            "chunk_id": self.chunk_id
        }


@dataclass
class DocumentChunk:
    """A chunk of a document for semantic search."""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    chunk_id: Optional[str] = None
    
    def __post_init__(self):
        if self.chunk_id is None:
            # Generate a simple hash-based ID
            import hashlib
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:12]


class SemanticSearchEngine:
    """
    Advanced semantic search engine with hybrid retrieval capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Embedding model
        self.embedding_model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.embedding_model = None
        self.embedding_dim = None
        
        # Vector storage
        self.faiss_index = None
        self.document_chunks: List[DocumentChunk] = []
        
        # BM25 for keyword search
        self.bm25_index = None
        self.bm25_corpus = []
        
        # Search parameters
        self.hybrid_alpha = self.config.get("hybrid_alpha", 0.7)  # Weight for semantic vs keyword
        self.max_chunk_size = self.config.get("max_chunk_size", 512)
        self.chunk_overlap = self.config.get("chunk_overlap", 50)
        
        # Caching
        self.query_cache: Dict[str, List[SemanticSearchResult]] = {}
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Performance tracking
        self.search_stats = {
            "total_searches": 0,
            "cache_hits": 0,
            "avg_search_time": 0.0
        }
    
    async def initialize(self):
        """Initialize the semantic search engine."""
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            self.logger.warning("SentenceTransformers not available - semantic search disabled")
            return
        
        try:
            # Load embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            self.logger.info(f"Initialized embedding model: {self.embedding_model_name} (dim={self.embedding_dim})")
            
            # Initialize FAISS index if available
            if FAISS_AVAILABLE:
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
                self.logger.info("FAISS index initialized")
            else:
                self.logger.warning("FAISS not available - using numpy for vector search")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize semantic search: {e}")
            raise
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[DocumentChunk]:
        """
        Split text into chunks for indexing.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks
            
        Returns:
            List of document chunks
        """
        
        if not text.strip():
            return []
        
        chunks = []
        metadata = metadata or {}
        
        # Simple sentence-based chunking
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            # Check if adding this sentence would exceed max chunk size
            if len(current_chunk) + len(sentence) + 2 > self.max_chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        metadata=metadata.copy()
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        # Take last part of current chunk as overlap
                        overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                        current_chunk = overlap_text + ". " + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata=metadata.copy()
            )
            chunks.append(chunk)
        
        return chunks
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the search index.
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
        """
        
        if not self.embedding_model:
            self.logger.warning("Embedding model not initialized")
            return
        
        all_chunks = []
        
        # Process each document
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Add document info to metadata
            metadata["source"] = doc.get("source", "unknown")
            metadata["doc_id"] = doc.get("id", f"doc_{len(self.document_chunks)}")
            
            # Chunk the document
            chunks = self.chunk_text(content, metadata)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return
        
        # Generate embeddings
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_model.encode(chunk_texts, convert_to_numpy=True)
        
        # Add embeddings to chunks
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk.embedding = embedding
        
        # Add to document store
        self.document_chunks.extend(all_chunks)
        
        # Update FAISS index
        if self.faiss_index is not None:
            self.faiss_index.add(embeddings)
        
        # Update BM25 index
        if BM25_AVAILABLE:
            # Tokenize for BM25
            tokenized_chunks = [chunk.content.lower().split() for chunk in all_chunks]
            
            if self.bm25_index is None:
                self.bm25_corpus = tokenized_chunks
                self.bm25_index = BM25Okapi(self.bm25_corpus)
            else:
                # Rebuild BM25 index with all documents
                self.bm25_corpus.extend(tokenized_chunks)
                self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        self.logger.info(f"Added {len(all_chunks)} chunks to search index (total: {len(self.document_chunks)})")
    
    async def search(
        self, 
        query: str, 
        k: int = 5, 
        min_score: float = 0.0,
        search_type: str = "hybrid"  # "semantic", "keyword", or "hybrid"
    ) -> List[SemanticSearchResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            min_score: Minimum similarity score
            search_type: Type of search to perform
            
        Returns:
            List of search results
        """
        
        if not self.document_chunks:
            return []
        
        # Check cache
        cache_key = f"{query}:{k}:{min_score}:{search_type}"
        if cache_key in self.query_cache:
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and datetime.now() - cache_time < timedelta(seconds=self.cache_ttl):
                self.search_stats["cache_hits"] += 1
                return self.query_cache[cache_key]
        
        start_time = datetime.now()
        self.search_stats["total_searches"] += 1
        
        try:
            if search_type == "semantic":
                results = await self._semantic_search(query, k, min_score)
            elif search_type == "keyword":
                results = await self._keyword_search(query, k, min_score)
            else:  # hybrid
                results = await self._hybrid_search(query, k, min_score)
            
            # Cache results
            self.query_cache[cache_key] = results
            self.cache_timestamps[cache_key] = datetime.now()
            
            # Update stats
            search_time = (datetime.now() - start_time).total_seconds()
            self.search_stats["avg_search_time"] = (
                (self.search_stats["avg_search_time"] * (self.search_stats["total_searches"] - 1) + search_time) /
                self.search_stats["total_searches"]
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    async def _semantic_search(self, query: str, k: int, min_score: float) -> List[SemanticSearchResult]:
        """Perform semantic vector search."""
        
        if not self.embedding_model:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        if self.faiss_index and len(self.document_chunks) > 0:
            # Use FAISS for efficient search
            scores, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1), 
                min(k * 2, len(self.document_chunks))  # Get more candidates
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_chunks) and score >= min_score:
                    chunk = self.document_chunks[idx]
                    results.append(SemanticSearchResult(
                        content=chunk.content,
                        score=float(score),
                        metadata=chunk.metadata,
                        source=chunk.metadata.get("source", "unknown"),
                        chunk_id=chunk.chunk_id
                    ))
            
            return sorted(results, key=lambda x: x.score, reverse=True)[:k]
        
        else:
            # Fallback to numpy-based search
            similarities = []
            
            for chunk in self.document_chunks:
                if chunk.embedding is not None:
                    # Cosine similarity
                    similarity = np.dot(query_embedding, chunk.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                    )
                    similarities.append((similarity, chunk))
            
            # Sort by similarity and filter
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            for score, chunk in similarities[:k]:
                if score >= min_score:
                    results.append(SemanticSearchResult(
                        content=chunk.content,
                        score=float(score),
                        metadata=chunk.metadata,
                        source=chunk.metadata.get("source", "unknown"),
                        chunk_id=chunk.chunk_id
                    ))
            
            return results
    
    async def _keyword_search(self, query: str, k: int, min_score: float) -> List[SemanticSearchResult]:
        """Perform keyword-based search using BM25."""
        
        if not BM25_AVAILABLE or not self.bm25_index:
            # Fallback to simple text matching
            return await self._simple_text_search(query, k, min_score)
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Create results
        results = []
        for i, score in enumerate(scores):
            if i < len(self.document_chunks) and score >= min_score:
                chunk = self.document_chunks[i]
                results.append(SemanticSearchResult(
                    content=chunk.content,
                    score=float(score),
                    metadata=chunk.metadata,
                    source=chunk.metadata.get("source", "unknown"),
                    chunk_id=chunk.chunk_id
                ))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    async def _simple_text_search(self, query: str, k: int, min_score: float) -> List[SemanticSearchResult]:
        """Simple text-based search fallback."""
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for chunk in self.document_chunks:
            content_lower = chunk.content.lower()
            content_words = set(content_lower.split())
            
            # Calculate simple word overlap score
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                score = overlap / len(query_words.union(content_words))
                
                if score >= min_score:
                    results.append(SemanticSearchResult(
                        content=chunk.content,
                        score=score,
                        metadata=chunk.metadata,
                        source=chunk.metadata.get("source", "unknown"),
                        chunk_id=chunk.chunk_id
                    ))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    async def _hybrid_search(self, query: str, k: int, min_score: float) -> List[SemanticSearchResult]:
        """Perform hybrid semantic + keyword search."""
        
        # Get results from both methods
        semantic_results = await self._semantic_search(query, k * 2, 0.0)  # Lower threshold for combining
        keyword_results = await self._keyword_search(query, k * 2, 0.0)
        
        # Combine and re-score
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            key = result.chunk_id or result.content[:50]
            combined_results[key] = {
                "result": result,
                "semantic_score": result.score,
                "keyword_score": 0.0
            }
        
        # Add keyword results
        for result in keyword_results:
            key = result.chunk_id or result.content[:50]
            if key in combined_results:
                combined_results[key]["keyword_score"] = result.score
            else:
                combined_results[key] = {
                    "result": result,
                    "semantic_score": 0.0,
                    "keyword_score": result.score
                }
        
        # Calculate hybrid scores
        final_results = []
        for key, data in combined_results.items():
            # Normalize scores (both should be 0-1 range)
            semantic_norm = min(1.0, max(0.0, data["semantic_score"]))
            keyword_norm = min(1.0, max(0.0, data["keyword_score"] / 10.0))  # BM25 scores can be > 1
            
            # Hybrid score
            hybrid_score = (self.hybrid_alpha * semantic_norm + 
                          (1 - self.hybrid_alpha) * keyword_norm)
            
            if hybrid_score >= min_score:
                result = data["result"]
                result.score = hybrid_score
                final_results.append(result)
        
        # Sort by hybrid score and return top k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:k]
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get search performance statistics."""
        return {
            "total_documents": len(self.document_chunks),
            "embedding_model": self.embedding_model_name,
            "index_type": "FAISS" if self.faiss_index else "numpy",
            "bm25_available": BM25_AVAILABLE and self.bm25_index is not None,
            "cache_size": len(self.query_cache),
            "search_stats": self.search_stats.copy()
        }
    
    async def add_code_repository(self, repo_path: Path, file_extensions: Optional[List[str]] = None):
        """
        Add a code repository to the search index.
        
        Args:
            repo_path: Path to the repository
            file_extensions: File extensions to include (default: common code files)
        """
        
        if file_extensions is None:
            file_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rb', '.go', '.rs', '.php']
        
        documents = []
        
        for ext in file_extensions:
            for file_path in repo_path.rglob(f"*{ext}"):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        
                        # Skip very large files
                        if len(content) > 100000:  # 100KB limit
                            continue
                        
                        documents.append({
                            "content": content,
                            "metadata": {
                                "file_path": str(file_path),
                                "file_extension": ext,
                                "file_size": len(content),
                                "content_type": "code"
                            },
                            "source": f"file://{file_path}",
                            "id": str(file_path)
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Could not read {file_path}: {e}")
        
        if documents:
            await self.add_documents(documents)
            self.logger.info(f"Added {len(documents)} code files from {repo_path}")
    
    async def add_documentation(self, docs_path: Path):
        """
        Add documentation files to the search index.
        
        Args:
            docs_path: Path to documentation directory
        """
        
        doc_extensions = ['.md', '.rst', '.txt', '.doc']
        documents = []
        
        for ext in doc_extensions:
            for file_path in docs_path.rglob(f"*{ext}"):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        
                        documents.append({
                            "content": content,
                            "metadata": {
                                "file_path": str(file_path),
                                "file_extension": ext,
                                "content_type": "documentation"
                            },
                            "source": f"file://{file_path}",
                            "id": str(file_path)
                        })
                        
                    except Exception as e:
                        self.logger.warning(f"Could not read {file_path}: {e}")
        
        if documents:
            await self.add_documents(documents)
            self.logger.info(f"Added {len(documents)} documentation files from {docs_path}")
    
    def clear_cache(self):
        """Clear the search cache."""
        self.query_cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("Search cache cleared")
    
    async def cleanup(self):
        """Cleanup resources."""
        self.query_cache.clear()
        self.cache_timestamps.clear()
        
        if hasattr(self.embedding_model, 'close'):
            try:
                self.embedding_model.close()
            except:
                pass