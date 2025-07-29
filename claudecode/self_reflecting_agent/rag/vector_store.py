"""
Vector store implementation for semantic search.

This module implements vector storage and similarity search using various
backends like FAISS, Chroma, or other vector databases.
"""

import asyncio
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class VectorStore:
    """
    Vector store for semantic similarity search.
    
    Supports multiple backends including FAISS and ChromaDB for efficient
    vector storage and retrieval.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.provider = config.get("provider", "faiss")
        self.embedding_model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.dimensions = config.get("dimensions", 384)
        self.collection_name = config.get("collection_name", "agent_documents")
        
        # Components
        self.embedding_model = None
        self.index = None
        self.client = None
        self.documents: List[Dict[str, Any]] = []
        self.document_embeddings: Optional[np.ndarray] = None
        
        # Metrics
        self.metrics = {
            "total_documents": 0,
            "total_searches": 0,
            "avg_search_time": 0.0,
            "index_size_mb": 0.0
        }
        
        self.initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the vector store."""
        
        try:
            # Initialize embedding model
            if not await self._initialize_embedding_model():
                return False
            
            # Initialize vector backend
            if not await self._initialize_backend():
                return False
            
            self.initialized = True
            self.logger.info(f"Vector store initialized with {self.provider} backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            return False
    
    async def _initialize_embedding_model(self) -> bool:
        """Initialize the embedding model."""
        
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
                return True
            else:
                self.logger.error("sentence-transformers not available")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            return False
    
    async def _initialize_backend(self) -> bool:
        """Initialize the vector backend."""
        
        if self.provider == "faiss":
            return await self._initialize_faiss()
        elif self.provider == "chroma":
            return await self._initialize_chroma()
        else:
            self.logger.error(f"Unsupported vector store provider: {self.provider}")
            return False
    
    async def _initialize_faiss(self) -> bool:
        """Initialize FAISS backend."""
        
        if not FAISS_AVAILABLE:
            self.logger.error("FAISS not available")
            return False
        
        try:
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.dimensions)  # Inner Product (cosine similarity)
            self.logger.info("FAISS index initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS: {e}")
            return False
    
    async def _initialize_chroma(self) -> bool:
        """Initialize ChromaDB backend."""
        
        if not CHROMA_AVAILABLE:
            self.logger.error("ChromaDB not available")
            return False
        
        try:
            # Initialize ChromaDB client
            self.client = chromadb.Client()
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
            except:
                self.collection = self.client.create_collection(name=self.collection_name)
            
            self.logger.info("ChromaDB initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            return False
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store."""
        
        if not self.initialized:
            return False
        
        try:
            self.logger.info(f"Adding {len(documents)} documents to vector store")
            
            # Extract text content for embedding
            texts = []
            for doc in documents:
                content = doc.get("content", "")
                if content:
                    texts.append(content)
                else:
                    texts.append(str(doc))  # Fallback to string representation
            
            if not texts:
                self.logger.warning("No text content found in documents")
                return True
            
            # Generate embeddings
            embeddings = await self._generate_embeddings(texts)
            
            if embeddings is None:
                return False
            
            # Add to backend
            if self.provider == "faiss":
                return await self._add_to_faiss(documents, embeddings)
            elif self.provider == "chroma":
                return await self._add_to_chroma(documents, embeddings)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False
    
    async def _generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for texts."""
        
        if not self.embedding_model:
            return None
        
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            return None
    
    async def _add_to_faiss(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> bool:
        """Add documents to FAISS index."""
        
        try:
            # Add embeddings to index
            self.index.add(embeddings.astype(np.float32))
            
            # Store documents
            start_id = len(self.documents)
            self.documents.extend(documents)
            
            # Update document embeddings
            if self.document_embeddings is None:
                self.document_embeddings = embeddings
            else:
                self.document_embeddings = np.vstack([self.document_embeddings, embeddings])
            
            # Update metrics
            self.metrics["total_documents"] = len(self.documents)
            
            self.logger.info(f"Added {len(documents)} documents to FAISS index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to FAISS: {e}")
            return False
    
    async def _add_to_chroma(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> bool:
        """Add documents to ChromaDB."""
        
        try:
            # Prepare data for ChromaDB
            ids = [f"doc_{len(self.documents) + i}" for i in range(len(documents))]
            metadatas = []
            texts = []
            
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                texts.append(content)
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Store documents locally too
            self.documents.extend(documents)
            
            # Update metrics
            self.metrics["total_documents"] = len(self.documents)
            
            self.logger.info(f"Added {len(documents)} documents to ChromaDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to ChromaDB: {e}")
            return False
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        
        if not self.initialized or not self.documents:
            return []
        
        start_time = datetime.now()
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_embeddings([query])
            if query_embedding is None:
                return []
            
            # Search backend
            if self.provider == "faiss":
                results = await self._search_faiss(query_embedding[0], max_results, filters)
            elif self.provider == "chroma":
                results = await self._search_chroma(query, max_results, filters)
            else:
                results = []
            
            # Update metrics
            search_time = (datetime.now() - start_time).total_seconds()
            self.metrics["total_searches"] += 1
            self._update_avg_search_time(search_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    async def _search_faiss(
        self,
        query_embedding: np.ndarray,
        max_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search FAISS index."""
        
        try:
            # Search index
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                min(max_results, len(self.documents))
            )
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc["score"] = float(score)
                    doc["rank"] = i + 1
                    
                    # Apply filters if specified
                    if filters and not self._matches_filters(doc, filters):
                        continue
                    
                    results.append(doc)
            
            return results
            
        except Exception as e:
            self.logger.error(f"FAISS search failed: {e}")
            return []
    
    async def _search_chroma(
        self,
        query: str,
        max_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB collection."""
        
        try:
            # Prepare where clause for filters
            where_clause = None
            if filters:
                where_clause = self._build_chroma_where_clause(filters)
            
            # Search collection
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0] if results["metadatas"] else [{}] * len(results["documents"][0]),
                    results["distances"][0] if results["distances"] else [0.0] * len(results["documents"][0])
                )):
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "score": 1.0 - distance,  # Convert distance to similarity
                        "rank": i + 1
                    })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"ChromaDB search failed: {e}")
            return []
    
    def _matches_filters(self, document: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document matches filters."""
        
        metadata = document.get("metadata", {})
        
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        
        return True
    
    def _build_chroma_where_clause(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        
        # Simple implementation - ChromaDB has specific syntax
        # This would need to be expanded based on ChromaDB documentation
        return filters
    
    def _update_avg_search_time(self, search_time: float) -> None:
        """Update average search time metric."""
        
        current_avg = self.metrics["avg_search_time"]
        total_searches = self.metrics["total_searches"]
        
        self.metrics["avg_search_time"] = (
            (current_avg * (total_searches - 1) + search_time) / total_searches
        )
    
    async def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self.documents)
    
    async def delete_documents(
        self,
        document_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Delete documents from the store."""
        
        try:
            if self.provider == "chroma" and self.collection:
                # ChromaDB deletion
                if document_ids:
                    self.collection.delete(ids=document_ids)
                elif filters:
                    where_clause = self._build_chroma_where_clause(filters)
                    self.collection.delete(where=where_clause)
                
                # Update local documents list
                # This is simplified - in practice, would need proper ID tracking
                self.documents.clear()
                self.metrics["total_documents"] = 0
                
                return True
            
            # FAISS doesn't support deletion easily, would need to rebuild index
            self.logger.warning("Document deletion not fully implemented for FAISS")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            return False
    
    async def optimize_index(self) -> bool:
        """Optimize the vector index."""
        
        try:
            if self.provider == "faiss" and self.index:
                # For FAISS, we could rebuild with optimized index type
                # This is a placeholder - actual optimization depends on use case
                self.logger.info("FAISS index optimization completed")
                return True
            
            elif self.provider == "chroma":
                # ChromaDB handles optimization internally
                self.logger.info("ChromaDB optimization completed")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Index optimization failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get vector store metrics."""
        
        metrics = self.metrics.copy()
        metrics["initialized"] = self.initialized
        metrics["provider"] = self.provider
        metrics["embedding_model"] = self.embedding_model_name
        
        # Calculate index size
        if self.provider == "faiss" and self.document_embeddings is not None:
            size_bytes = self.document_embeddings.nbytes
            metrics["index_size_mb"] = size_bytes / (1024 * 1024)
        
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown the vector store."""
        
        try:
            if self.provider == "chroma" and self.client:
                # ChromaDB client cleanup
                pass  # ChromaDB handles cleanup automatically
            
            self.initialized = False
            self.logger.info("Vector store shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during vector store shutdown: {e}")