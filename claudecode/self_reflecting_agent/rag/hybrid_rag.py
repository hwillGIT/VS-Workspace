"""
Hybrid RAG system combining BM25 and vector search.

This module implements the main HybridRAG class that orchestrates
keyword-based and semantic search with fusion techniques for
optimal information retrieval.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path

from .vector_store import VectorStore
from .bm25_search import BM25Search
from .document_processor import DocumentProcessor
from .retrieval_fusion import RetrievalFusion


class HybridRAG:
    """
    Hybrid Retrieval-Augmented Generation system.
    
    This class combines BM25 keyword search with vector similarity search,
    using Reciprocal Rank Fusion to merge and rank results for optimal
    information retrieval performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.vector_store = VectorStore(config.get("vector_store", {}))
        self.bm25_search = BM25Search(config.get("bm25", {}))
        self.document_processor = DocumentProcessor(config.get("document_processor", {}))
        self.fusion = RetrievalFusion(config.get("fusion", {}))
        
        # Configuration
        self.max_results = config.get("max_results", 10)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        self.enable_reranking = config.get("enable_reranking", True)
        
        # Weights for fusion
        self.bm25_weight = config.get("bm25_weight", 0.3)
        self.vector_weight = config.get("vector_weight", 0.7)
        
        # Performance tracking
        self.search_metrics = {
            "total_searches": 0,
            "avg_response_time": 0.0,
            "cache_hits": 0,
            "bm25_searches": 0,
            "vector_searches": 0,
            "hybrid_searches": 0
        }
        
        # Simple cache for frequent queries
        self.query_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_max_size = config.get("cache_max_size", 100)
        
        self.logger.info("Hybrid RAG system initialized")
    
    async def initialize(self) -> bool:
        """Initialize all RAG components."""
        
        try:
            # Initialize components in parallel
            initialization_tasks = [
                self.vector_store.initialize(),
                self.bm25_search.initialize(),
                self.document_processor.initialize()
            ]
            
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # Check for initialization failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    component_names = ["vector_store", "bm25_search", "document_processor"]
                    self.logger.error(f"Failed to initialize {component_names[i]}: {result}")
                    return False
                elif not result:
                    component_names = ["vector_store", "bm25_search", "document_processor"]
                    self.logger.error(f"Failed to initialize {component_names[i]}")
                    return False
            
            self.logger.info("All RAG components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"RAG initialization failed: {e}")
            return False
    
    async def add_documents(
        self, 
        documents: List[Dict[str, Any]],
        source: str = "unknown"
    ) -> bool:
        """
        Add documents to the RAG system.
        
        Args:
            documents: List of documents with 'content' and optional metadata
            source: Source identifier for the documents
            
        Returns:
            True if documents were successfully added
        """
        
        try:
            self.logger.info(f"Adding {len(documents)} documents from source: {source}")
            
            # Process documents
            processed_documents = []
            for doc in documents:
                processed_doc = await self.document_processor.process_document(
                    content=doc.get("content", ""),
                    metadata={
                        "source": source,
                        "timestamp": datetime.now().isoformat(),
                        **doc.get("metadata", {})
                    }
                )
                processed_documents.append(processed_doc)
            
            # Add to both search systems in parallel
            tasks = [
                self.vector_store.add_documents(processed_documents),
                self.bm25_search.add_documents(processed_documents)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            success = True
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    system_names = ["vector_store", "bm25_search"]
                    self.logger.error(f"Failed to add documents to {system_names[i]}: {result}")
                    success = False
                elif not result:
                    system_names = ["vector_store", "bm25_search"]
                    self.logger.error(f"Failed to add documents to {system_names[i]}")
                    success = False
            
            if success:
                # Clear query cache since index has changed
                self.query_cache.clear()
                self.logger.info(f"Successfully added {len(documents)} documents")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to add documents: {e}")
            return False
    
    async def search(
        self, 
        query: str,
        search_type: str = "hybrid",
        max_results: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            search_type: Type of search ("hybrid", "vector", "bm25")
            max_results: Maximum number of results to return
            filters: Optional filters for search
            
        Returns:
            List of relevant documents with scores
        """
        
        start_time = datetime.now()
        
        try:
            # Update metrics
            self.search_metrics["total_searches"] += 1
            
            # Check cache first
            cache_key = f"{query}:{search_type}:{max_results}:{str(filters)}"
            if cache_key in self.query_cache:
                self.search_metrics["cache_hits"] += 1
                return self.query_cache[cache_key]
            
            # Set default max_results
            if max_results is None:
                max_results = self.max_results
            
            # Route to appropriate search method
            if search_type == "vector":
                results = await self._vector_search(query, max_results, filters)
                self.search_metrics["vector_searches"] += 1
                
            elif search_type == "bm25":
                results = await self._bm25_search(query, max_results, filters)
                self.search_metrics["bm25_searches"] += 1
                
            else:  # hybrid
                results = await self._hybrid_search(query, max_results, filters)
                self.search_metrics["hybrid_searches"] += 1
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results
                if result.get("score", 0.0) >= self.similarity_threshold
            ]
            
            # Limit results
            final_results = filtered_results[:max_results]
            
            # Cache results
            if len(self.query_cache) < self.cache_max_size:
                self.query_cache[cache_key] = final_results
            
            # Update performance metrics
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_response_time_metric(response_time)
            
            self.logger.debug(f"Search completed: {len(final_results)} results in {response_time:.3f}s")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    async def _vector_search(
        self, 
        query: str, 
        max_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        
        return await self.vector_store.search(
            query=query,
            max_results=max_results,
            filters=filters
        )
    
    async def _bm25_search(
        self, 
        query: str, 
        max_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform BM25 keyword search."""
        
        return await self.bm25_search.search(
            query=query,
            max_results=max_results,
            filters=filters
        )
    
    async def _hybrid_search(
        self, 
        query: str, 
        max_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and BM25."""
        
        # Perform both searches in parallel
        vector_task = self._vector_search(query, max_results * 2, filters)  # Get more for fusion
        bm25_task = self._bm25_search(query, max_results * 2, filters)
        
        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)
        
        # Fuse the results
        fused_results = await self.fusion.fuse_results(
            vector_results=vector_results,
            bm25_results=bm25_results,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
            max_results=max_results
        )
        
        return fused_results
    
    async def add_document_from_file(
        self, 
        file_path: Union[str, Path],
        source: Optional[str] = None
    ) -> bool:
        """Add a document from a file."""
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return False
            
            # Read file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            # Create document
            document = {
                "content": content,
                "metadata": {
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "file_extension": file_path.suffix
                }
            }
            
            source = source or f"file:{file_path.parent.name}"
            
            return await self.add_documents([document], source)
            
        except Exception as e:
            self.logger.error(f"Failed to add document from file {file_path}: {e}")
            return False
    
    async def add_documents_from_directory(
        self, 
        directory_path: Union[str, Path],
        file_patterns: List[str] = None,
        recursive: bool = True
    ) -> int:
        """
        Add all documents from a directory.
        
        Args:
            directory_path: Path to directory
            file_patterns: List of file patterns to include (e.g., ["*.py", "*.md"])
            recursive: Whether to search recursively
            
        Returns:
            Number of documents successfully added
        """
        
        try:
            directory_path = Path(directory_path)
            
            if not directory_path.exists() or not directory_path.is_dir():
                self.logger.error(f"Directory not found: {directory_path}")
                return 0
            
            # Default file patterns
            if file_patterns is None:
                file_patterns = ["*.py", "*.md", "*.txt", "*.rst", "*.json", "*.yaml", "*.yml"]
            
            # Find all matching files
            files_to_process = []
            
            for pattern in file_patterns:
                if recursive:
                    files_to_process.extend(directory_path.rglob(pattern))
                else:
                    files_to_process.extend(directory_path.glob(pattern))
            
            # Remove duplicates
            files_to_process = list(set(files_to_process))
            
            self.logger.info(f"Found {len(files_to_process)} files to process in {directory_path}")
            
            # Process files in batches
            batch_size = 10
            total_added = 0
            
            for i in range(0, len(files_to_process), batch_size):
                batch_files = files_to_process[i:i + batch_size]
                batch_documents = []
                
                for file_path in batch_files:
                    try:
                        # Skip large files
                        if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                            self.logger.warning(f"Skipping large file: {file_path}")
                            continue
                        
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        document = {
                            "content": content,
                            "metadata": {
                                "file_path": str(file_path.relative_to(directory_path)),
                                "file_name": file_path.name,
                                "file_size": file_path.stat().st_size,
                                "file_extension": file_path.suffix
                            }
                        }
                        batch_documents.append(document)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to read file {file_path}: {e}")
                        continue
                
                # Add batch to RAG system
                if batch_documents:
                    source = f"directory:{directory_path.name}"
                    success = await self.add_documents(batch_documents, source)
                    if success:
                        total_added += len(batch_documents)
            
            self.logger.info(f"Successfully added {total_added} documents from {directory_path}")
            return total_added
            
        except Exception as e:
            self.logger.error(f"Failed to add documents from directory {directory_path}: {e}")
            return 0
    
    async def semantic_search(
        self, 
        query: str,
        context: Optional[str] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with optional context.
        
        Args:
            query: Search query
            context: Optional context to improve search
            max_results: Maximum results to return
            
        Returns:
            List of semantically relevant documents
        """
        
        # Enhance query with context if provided
        enhanced_query = query
        if context:
            enhanced_query = f"Context: {context}\n\nQuery: {query}"
        
        # Use vector search for semantic similarity
        return await self.search(
            query=enhanced_query,
            search_type="vector",
            max_results=max_results
        )
    
    async def keyword_search(
        self, 
        query: str,
        exact_match: bool = False,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search.
        
        Args:
            query: Search query with keywords
            exact_match: Whether to require exact phrase matching
            max_results: Maximum results to return
            
        Returns:
            List of documents matching keywords
        """
        
        # Modify query for exact matching if requested
        search_query = f'"{query}"' if exact_match else query
        
        # Use BM25 search for keyword matching
        return await self.search(
            query=search_query,
            search_type="bm25",
            max_results=max_results
        )
    
    def get_search_metrics(self) -> Dict[str, Any]:
        """Get search performance metrics."""
        
        return {
            **self.search_metrics,
            "vector_store_metrics": self.vector_store.get_metrics(),
            "bm25_metrics": self.bm25_search.get_metrics(),
            "cache_size": len(self.query_cache),
            "cache_hit_rate": (
                self.search_metrics["cache_hits"] / max(self.search_metrics["total_searches"], 1)
            )
        }
    
    def _update_response_time_metric(self, response_time: float) -> None:
        """Update average response time metric."""
        
        current_avg = self.search_metrics["avg_response_time"]
        total_searches = self.search_metrics["total_searches"]
        
        # Calculate new average
        self.search_metrics["avg_response_time"] = (
            (current_avg * (total_searches - 1) + response_time) / total_searches
        )
    
    async def clear_cache(self) -> None:
        """Clear the query cache."""
        self.query_cache.clear()
        self.logger.info("Query cache cleared")
    
    async def optimize_indices(self) -> bool:
        """Optimize search indices for better performance."""
        
        try:
            # Optimize both indices in parallel
            tasks = [
                self.vector_store.optimize_index(),
                self.bm25_search.optimize_index()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success = all(
                not isinstance(result, Exception) and result
                for result in results
            )
            
            if success:
                self.logger.info("Search indices optimized successfully")
            else:
                self.logger.warning("Some index optimizations failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Index optimization failed: {e}")
            return False
    
    async def get_document_count(self) -> int:
        """Get total number of documents in the system."""
        
        try:
            # Get count from vector store (should be authoritative)
            return await self.vector_store.get_document_count()
        except Exception as e:
            self.logger.error(f"Failed to get document count: {e}")
            return 0
    
    async def delete_documents(
        self, 
        document_ids: List[str] = None,
        filters: Dict[str, Any] = None
    ) -> bool:
        """Delete documents from the system."""
        
        try:
            # Delete from both systems in parallel
            tasks = [
                self.vector_store.delete_documents(document_ids, filters),
                self.bm25_search.delete_documents(document_ids, filters)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success = all(
                not isinstance(result, Exception) and result
                for result in results
            )
            
            if success:
                # Clear cache since documents were deleted
                self.query_cache.clear()
                self.logger.info(f"Successfully deleted documents")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the RAG system and cleanup resources."""
        
        try:
            # Shutdown components in parallel
            tasks = [
                self.vector_store.shutdown(),
                self.bm25_search.shutdown()
            ]
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Clear cache
            self.query_cache.clear()
            
            self.logger.info("Hybrid RAG system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during RAG shutdown: {e}")