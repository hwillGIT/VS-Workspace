"""
Retrieval fusion implementation for combining search results.

This module implements Reciprocal Rank Fusion (RRF) and other fusion techniques
to combine results from different retrieval systems (BM25 and vector search)
for optimal information retrieval performance.
"""

import asyncio
import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class RetrievalFusion:
    """
    Retrieval fusion system for combining multiple search results.
    
    Implements various fusion techniques including Reciprocal Rank Fusion (RRF),
    Weighted Score Fusion, and Normalized Score Fusion to optimally combine
    results from different retrieval systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Fusion parameters
        self.default_k = config.get("rrf_k", 60)  # RRF parameter
        self.score_normalization = config.get("score_normalization", "min_max")
        self.fusion_method = config.get("fusion_method", "rrf")  # rrf, weighted, rank_fusion
        
        # Deduplication settings
        self.enable_deduplication = config.get("enable_deduplication", True)
        self.similarity_threshold = config.get("dedup_similarity_threshold", 0.9)
        self.content_hash_dedup = config.get("content_hash_dedup", True)
        
        # Diversity settings
        self.promote_diversity = config.get("promote_diversity", True)
        self.diversity_weight = config.get("diversity_weight", 0.1)
        
        # Performance tracking
        self.fusion_stats = {
            "total_fusions": 0,
            "avg_fusion_time": 0.0,
            "documents_deduplicated": 0,
            "fusion_methods_used": defaultdict(int),
            "avg_input_lists_size": 0.0,
            "avg_output_size": 0.0
        }
        
        self.logger.info("Retrieval fusion system initialized")
    
    async def fuse_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        max_results: int = 10,
        fusion_method: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fuse results from vector and BM25 search.
        
        Args:
            vector_results: Results from vector similarity search
            bm25_results: Results from BM25 keyword search
            vector_weight: Weight for vector search results
            bm25_weight: Weight for BM25 search results
            max_results: Maximum number of results to return
            fusion_method: Specific fusion method to use
            
        Returns:
            Fused and ranked results
        """
        
        start_time = datetime.now()
        
        try:
            # Update stats
            self.fusion_stats["total_fusions"] += 1
            method = fusion_method or self.fusion_method
            self.fusion_stats["fusion_methods_used"][method] += 1
            
            # Log input sizes
            input_size = len(vector_results) + len(bm25_results)
            self._update_avg_input_size(input_size)
            
            self.logger.debug(f"Fusing {len(vector_results)} vector + {len(bm25_results)} BM25 results")
            
            # Apply fusion method
            if method == "rrf":
                fused_results = await self._reciprocal_rank_fusion(
                    vector_results, bm25_results, vector_weight, bm25_weight
                )
            elif method == "weighted":
                fused_results = await self._weighted_score_fusion(
                    vector_results, bm25_results, vector_weight, bm25_weight
                )
            elif method == "rank_fusion":
                fused_results = await self._rank_fusion(
                    vector_results, bm25_results, vector_weight, bm25_weight
                )
            else:
                # Fallback to RRF
                fused_results = await self._reciprocal_rank_fusion(
                    vector_results, bm25_results, vector_weight, bm25_weight
                )
            
            # Deduplicate results
            if self.enable_deduplication:
                fused_results = await self._deduplicate_results(fused_results)
            
            # Promote diversity if enabled
            if self.promote_diversity:
                fused_results = await self._promote_diversity(fused_results)
            
            # Limit results
            final_results = fused_results[:max_results]
            
            # Update ranks
            for i, result in enumerate(final_results):
                result["rank"] = i + 1
                result["fusion_method"] = method
            
            # Update performance stats
            fusion_time = (datetime.now() - start_time).total_seconds()
            self._update_avg_fusion_time(fusion_time)
            self._update_avg_output_size(len(final_results))
            
            self.logger.debug(f"Fusion completed: {len(final_results)} results in {fusion_time:.3f}s")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Fusion failed: {e}")
            # Fallback: return top results from both lists
            return await self._fallback_fusion(vector_results, bm25_results, max_results)
    
    async def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float,
        bm25_weight: float,
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Implement Reciprocal Rank Fusion (RRF).
        
        RRF combines rankings from multiple systems using the formula:
        RRF_score = Σ(weight / (k + rank))
        """
        
        k = k or self.default_k
        document_scores = defaultdict(float)
        document_data = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            doc_id = self._get_document_id(result)
            rrf_score = vector_weight / (k + rank)
            document_scores[doc_id] += rrf_score
            
            if doc_id not in document_data:
                document_data[doc_id] = result.copy()
                document_data[doc_id]["fusion_scores"] = {"vector_rrf": rrf_score}
            else:
                document_data[doc_id]["fusion_scores"]["vector_rrf"] = rrf_score
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            doc_id = self._get_document_id(result)
            rrf_score = bm25_weight / (k + rank)
            document_scores[doc_id] += rrf_score
            
            if doc_id not in document_data:
                document_data[doc_id] = result.copy()
                document_data[doc_id]["fusion_scores"] = {"bm25_rrf": rrf_score}
            else:
                document_data[doc_id]["fusion_scores"]["bm25_rrf"] = rrf_score
        
        # Create fused results
        fused_results = []
        for doc_id, total_score in document_scores.items():
            result = document_data[doc_id].copy()
            result["fusion_score"] = total_score
            result["score"] = total_score  # Update main score
            fused_results.append(result)
        
        # Sort by fusion score
        fused_results.sort(key=lambda x: x["fusion_score"], reverse=True)
        
        return fused_results
    
    async def _weighted_score_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float,
        bm25_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Implement weighted score fusion.
        
        Combines normalized scores from both systems using weighted average.
        """
        
        # Normalize scores
        normalized_vector = await self._normalize_scores(vector_results)
        normalized_bm25 = await self._normalize_scores(bm25_results)
        
        document_scores = defaultdict(float)
        document_data = {}
        
        # Process vector results
        for result in normalized_vector:
            doc_id = self._get_document_id(result)
            weighted_score = result["normalized_score"] * vector_weight
            document_scores[doc_id] += weighted_score
            
            if doc_id not in document_data:
                document_data[doc_id] = result.copy()
                document_data[doc_id]["fusion_scores"] = {"vector_weighted": weighted_score}
            else:
                document_data[doc_id]["fusion_scores"]["vector_weighted"] = weighted_score
        
        # Process BM25 results
        for result in normalized_bm25:
            doc_id = self._get_document_id(result)
            weighted_score = result["normalized_score"] * bm25_weight
            document_scores[doc_id] += weighted_score
            
            if doc_id not in document_data:
                document_data[doc_id] = result.copy()
                document_data[doc_id]["fusion_scores"] = {"bm25_weighted": weighted_score}
            else:
                document_data[doc_id]["fusion_scores"]["bm25_weighted"] = weighted_score
        
        # Create fused results
        fused_results = []
        for doc_id, total_score in document_scores.items():
            result = document_data[doc_id].copy()
            result["fusion_score"] = total_score
            result["score"] = total_score
            fused_results.append(result)
        
        # Sort by fusion score
        fused_results.sort(key=lambda x: x["fusion_score"], reverse=True)
        
        return fused_results
    
    async def _rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float,
        bm25_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Implement rank-based fusion without reciprocal transformation.
        
        Uses weighted rank positions to determine final ranking.
        """
        
        document_ranks = defaultdict(list)
        document_data = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            doc_id = self._get_document_id(result)
            weighted_rank = rank * (1 / vector_weight) if vector_weight > 0 else float('inf')
            document_ranks[doc_id].append(("vector", rank, weighted_rank))
            
            if doc_id not in document_data:
                document_data[doc_id] = result.copy()
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            doc_id = self._get_document_id(result)
            weighted_rank = rank * (1 / bm25_weight) if bm25_weight > 0 else float('inf')
            document_ranks[doc_id].append(("bm25", rank, weighted_rank))
            
            if doc_id not in document_data:
                document_data[doc_id] = result.copy()
        
        # Calculate final scores
        fused_results = []
        for doc_id, ranks in document_ranks.items():
            # Use minimum weighted rank as final score (lower is better)
            min_weighted_rank = min(rank_info[2] for rank_info in ranks)
            
            # Convert rank to score (higher is better)
            fusion_score = 1.0 / (1.0 + min_weighted_rank)
            
            result = document_data[doc_id].copy()
            result["fusion_score"] = fusion_score
            result["score"] = fusion_score
            result["rank_info"] = ranks
            
            fused_results.append(result)
        
        # Sort by fusion score
        fused_results.sort(key=lambda x: x["fusion_score"], reverse=True)
        
        return fused_results
    
    async def _normalize_scores(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize scores in results list."""
        
        if not results:
            return results
        
        scores = [result.get("score", 0.0) for result in results]
        
        if not scores or max(scores) == min(scores):
            # All scores are the same or missing
            for result in results:
                result["normalized_score"] = 1.0
            return results
        
        # Apply normalization method
        if self.score_normalization == "min_max":
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score
            
            for i, result in enumerate(results):
                normalized = (scores[i] - min_score) / score_range if score_range > 0 else 0.0
                result["normalized_score"] = normalized
        
        elif self.score_normalization == "z_score" and SCIPY_AVAILABLE:
            # Z-score normalization
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            for i, result in enumerate(results):
                if std_score > 0:
                    z_score = (scores[i] - mean_score) / std_score
                    # Convert z-score to positive range using sigmoid
                    normalized = 1 / (1 + math.exp(-z_score))
                else:
                    normalized = 0.5
                result["normalized_score"] = normalized
        
        else:
            # Fallback: simple scale to [0,1]
            max_score = max(scores)
            for i, result in enumerate(results):
                result["normalized_score"] = scores[i] / max_score if max_score > 0 else 0.0
        
        return results
    
    async def _deduplicate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate documents from results."""
        
        if not self.enable_deduplication:
            return results
        
        seen_hashes = set()
        seen_content = set()
        deduplicated = []
        
        for result in results:
            # Check content hash if available
            content_hash = result.get("metadata", {}).get("content_hash")
            if content_hash and self.content_hash_dedup:
                if content_hash in seen_hashes:
                    self.fusion_stats["documents_deduplicated"] += 1
                    continue
                seen_hashes.add(content_hash)
            
            # Check content similarity
            content = result.get("content", "")
            if content:
                # Simple content deduplication
                content_key = content[:200].strip().lower()  # First 200 chars
                
                if content_key in seen_content:
                    self.fusion_stats["documents_deduplicated"] += 1
                    continue
                
                seen_content.add(content_key)
            
            deduplicated.append(result)
        
        return deduplicated
    
    async def _promote_diversity(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Promote diversity in results by adjusting scores."""
        
        if not results or not self.promote_diversity:
            return results
        
        # Group by source or type
        source_counts = defaultdict(int)
        
        for result in results:
            source = result.get("metadata", {}).get("source", "unknown")
            source_counts[source] += 1
        
        # Apply diversity penalty
        adjusted_results = []
        source_seen = defaultdict(int)
        
        for result in results:
            source = result.get("metadata", {}).get("source", "unknown")
            source_seen[source] += 1
            
            # Calculate diversity penalty
            penalty = source_seen[source] * self.diversity_weight
            original_score = result.get("fusion_score", result.get("score", 0.0))
            
            # Apply penalty (reduce score for repeated sources)
            adjusted_score = original_score * (1.0 - penalty)
            
            result_copy = result.copy()
            result_copy["diversity_adjusted_score"] = adjusted_score
            result_copy["score"] = adjusted_score
            result_copy["diversity_penalty"] = penalty
            
            adjusted_results.append(result_copy)
        
        # Re-sort by adjusted scores
        adjusted_results.sort(key=lambda x: x["score"], reverse=True)
        
        return adjusted_results
    
    def _get_document_id(self, document: Dict[str, Any]) -> str:
        """Generate a unique ID for a document."""
        
        # Try to use existing ID
        if "id" in document:
            return str(document["id"])
        
        # Use content hash if available
        content_hash = document.get("metadata", {}).get("content_hash")
        if content_hash:
            return content_hash
        
        # Generate ID from content
        content = document.get("content", "")
        if content:
            import hashlib
            return hashlib.md5(content.encode()).hexdigest()
        
        # Fallback: use string representation
        return str(hash(str(document)))
    
    async def _fallback_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Fallback fusion method when main fusion fails."""
        
        try:
            # Simple interleaving
            fused = []
            v_idx = b_idx = 0
            
            while len(fused) < max_results and (v_idx < len(vector_results) or b_idx < len(bm25_results)):
                # Alternate between vector and BM25 results
                if v_idx < len(vector_results) and (len(fused) % 2 == 0 or b_idx >= len(bm25_results)):
                    result = vector_results[v_idx].copy()
                    result["source_type"] = "vector"
                    fused.append(result)
                    v_idx += 1
                elif b_idx < len(bm25_results):
                    result = bm25_results[b_idx].copy()
                    result["source_type"] = "bm25"
                    fused.append(result)
                    b_idx += 1
            
            # Set ranks
            for i, result in enumerate(fused):
                result["rank"] = i + 1
                result["fusion_method"] = "fallback"
            
            return fused
            
        except Exception as e:
            self.logger.error(f"Fallback fusion failed: {e}")
            return []
    
    async def fuse_multiple_results(
        self,
        result_lists: List[Tuple[List[Dict[str, Any]], float, str]],
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fuse results from multiple retrieval systems.
        
        Args:
            result_lists: List of (results, weight, system_name) tuples
            max_results: Maximum number of results to return
            
        Returns:
            Fused results from all systems
        """
        
        try:
            if len(result_lists) < 2:
                return result_lists[0][0][:max_results] if result_lists else []
            
            # Use RRF for multiple systems
            document_scores = defaultdict(float)
            document_data = {}
            
            for results, weight, system_name in result_lists:
                for rank, result in enumerate(results, 1):
                    doc_id = self._get_document_id(result)
                    rrf_score = weight / (self.default_k + rank)
                    document_scores[doc_id] += rrf_score
                    
                    if doc_id not in document_data:
                        document_data[doc_id] = result.copy()
                        document_data[doc_id]["fusion_scores"] = {}
                    
                    document_data[doc_id]["fusion_scores"][f"{system_name}_rrf"] = rrf_score
            
            # Create fused results
            fused_results = []
            for doc_id, total_score in document_scores.items():
                result = document_data[doc_id].copy()
                result["fusion_score"] = total_score
                result["score"] = total_score
                fused_results.append(result)
            
            # Sort and limit
            fused_results.sort(key=lambda x: x["fusion_score"], reverse=True)
            final_results = fused_results[:max_results]
            
            # Set ranks
            for i, result in enumerate(final_results):
                result["rank"] = i + 1
                result["fusion_method"] = "multi_rrf"
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Multiple fusion failed: {e}")
            return []
    
    def _update_avg_fusion_time(self, fusion_time: float) -> None:
        """Update average fusion time metric."""
        
        current_avg = self.fusion_stats["avg_fusion_time"]
        total_fusions = self.fusion_stats["total_fusions"]
        
        self.fusion_stats["avg_fusion_time"] = (
            (current_avg * (total_fusions - 1) + fusion_time) / total_fusions
        )
    
    def _update_avg_input_size(self, input_size: int) -> None:
        """Update average input list size metric."""
        
        current_avg = self.fusion_stats["avg_input_lists_size"]
        total_fusions = self.fusion_stats["total_fusions"]
        
        self.fusion_stats["avg_input_lists_size"] = (
            (current_avg * (total_fusions - 1) + input_size) / total_fusions
        )
    
    def _update_avg_output_size(self, output_size: int) -> None:
        """Update average output size metric."""
        
        current_avg = self.fusion_stats["avg_output_size"]
        total_fusions = self.fusion_stats["total_fusions"]
        
        self.fusion_stats["avg_output_size"] = (
            (current_avg * (total_fusions - 1) + output_size) / total_fusions
        )
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion performance statistics."""
        
        return {
            **self.fusion_stats,
            "config": {
                "default_k": self.default_k,
                "score_normalization": self.score_normalization,
                "fusion_method": self.fusion_method,
                "enable_deduplication": self.enable_deduplication,
                "promote_diversity": self.promote_diversity
            },
            "available_libraries": {
                "numpy": NUMPY_AVAILABLE,
                "scipy": SCIPY_AVAILABLE
            }
        }
    
    async def benchmark_fusion_methods(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        methods: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Benchmark different fusion methods.
        
        Args:
            vector_results: Vector search results
            bm25_results: BM25 search results
            methods: List of methods to benchmark
            
        Returns:
            Performance comparison of fusion methods
        """
        
        methods = methods or ["rrf", "weighted", "rank_fusion"]
        benchmark_results = {}
        
        for method in methods:
            start_time = datetime.now()
            
            try:
                fused = await self.fuse_results(
                    vector_results=vector_results,
                    bm25_results=bm25_results,
                    fusion_method=method,
                    max_results=20
                )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                benchmark_results[method] = {
                    "processing_time": processing_time,
                    "result_count": len(fused),
                    "success": True,
                    "unique_documents": len(set(self._get_document_id(doc) for doc in fused)),
                    "avg_score": sum(doc.get("score", 0) for doc in fused) / len(fused) if fused else 0
                }
                
            except Exception as e:
                benchmark_results[method] = {
                    "processing_time": 0.0,
                    "result_count": 0,
                    "success": False,
                    "error": str(e)
                }
        
        return benchmark_results