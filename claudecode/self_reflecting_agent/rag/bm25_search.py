"""
BM25 search implementation for keyword-based document retrieval.

This module implements BM25 (Best Matching 25) ranking function for
keyword-based search, providing exact term matching and phrase search
capabilities.
"""

import asyncio
import logging
import math
import re
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class BM25Search:
    """
    BM25-based keyword search system.
    
    Implements the BM25 ranking function for efficient keyword-based
    document retrieval with preprocessing, tokenization, and scoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.k1 = config.get("k1", 1.2)  # Term frequency saturation parameter
        self.b = config.get("b", 0.75)   # Length normalization parameter
        self.use_stemming = config.get("use_stemming", True)
        self.remove_stopwords = config.get("remove_stopwords", True)
        self.min_term_length = config.get("min_term_length", 2)
        
        # Storage
        self.documents: List[Dict[str, Any]] = []
        self.processed_documents: List[List[str]] = []
        self.document_frequencies: Dict[str, int] = defaultdict(int)
        self.document_lengths: List[int] = []
        self.avg_document_length: float = 0.0
        
        # BM25 index
        self.bm25_index: Optional[Any] = None
        
        # Text processing components
        self.stemmer = None
        self.stopwords_set: Set[str] = set()
        
        # Phrase search support
        self.phrase_patterns: Dict[str, re.Pattern] = {}
        
        # Metrics
        self.metrics = {
            "total_documents": 0,
            "total_searches": 0,
            "avg_search_time": 0.0,
            "unique_terms": 0,
            "phrase_searches": 0
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the BM25 search system."""
        
        try:
            # Initialize text processing components
            await self._initialize_text_processing()
            
            # Download required NLTK data if available
            if NLTK_AVAILABLE:
                await self._ensure_nltk_data()
            
            self.initialized = True
            self.logger.info("BM25 search system initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BM25 search: {e}")
            return False
    
    async def _initialize_text_processing(self) -> None:
        """Initialize text processing components."""
        
        if NLTK_AVAILABLE:
            # Initialize stemmer
            if self.use_stemming:
                self.stemmer = PorterStemmer()
            
            # Initialize stopwords
            if self.remove_stopwords:
                try:
                    self.stopwords_set = set(stopwords.words('english'))
                except:
                    # Fallback to basic stopwords
                    self.stopwords_set = {
                        'a', 'an', 'and', 'the', 'is', 'in', 'it', 'you', 'that', 'he',
                        'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i',
                        'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by',
                        'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when'
                    }
        else:
            # Basic fallback stopwords
            if self.remove_stopwords:
                self.stopwords_set = {
                    'a', 'an', 'and', 'the', 'is', 'in', 'it', 'you', 'that', 'he',
                    'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i',
                    'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by'
                }
    
    async def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is downloaded."""
        
        try:
            import nltk
            
            # Download required data
            for resource in ['punkt', 'stopwords']:
                try:
                    nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
                except LookupError:
                    try:
                        nltk.download(resource, quiet=True)
                    except:
                        self.logger.warning(f"Failed to download NLTK {resource}")
                        
        except Exception as e:
            self.logger.warning(f"NLTK data setup failed: {e}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize and preprocess text."""
        
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Use NLTK tokenizer if available, otherwise use basic regex
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
            except:
                # Fallback to regex tokenization
                tokens = re.findall(r'\b\w+\b', text)
        else:
            # Basic regex tokenization
            tokens = re.findall(r'\b\w+\b', text)
        
        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Skip short tokens
            if len(token) < self.min_term_length:
                continue
            
            # Skip stopwords
            if self.remove_stopwords and token in self.stopwords_set:
                continue
            
            # Apply stemming
            if self.use_stemming and self.stemmer:
                token = self.stemmer.stem(token)
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the BM25 index."""
        
        if not self.initialized:
            return False
        
        try:
            self.logger.info(f"Adding {len(documents)} documents to BM25 index")
            
            # Process new documents
            new_processed_docs = []
            for doc in documents:
                content = doc.get("content", "")
                if content:
                    tokens = self._tokenize_text(content)
                    new_processed_docs.append(tokens)
                    
                    # Update document frequencies
                    unique_tokens = set(tokens)
                    for token in unique_tokens:
                        self.document_frequencies[token] += 1
                else:
                    new_processed_docs.append([])
            
            # Add to storage
            self.documents.extend(documents)
            self.processed_documents.extend(new_processed_docs)
            
            # Update document lengths
            new_lengths = [len(doc) for doc in new_processed_docs]
            self.document_lengths.extend(new_lengths)
            
            # Update average document length
            total_length = sum(self.document_lengths)
            self.avg_document_length = total_length / len(self.document_lengths) if self.document_lengths else 0.0
            
            # Rebuild BM25 index if library is available
            if BM25_AVAILABLE and self.processed_documents:
                self.bm25_index = BM25Okapi(self.processed_documents, k1=self.k1, b=self.b)
            
            # Update metrics
            self.metrics["total_documents"] = len(self.documents)
            self.metrics["unique_terms"] = len(self.document_frequencies)
            
            self.logger.info(f"Successfully added {len(documents)} documents to BM25 index")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to BM25 index: {e}")
            return False
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for documents using BM25 scoring."""
        
        if not self.initialized or not self.documents:
            return []
        
        start_time = datetime.now()
        
        try:
            self.metrics["total_searches"] += 1
            
            # Handle phrase searches (quoted text)
            if '"' in query:
                results = await self._phrase_search(query, max_results, filters)
                self.metrics["phrase_searches"] += 1
            else:
                results = await self._keyword_search(query, max_results, filters)
            
            # Update search time metric
            search_time = (datetime.now() - start_time).total_seconds()
            self._update_avg_search_time(search_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"BM25 search failed: {e}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        max_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based BM25 search."""
        
        # Tokenize query
        query_tokens = self._tokenize_text(query)
        
        if not query_tokens:
            return []
        
        # Use BM25Okapi if available
        if BM25_AVAILABLE and self.bm25_index:
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Create scored results
            scored_docs = []
            for i, score in enumerate(scores):
                if i < len(self.documents):
                    doc = self.documents[i].copy()
                    doc["score"] = float(score)
                    doc["rank"] = 0  # Will be set after sorting
                    scored_docs.append(doc)
        
        else:
            # Fallback BM25 implementation
            scored_docs = await self._manual_bm25_search(query_tokens)
        
        # Sort by score
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply filters
        if filters:
            scored_docs = [doc for doc in scored_docs if self._matches_filters(doc, filters)]
        
        # Limit results and set ranks
        results = scored_docs[:max_results]
        for i, doc in enumerate(results):
            doc["rank"] = i + 1
        
        return results
    
    async def _manual_bm25_search(self, query_tokens: List[str]) -> List[Dict[str, Any]]:
        """Manual BM25 implementation fallback."""
        
        scored_docs = []
        n_docs = len(self.documents)
        
        for doc_idx, (doc, doc_tokens) in enumerate(zip(self.documents, self.processed_documents)):
            if not doc_tokens:
                continue
            
            score = 0.0
            doc_length = self.document_lengths[doc_idx]
            
            # Calculate BM25 score for each query term
            for term in query_tokens:
                # Term frequency in document
                tf = doc_tokens.count(term)
                if tf == 0:
                    continue
                
                # Document frequency (how many documents contain this term)
                df = self.document_frequencies.get(term, 0)
                if df == 0:
                    continue
                
                # Inverse document frequency
                idf = math.log((n_docs - df + 0.5) / (df + 0.5))
                
                # BM25 score component
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_document_length)
                )
            
            doc_copy = doc.copy()
            doc_copy["score"] = score
            scored_docs.append(doc_copy)
        
        return scored_docs
    
    async def _phrase_search(
        self,
        query: str,
        max_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for exact phrases in documents."""
        
        # Extract phrases (text within quotes)
        phrases = re.findall(r'"([^"]*)"', query)
        non_phrase_query = re.sub(r'"[^"]*"', '', query).strip()
        
        results = []
        
        for doc_idx, doc in enumerate(self.documents):
            content = doc.get("content", "").lower()
            score = 0.0
            
            # Score phrase matches
            for phrase in phrases:
                phrase_lower = phrase.lower()
                if phrase_lower in content:
                    # Higher score for exact phrase matches
                    phrase_count = content.count(phrase_lower)
                    score += phrase_count * 10.0  # Boost phrase matches
            
            # Add keyword score for non-phrase terms
            if non_phrase_query:
                keyword_results = await self._keyword_search(non_phrase_query, 1)
                if keyword_results:
                    # Find this document in keyword results
                    for kw_result in keyword_results:
                        if kw_result.get("content") == doc.get("content"):
                            score += kw_result.get("score", 0.0)
                            break
            
            if score > 0:
                doc_copy = doc.copy()
                doc_copy["score"] = score
                results.append(doc_copy)
        
        # Sort and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply filters
        if filters:
            results = [doc for doc in results if self._matches_filters(doc, filters)]
        
        # Set ranks
        final_results = results[:max_results]
        for i, doc in enumerate(final_results):
            doc["rank"] = i + 1
        
        return final_results
    
    def _matches_filters(self, document: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document matches the given filters."""
        
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
    
    def _update_avg_search_time(self, search_time: float) -> None:
        """Update average search time metric."""
        
        current_avg = self.metrics["avg_search_time"]
        total_searches = self.metrics["total_searches"]
        
        self.metrics["avg_search_time"] = (
            (current_avg * (total_searches - 1) + search_time) / total_searches
        )
    
    async def get_term_frequencies(self, terms: List[str]) -> Dict[str, int]:
        """Get document frequencies for specific terms."""
        
        result = {}
        for term in terms:
            processed_term = self._tokenize_text(term)
            if processed_term:
                result[term] = self.document_frequencies.get(processed_term[0], 0)
        
        return result
    
    async def get_document_count(self) -> int:
        """Get total number of documents."""
        return len(self.documents)
    
    async def delete_documents(
        self,
        document_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Delete documents from the index."""
        
        try:
            if not document_ids and not filters:
                # Clear all documents
                self.documents.clear()
                self.processed_documents.clear()
                self.document_lengths.clear()
                self.document_frequencies.clear()
                self.avg_document_length = 0.0
                self.bm25_index = None
                
                self.metrics["total_documents"] = 0
                self.metrics["unique_terms"] = 0
                
                self.logger.info("All documents deleted from BM25 index")
                return True
            
            # For specific deletions, we'd need document IDs
            # This is a simplified implementation
            self.logger.warning("Selective document deletion not fully implemented for BM25")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents from BM25 index: {e}")
            return False
    
    async def optimize_index(self) -> bool:
        """Optimize the BM25 index."""
        
        try:
            if BM25_AVAILABLE and self.processed_documents:
                # Rebuild index with current parameters
                self.bm25_index = BM25Okapi(self.processed_documents, k1=self.k1, b=self.b)
                self.logger.info("BM25 index optimized")
                return True
            
            self.logger.info("BM25 index optimization completed (no-op)")
            return True
            
        except Exception as e:
            self.logger.error(f"BM25 index optimization failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get BM25 search metrics."""
        
        return {
            **self.metrics,
            "initialized": self.initialized,
            "bm25_library_available": BM25_AVAILABLE,
            "nltk_available": NLTK_AVAILABLE,
            "avg_doc_length": self.avg_document_length,
            "k1_parameter": self.k1,
            "b_parameter": self.b,
            "use_stemming": self.use_stemming,
            "remove_stopwords": self.remove_stopwords
        }
    
    async def get_similar_terms(self, term: str, max_terms: int = 5) -> List[str]:
        """Get similar terms based on document co-occurrence."""
        
        try:
            processed_term = self._tokenize_text(term)
            if not processed_term:
                return []
            
            target_term = processed_term[0]
            
            # Find documents containing the target term
            docs_with_term = []
            for i, doc_tokens in enumerate(self.processed_documents):
                if target_term in doc_tokens:
                    docs_with_term.append(i)
            
            if not docs_with_term:
                return []
            
            # Count co-occurring terms
            cooccurrence_counts = Counter()
            for doc_idx in docs_with_term:
                doc_tokens = self.processed_documents[doc_idx]
                for token in set(doc_tokens):
                    if token != target_term:
                        cooccurrence_counts[token] += 1
            
            # Return most frequent co-occurring terms
            similar_terms = [
                term for term, count in cooccurrence_counts.most_common(max_terms)
            ]
            
            return similar_terms
            
        except Exception as e:
            self.logger.error(f"Failed to get similar terms: {e}")
            return []
    
    async def shutdown(self) -> None:
        """Shutdown the BM25 search system."""
        
        try:
            # Clear data structures
            self.documents.clear()
            self.processed_documents.clear()
            self.document_lengths.clear()
            self.document_frequencies.clear()
            self.phrase_patterns.clear()
            
            self.bm25_index = None
            self.initialized = False
            
            self.logger.info("BM25 search system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during BM25 shutdown: {e}")