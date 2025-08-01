"""
Document processor for preprocessing and structuring documents.

This module handles document preprocessing, chunking, metadata extraction,
and format conversion for optimal RAG system performance.
"""

import asyncio
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


class DocumentProcessor:
    """
    Document processor for RAG system preprocessing.
    
    Handles document chunking, metadata extraction, cleaning,
    and format conversion for optimal retrieval performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Chunking configuration
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.min_chunk_size = config.get("min_chunk_size", 100)
        self.max_chunk_size = config.get("max_chunk_size", 2048)
        
        # Content processing
        self.preserve_code_blocks = config.get("preserve_code_blocks", True)
        self.preserve_tables = config.get("preserve_tables", True)
        self.remove_html = config.get("remove_html", True)
        self.normalize_whitespace = config.get("normalize_whitespace", True)
        
        # Language detection
        self.detect_language = config.get("detect_language", False)
        
        # Metadata extraction
        self.extract_keywords = config.get("extract_keywords", True)
        self.extract_entities = config.get("extract_entities", False)
        
        # Initialize tfidf_vectorizer to None
        self.tfidf_vectorizer = None

        # Processing stats
        self.processing_stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "avg_chunks_per_document": 0.0,
            "avg_processing_time": 0.0,
            "documents_by_type": {},
            "errors": 0
        }
        
        # Pre-compiled regex patterns
        self.patterns = {
            "code_block": re.compile(r'```[\s\S]*?```|`[^`]+`', re.MULTILINE),
            "html_tag": re.compile(r'<[^>]+>'),
            "whitespace": re.compile(r'\s+'),
            "url": re.compile(r'https?://[^\s]+'),
            "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            "sentence_boundary": re.compile(r'(?<=[.!?])\s+(?=[A-Z])'),
            "paragraph_boundary": re.compile(r'\n\s*\n')
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the document processor."""
        
        try:
            # Initialize any language processing components
            if self.detect_language:
                try:
                    # Try to import and initialize language detection
                    from langdetect import detect
                    self._detect_language = detect
                except ImportError:
                    self.logger.warning("Language detection not available")
                    self.detect_language = False
            
            # Initialize keyword extraction
            if self.extract_keywords:
                await self._initialize_keyword_extraction()
            
            self.initialized = True
            self.logger.info("Document processor initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize document processor: {e}")
            return False
    
    async def _initialize_keyword_extraction(self) -> None:
        """Initialize keyword extraction components."""
        
        try:
            # Try to use TF-IDF for keyword extraction
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english',
                ngram_range=(1, 2)
            )
        except ImportError:
            self.logger.warning("sklearn not available, using simple keyword extraction")
            self.tfidf_vectorizer = None
    
    async def process_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_format: str = "text"
    ) -> Dict[str, Any]:
        """
        Process a single document.
        
        Args:
            content: Raw document content
            metadata: Optional document metadata
            source_format: Format of source document (text, html, markdown, etc.)
            
        Returns:
            Processed document with chunks and enhanced metadata
        """
        
        start_time = datetime.now()
        
        try:
            if not content or not content.strip():
                return self._create_empty_document(metadata)
            
            # Clean and normalize content
            cleaned_content = await self._clean_content(content, source_format)
            
            # Extract base metadata
            doc_metadata = await self._extract_metadata(cleaned_content, metadata)
            
            # Create document chunks
            chunks = await self._chunk_document(cleaned_content, doc_metadata)
            
            # Update processing stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_processing_stats(len(chunks), processing_time, source_format)
            
            # Create processed document
            processed_doc = {
                "content": cleaned_content,
                "chunks": chunks,
                "metadata": doc_metadata,
                "processing_info": {
                    "processed_at": datetime.now().isoformat(),
                    "processing_time": processing_time,
                    "source_format": source_format,
                    "chunk_count": len(chunks),
                    "content_length": len(cleaned_content)
                }
            }
            
            self.logger.debug(f"Processed document: {len(chunks)} chunks, {processing_time:.3f}s")
            
            return processed_doc
            
        except Exception as e:
            self.processing_stats["errors"] += 1
            self.logger.error(f"Document processing failed: {e}")
            return self._create_error_document(str(e), metadata)
    
    async def _clean_content(self, content: str, source_format: str) -> str:
        """Clean and normalize document content."""
        
        cleaned = content
        
        # Format-specific cleaning
        if source_format == "html":
            cleaned = await self._clean_html(cleaned)
        elif source_format == "markdown":
            cleaned = await self._clean_markdown(cleaned)
        
        # General cleaning
        if self.remove_html:
            cleaned = self.patterns["html_tag"].sub(" ", cleaned)
        
        if self.normalize_whitespace:
            cleaned = self.patterns["whitespace"].sub(" ", cleaned)
            cleaned = cleaned.strip()
        
        return cleaned
    
    async def _clean_html(self, html_content: str) -> str:
        """Clean HTML content."""
        
        if not BS4_AVAILABLE:
            # Fallback: simple HTML tag removal
            return self.patterns["html_tag"].sub(" ", html_content)
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            self.logger.warning(f"HTML cleaning failed: {e}")
            return self.patterns["html_tag"].sub(" ", html_content)
    
    async def _clean_markdown(self, markdown_content: str) -> str:
        """Clean Markdown content."""
        
        if not MARKDOWN_AVAILABLE:
            # Simple markdown cleaning
            cleaned = re.sub(r'#+\s*', '', markdown_content)  # Remove headers
            cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)  # Remove bold
            cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)  # Remove italic
            cleaned = re.sub(r'`(.*?)`', r'\1', cleaned)  # Remove inline code
            return cleaned
        
        try:
            # Convert to HTML then extract text
            html = markdown.markdown(markdown_content)
            return await self._clean_html(html)
            
        except Exception as e:
            self.logger.warning(f"Markdown cleaning failed: {e}")
            return markdown_content
    
    async def _extract_metadata(
        self,
        content: str,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract metadata from document content."""
        
        metadata = base_metadata.copy() if base_metadata else {}
        
        # Basic statistics
        metadata.update({
            "content_length": len(content),
            "word_count": len(content.split()),
            "line_count": content.count('\n') + 1,
            "char_count": len(content),
            "paragraph_count": len(self.patterns["paragraph_boundary"].split(content))
        })
        
        # Content hash for deduplication
        metadata["content_hash"] = hashlib.md5(content.encode()).hexdigest()
        
        # Language detection
        if self.detect_language and hasattr(self, '_detect_language'):
            try:
                metadata["language"] = self._detect_language(content)
            except:
                metadata["language"] = "unknown"
        
        # Extract URLs and emails
        urls = self.patterns["url"].findall(content)
        emails = self.patterns["email"].findall(content)
        
        if urls:
            metadata["urls"] = ", ".join(urls[:10])  # Convert list to string
            metadata["url_count"] = len(urls)
        
        if emails:
            metadata["emails"] = ", ".join(emails[:5])  # Convert list to string
            metadata["email_count"] = len(emails)
        
        # Code block detection
        code_blocks = self.patterns["code_block"].findall(content)
        if code_blocks:
            metadata["has_code"] = True
            metadata["code_block_count"] = len(code_blocks)
        
        # Extract keywords
        if self.extract_keywords:
            keywords = await self._extract_keywords(content)
            metadata["keywords"] = ", ".join(keywords) # Convert list to string
        
        return metadata
    
    async def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from content."""
        
        try:
            if self.tfidf_vectorizer:
                # Use TF-IDF
                try:
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform([content])
                    feature_names = self.tfidf_vectorizer.get_feature_names_out()
                    scores = tfidf_matrix.toarray()[0]
                    
                    # Get top keywords
                    keyword_scores = list(zip(feature_names, scores))
                    keyword_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    return [keyword for keyword, score in keyword_scores[:max_keywords] if score > 0]
                    
                except Exception as e:
                    self.logger.warning(f"TF-IDF keyword extraction failed: {e}")
            
            # Fallback: simple frequency-based extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            word_freq = {}
            
            # Common stopwords to filter
            stopwords = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was',
                'one', 'our', 'had', 'but', 'not', 'what', 'all', 'were', 'they', 'have',
                'this', 'that', 'will', 'from', 'they', 'know', 'want', 'been', 'good',
                'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like',
                'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well'
            }
            
            for word in words:
                if word not in stopwords and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency and return top keywords
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:max_keywords]]
            
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    async def _chunk_document(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Split document into chunks for processing."""
        
        try:
            # Use sentence-based chunking for better semantic coherence
            sentences = self.patterns["sentence_boundary"].split(content)
            
            chunks = []
            current_chunk = ""
            current_chunk_size = 0
            chunk_id = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_size = len(sentence)
                
                # Check if adding this sentence would exceed chunk size
                if (current_chunk_size + sentence_size > self.chunk_size and 
                    current_chunk_size >= self.min_chunk_size):
                    
                    # Create chunk
                    if current_chunk.strip():
                        chunk = await self._create_chunk(
                            current_chunk.strip(),
                            chunk_id,
                            metadata
                        )
                        chunks.append(chunk)
                        chunk_id += 1
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                        current_chunk = overlap_text + " " + sentence
                        current_chunk_size = len(current_chunk)
                    else:
                        current_chunk = sentence
                        current_chunk_size = sentence_size
                
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                    current_chunk_size += sentence_size
                
                # Check maximum chunk size
                if current_chunk_size > self.max_chunk_size:
                    # Force split at word boundary
                    words = current_chunk.split()
                    split_point = len(words) // 2
                    
                    first_part = " ".join(words[:split_point])
                    second_part = " ".join(words[split_point:])
                    
                    if first_part.strip():
                        chunk = await self._create_chunk(first_part.strip(), chunk_id, metadata)
                        chunks.append(chunk)
                        chunk_id += 1
                    
                    current_chunk = second_part
                    current_chunk_size = len(current_chunk)
            
            # Add final chunk
            if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
                chunk = await self._create_chunk(current_chunk.strip(), chunk_id, metadata)
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Document chunking failed: {e}")
            # Return single chunk with full content
            return [await self._create_chunk(content, 0, metadata)]
    
    async def _create_chunk(
        self,
        content: str,
        chunk_id: int,
        doc_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a document chunk with metadata."""
        
        chunk_metadata = doc_metadata.copy()
        chunk_metadata.update({
            "chunk_id": chunk_id,
            "chunk_size": len(content),
            "chunk_word_count": len(content.split()),
            "is_chunk": True
        })
        
        return {
            "content": content,
            "metadata": chunk_metadata
        }
    
    def _create_empty_document(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an empty document structure."""
        
        return {
            "content": "",
            "chunks": [],
            "metadata": metadata or {},
            "processing_info": {
                "processed_at": datetime.now().isoformat(),
                "processing_time": 0.0,
                "source_format": "empty",
                "chunk_count": 0,
                "content_length": 0
            }
        }
    
    def _create_error_document(
        self,
        error_message: str,
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create an error document structure."""
        
        return {
            "content": "",
            "chunks": [],
            "metadata": metadata or {},
            "processing_info": {
                "processed_at": datetime.now().isoformat(),
                "processing_time": 0.0,
                "source_format": "error",
                "chunk_count": 0,
                "content_length": 0,
                "error": error_message
            }
        }
    
    def _update_processing_stats(
        self,
        chunk_count: int,
        processing_time: float,
        source_format: str
    ) -> None:
        """Update processing statistics."""
        
        self.processing_stats["total_documents"] += 1
        self.processing_stats["total_chunks"] += chunk_count
        
        # Update average chunks per document
        total_docs = self.processing_stats["total_documents"]
        self.processing_stats["avg_chunks_per_document"] = (
            self.processing_stats["total_chunks"] / total_docs
        )
        
        # Update average processing time
        current_avg = self.processing_stats["avg_processing_time"]
        self.processing_stats["avg_processing_time"] = (
            (current_avg * (total_docs - 1) + processing_time) / total_docs
        )
        
        # Update documents by type
        if source_format in self.processing_stats["documents_by_type"]:
            self.processing_stats["documents_by_type"][source_format] += 1
        else:
            self.processing_stats["documents_by_type"][source_format] = 1
    
    async def process_batch(
        self,
        documents: List[Tuple[str, Optional[Dict[str, Any]], str]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of (content, metadata, source_format) tuples
            
        Returns:
            List of processed documents
        """
        
        try:
            self.logger.info(f"Processing batch of {len(documents)} documents")
            
            # Process documents concurrently
            tasks = [
                self.process_document(content, metadata, source_format)
                for content, metadata, source_format in documents
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log errors
            processed_documents = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to process document {i}: {result}")
                    self.processing_stats["errors"] += 1
                else:
                    processed_documents.append(result)
            
            self.logger.info(f"Successfully processed {len(processed_documents)}/{len(documents)} documents")
            
            return processed_documents
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return []
    
    async def extract_document_structure(self, content: str) -> Dict[str, Any]:
        """Extract document structure (headers, sections, etc.)."""
        
        try:
            structure = {
                "headers": [],
                "sections": [],
                "tables": [],
                "lists": [],
                "code_blocks": []
            }
            
            # Extract headers (markdown style)
            header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
            headers = header_pattern.findall(content)
            
            for level_str, title in headers:
                structure["headers"].append({
                    "level": len(level_str),
                    "title": title.strip(),
                    "position": content.find(f"{level_str} {title}")
                })
            
            # Extract code blocks
            code_blocks = self.patterns["code_block"].findall(content)
            for i, block in enumerate(code_blocks):
                structure["code_blocks"].append({
                    "id": i,
                    "content": block,
                    "position": content.find(block)
                })
            
            # Extract lists (simple detection)
            list_pattern = re.compile(r'^[\s]*[-*+]\s+(.+)$', re.MULTILINE)
            list_items = list_pattern.findall(content)
            
            if list_items:
                structure["lists"].append({
                    "type": "unordered",
                    "items": list_items
                })
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Structure extraction failed: {e}")
            return {}
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get document processing statistics."""
        
        return {
            **self.processing_stats,
            "initialized": self.initialized,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "available_libraries": {
                "pandas": PANDAS_AVAILABLE,
                "beautifulsoup4": BS4_AVAILABLE,
                "markdown": MARKDOWN_AVAILABLE
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown the document processor."""
        
        try:
            # Clear any cached data
            self.patterns.clear()
            
            self.initialized = False
            self.logger.info("Document processor shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during document processor shutdown: {e}")