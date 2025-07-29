"""
Context optimizer for intelligent context window compression.

This module provides algorithms for optimizing context windows through
summarization, compression, and intelligent pruning while preserving
semantic coherence and critical information.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
import re

from .context_types import (
    ContextType, ContextPriority, ContextEntry,
    ContextWindow, ContextOptimizationResult
)

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class ContextOptimizer:
    """
    Context window optimizer using various compression techniques.
    
    Implements intelligent context compression through summarization,
    deduplication, pruning, and semantic compression while preserving
    critical information and maintaining coherence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Optimization strategies
        self.enable_summarization = config.get("enable_summarization", True)
        self.enable_deduplication = config.get("enable_deduplication", True)
        self.enable_pruning = config.get("enable_pruning", True)
        self.enable_compression = config.get("enable_compression", True)
        
        # Summarization configuration
        self.summarization_model = config.get("summarization_model", "facebook/bart-large-cnn")
        self.max_summary_ratio = config.get("max_summary_ratio", 0.3)  # Summary should be at most 30% of original
        self.min_content_length = config.get("min_content_length", 100)  # Don't summarize short content
        
        # OpenAI configuration for advanced summarization
        self.use_openai_summarization = config.get("use_openai_summarization", False)
        self.openai_model = config.get("openai_model", "gpt-3.5-turbo")
        
        # Deduplication settings
        self.similarity_threshold = config.get("similarity_threshold", 0.8)
        self.content_overlap_threshold = config.get("content_overlap_threshold", 0.7)
        
        # Pruning strategy
        self.pruning_strategy = config.get("pruning_strategy", "least_recently_used")  # lru, importance, age
        self.preserve_critical_types = config.get("preserve_critical_types", [
            ContextType.SYSTEM.value,
            ContextType.TASK.value
        ])
        
        # Components
        self.summarizer = None
        
        # Performance tracking
        self.optimization_stats = {
            "total_optimizations": 0,
            "total_tokens_saved": 0,
            "summarizations_performed": 0,
            "entries_deduplicated": 0,
            "entries_pruned": 0,
            "avg_optimization_time": 0.0
        }
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the context optimizer."""
        
        try:
            # Initialize summarization model if available
            if self.enable_summarization and TRANSFORMERS_AVAILABLE:
                try:
                    self.summarizer = pipeline("summarization", model=self.summarization_model)
                    self.logger.info(f"Loaded summarization model: {self.summarization_model}")
                except Exception as e:
                    self.logger.warning(f"Failed to load summarization model: {e}")
                    self.summarizer = None
            
            self.initialized = True
            self.logger.info("Context optimizer initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize context optimizer: {e}")
            return False
    
    async def optimize_window(self, context_window: ContextWindow) -> ContextOptimizationResult:
        """
        Optimize a context window to reduce token usage.
        
        Args:
            context_window: Context window to optimize
            
        Returns:
            Optimization result with statistics and actions taken
        """
        
        start_time = datetime.now()
        
        try:
            # Capture initial state
            tokens_before = context_window.current_tokens
            entries_before = len(context_window.entries)
            
            result = ContextOptimizationResult(
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                entries_before=entries_before,
                entries_after=entries_before
            )
            
            # Apply optimization strategies in order
            
            # 1. Remove expired entries
            await self._remove_expired_entries(context_window, result)
            
            # 2. Deduplicate similar entries
            if self.enable_deduplication:
                await self._deduplicate_entries(context_window, result)
            
            # 3. Summarize content where appropriate
            if self.enable_summarization:
                await self._summarize_entries(context_window, result)
            
            # 4. Compress redundant information
            if self.enable_compression:
                await self._compress_entries(context_window, result)
            
            # 5. Prune low-priority entries if still needed
            if self.enable_pruning and context_window.is_over_limit():
                await self._prune_entries(context_window, result)
            
            # Update final state
            context_window.update_statistics()
            result.tokens_after = context_window.current_tokens
            result.entries_after = len(context_window.entries)
            result.compression_ratio = tokens_before / result.tokens_after if result.tokens_after > 0 else 1.0
            
            # Update statistics
            optimization_time = (datetime.now() - start_time).total_seconds()
            result.optimization_time = optimization_time
            self._update_stats(result)
            
            self.logger.debug(f"Context optimization completed: saved {result.tokens_saved()} tokens")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Context optimization failed: {e}")
            result.success = False
            result.error_message = str(e)
            return result
    
    async def _remove_expired_entries(
        self,
        context_window: ContextWindow,
        result: ContextOptimizationResult
    ) -> None:
        """Remove expired context entries."""
        
        expired_entries = []
        for entry in context_window.entries:
            if entry.is_expired() and not entry.is_persistent:
                expired_entries.append(entry)
        
        for entry in expired_entries:
            context_window.entries.remove(entry)
            result.entries_removed.append(entry.id)
        
        if expired_entries:
            self.logger.debug(f"Removed {len(expired_entries)} expired entries")
    
    async def _deduplicate_entries(
        self,
        context_window: ContextWindow,
        result: ContextOptimizationResult
    ) -> None:
        """Remove duplicate or highly similar entries."""
        
        try:
            entries_to_remove = []
            processed_entries = set()
            
            for entry in context_window.entries:
                if entry.id in processed_entries:
                    continue
                
                # Find similar entries
                similar_entries = []
                for other_entry in context_window.entries:
                    if (other_entry.id != entry.id and 
                        other_entry.id not in processed_entries and
                        self._are_entries_similar(entry, other_entry)):
                        similar_entries.append(other_entry)
                
                if similar_entries:
                    # Keep the most important/recent entry
                    all_entries = [entry] + similar_entries
                    best_entry = max(all_entries, key=lambda e: (
                        e.importance_score,
                        e.priority.value,
                        e.access_count,
                        e.created_at.timestamp()
                    ))
                    
                    # Mark others for removal
                    for e in all_entries:
                        if e.id != best_entry.id:
                            entries_to_remove.append(e)
                        processed_entries.add(e.id)
            
            # Remove duplicates
            for entry in entries_to_remove:
                if entry in context_window.entries:
                    context_window.entries.remove(entry)
                    result.entries_removed.append(entry.id)
                    self.optimization_stats["entries_deduplicated"] += 1
            
            if entries_to_remove:
                self.logger.debug(f"Deduplicated {len(entries_to_remove)} similar entries")
                
        except Exception as e:
            self.logger.error(f"Deduplication failed: {e}")
    
    def _are_entries_similar(self, entry1: ContextEntry, entry2: ContextEntry) -> bool:
        """Check if two entries are similar enough to deduplicate."""
        
        # Must be same type
        if entry1.context_type != entry2.context_type:
            return False
        
        # Don't deduplicate critical entries
        if (entry1.context_type.value in self.preserve_critical_types or
            entry2.context_type.value in self.preserve_critical_types):
            return False
        
        # Check content similarity
        content1_words = set(entry1.content.lower().split())
        content2_words = set(entry2.content.lower().split())
        
        if not content1_words or not content2_words:
            return False
        
        # Jaccard similarity
        intersection = len(content1_words.intersection(content2_words))
        union = len(content1_words.union(content2_words))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        
        return jaccard_similarity > self.similarity_threshold
    
    async def _summarize_entries(
        self,
        context_window: ContextWindow,
        result: ContextOptimizationResult
    ) -> None:
        """Summarize appropriate entries to reduce token usage."""
        
        try:
            for entry in context_window.entries:
                # Skip if already summarized or can't be summarized
                if entry.summary or not entry.can_summarize:
                    continue
                
                # Skip short content
                if len(entry.content) < self.min_content_length:
                    continue
                
                # Skip critical types that shouldn't be summarized
                if entry.context_type.value in self.preserve_critical_types:
                    continue
                
                # Generate summary
                summary = await self._generate_summary(entry.content, entry.context_type)
                
                if summary and len(summary) < len(entry.content) * self.max_summary_ratio:
                    entry.summary = summary
                    entry.compression_ratio = len(entry.content) / len(summary)
                    result.entries_summarized.append(entry.id)
                    self.optimization_stats["summarizations_performed"] += 1
                    
                    self.logger.debug(f"Summarized entry {entry.id}: {len(entry.content)} -> {len(summary)} chars")
            
        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
    
    async def _generate_summary(self, content: str, context_type: ContextType) -> Optional[str]:
        """Generate a summary for content."""
        
        try:
            # Use OpenAI for summarization if configured and available
            if self.use_openai_summarization and OPENAI_AVAILABLE:
                return await self._generate_openai_summary(content, context_type)
            
            # Use local transformer model
            elif self.summarizer:
                return await self._generate_transformer_summary(content)
            
            # Fallback to extractive summarization
            else:
                return await self._generate_extractive_summary(content)
                
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return None
    
    async def _generate_openai_summary(self, content: str, context_type: ContextType) -> Optional[str]:
        """Generate summary using OpenAI."""
        
        try:
            # Create context-appropriate prompt
            if context_type == ContextType.CONVERSATION:
                prompt = f"Summarize this conversation while preserving key information and decisions:\n\n{content}"
            elif context_type == ContextType.CODE:
                prompt = f"Summarize this code while preserving key functionality and logic:\n\n{content}"
            elif context_type == ContextType.DOCUMENTATION:
                prompt = f"Create a concise summary of this documentation:\n\n{content}"
            else:
                prompt = f"Create a concise summary of this content:\n\n{content}"
            
            # Make API call (async simulation - actual implementation would use aiohttp)
            # This is a placeholder for the actual OpenAI API call
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._call_openai_api(prompt)
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"OpenAI summarization failed: {e}")
            return None
    
    def _call_openai_api(self, prompt: str) -> str:
        """Placeholder for OpenAI API call."""
        # In real implementation, this would make an actual API call
        # For now, return a simple extractive summary
        sentences = prompt.split('. ')
        return '. '.join(sentences[:2]) + '.' if len(sentences) > 2 else prompt
    
    async def _generate_transformer_summary(self, content: str) -> Optional[str]:
        """Generate summary using transformer model."""
        
        try:
            # Limit input length for transformer
            max_input_length = 1024
            if len(content) > max_input_length:
                content = content[:max_input_length]
            
            # Generate summary
            summary_result = self.summarizer(
                content,
                max_length=min(150, len(content.split()) // 2),
                min_length=30,
                do_sample=False
            )
            
            return summary_result[0]['summary_text']
            
        except Exception as e:
            self.logger.error(f"Transformer summarization failed: {e}")
            return None
    
    async def _generate_extractive_summary(self, content: str) -> Optional[str]:
        """Generate extractive summary using simple heuristics."""
        
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 2:
                return content
            
            # Score sentences by position and word frequency
            word_freq = {}
            for sentence in sentences:
                words = sentence.lower().split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score sentences
            sentence_scores = []
            for i, sentence in enumerate(sentences):
                score = 0
                words = sentence.lower().split()
                
                # Position score (first and last sentences are important)
                if i == 0 or i == len(sentences) - 1:
                    score += 0.3
                
                # Word frequency score
                for word in words:
                    if word in word_freq:
                        score += word_freq[word] / len(words)
                
                sentence_scores.append((sentence, score))
            
            # Select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            top_sentences = sentence_scores[:max(1, len(sentences) // 3)]
            
            # Maintain original order
            selected_sentences = []
            for sentence in sentences:
                if any(sentence == s[0] for s in top_sentences):
                    selected_sentences.append(sentence)
            
            return '. '.join(selected_sentences) + '.'
            
        except Exception as e:
            self.logger.error(f"Extractive summarization failed: {e}")
            return content
    
    async def _compress_entries(
        self,
        context_window: ContextWindow,
        result: ContextOptimizationResult
    ) -> None:
        """Compress entries by removing redundant information."""
        
        try:
            for entry in context_window.entries:
                # Skip already compressed entries
                if entry.compression_ratio < 1.0:
                    continue
                
                # Apply content-specific compression
                original_content = entry.content
                compressed_content = await self._compress_content(entry.content, entry.context_type)
                
                if len(compressed_content) < len(original_content):
                    entry.content = compressed_content
                    entry.content_length = len(compressed_content)
                    entry.estimated_tokens = max(1, len(compressed_content) // 4)
                    entry.compression_ratio = len(original_content) / len(compressed_content)
                    result.entries_compressed.append(entry.id)
                    
                    self.logger.debug(f"Compressed entry {entry.id}: {len(original_content)} -> {len(compressed_content)} chars")
            
        except Exception as e:
            self.logger.error(f"Content compression failed: {e}")
    
    async def _compress_content(self, content: str, context_type: ContextType) -> str:
        """Apply content-specific compression techniques."""
        
        try:
            compressed = content
            
            # Remove excessive whitespace
            compressed = re.sub(r'\s+', ' ', compressed)
            compressed = re.sub(r'\n\s*\n\s*\n+', '\n\n', compressed)
            
            # For code, remove comments and empty lines
            if context_type == ContextType.CODE:
                lines = compressed.split('\n')
                filtered_lines = []
                for line in lines:
                    stripped = line.strip()
                    # Keep non-empty lines and non-comment lines
                    if stripped and not stripped.startswith(('#', '//', '/*', '*')):
                        filtered_lines.append(line)
                compressed = '\n'.join(filtered_lines)
            
            # For conversation, remove filler words and repetitions
            elif context_type == ContextType.CONVERSATION:
                # Remove common filler words
                filler_words = ['um', 'uh', 'like', 'you know', 'I mean', 'sort of', 'kind of']
                for filler in filler_words:
                    compressed = re.sub(f'\\b{filler}\\b', '', compressed, flags=re.IGNORECASE)
                
                # Remove repeated phrases
                compressed = re.sub(r'\b(.+?)\b\s+\1\b', r'\1', compressed)
            
            # General cleanup
            compressed = compressed.strip()
            
            return compressed if compressed else content
            
        except Exception as e:
            self.logger.error(f"Content compression failed: {e}")
            return content
    
    async def _prune_entries(
        self,
        context_window: ContextWindow,
        result: ContextOptimizationResult
    ) -> None:
        """Prune low-priority entries if context is still over limit."""
        
        try:
            if not context_window.is_over_limit():
                return
            
            # Don't prune critical types
            prunable_entries = [
                entry for entry in context_window.entries
                if (entry.context_type.value not in self.preserve_critical_types and
                    not entry.is_persistent)
            ]
            
            if not prunable_entries:
                return
            
            # Sort entries by pruning strategy
            if self.pruning_strategy == "least_recently_used":
                prunable_entries.sort(key=lambda e: e.last_accessed or e.created_at)
            elif self.pruning_strategy == "importance":
                prunable_entries.sort(key=lambda e: e.importance_score)
            elif self.pruning_strategy == "age":
                prunable_entries.sort(key=lambda e: e.created_at, reverse=True)
            else:  # Default to LRU
                prunable_entries.sort(key=lambda e: e.last_accessed or e.created_at)
            
            # Remove entries until we're under limit
            tokens_to_remove = context_window.current_tokens - context_window.max_tokens
            tokens_removed = 0
            
            for entry in prunable_entries:
                if tokens_removed >= tokens_to_remove:
                    break
                
                context_window.entries.remove(entry)
                result.entries_removed.append(entry.id)
                tokens_removed += entry.estimated_tokens
                self.optimization_stats["entries_pruned"] += 1
            
            if tokens_removed > 0:
                self.logger.debug(f"Pruned {len(result.entries_removed)} entries, saved {tokens_removed} tokens")
            
        except Exception as e:
            self.logger.error(f"Entry pruning failed: {e}")
    
    def _update_stats(self, result: ContextOptimizationResult) -> None:
        """Update optimization statistics."""
        
        self.optimization_stats["total_optimizations"] += 1
        self.optimization_stats["total_tokens_saved"] += result.tokens_saved()
        
        # Update average optimization time
        current_avg = self.optimization_stats["avg_optimization_time"]
        total_opts = self.optimization_stats["total_optimizations"]
        
        self.optimization_stats["avg_optimization_time"] = (
            (current_avg * (total_opts - 1) + result.optimization_time) / total_opts
        )
    
    async def analyze_context_efficiency(self, context_window: ContextWindow) -> Dict[str, Any]:
        """Analyze context window efficiency and provide recommendations."""
        
        try:
            analysis = {
                "efficiency_score": 0.0,
                "recommendations": [],
                "token_distribution": {},
                "redundancy_analysis": {},
                "optimization_potential": {}
            }
            
            # Calculate efficiency score
            critical_tokens = sum(
                entry.estimated_tokens for entry in context_window.entries
                if entry.priority == ContextPriority.CRITICAL
            )
            
            high_tokens = sum(
                entry.estimated_tokens for entry in context_window.entries
                if entry.priority == ContextPriority.HIGH
            )
            
            total_tokens = context_window.current_tokens
            if total_tokens > 0:
                analysis["efficiency_score"] = (critical_tokens + high_tokens * 0.8) / total_tokens
            
            # Token distribution by type and priority
            type_distribution = {}
            priority_distribution = {}
            
            for entry in context_window.entries:
                # By type
                context_type = entry.context_type.value
                type_distribution[context_type] = (
                    type_distribution.get(context_type, 0) + entry.estimated_tokens
                )
                
                # By priority
                priority = entry.priority.value
                priority_distribution[priority] = (
                    priority_distribution.get(priority, 0) + entry.estimated_tokens
                )
            
            analysis["token_distribution"] = {
                "by_type": type_distribution,
                "by_priority": priority_distribution
            }
            
            # Identify optimization opportunities
            summarizable_tokens = sum(
                entry.estimated_tokens for entry in context_window.entries
                if entry.can_summarize and not entry.summary and len(entry.content) > self.min_content_length
            )
            
            redundant_entries = 0
            for entry in context_window.entries:
                similar_count = sum(
                    1 for other in context_window.entries
                    if other.id != entry.id and self._are_entries_similar(entry, other)
                )
                if similar_count > 0:
                    redundant_entries += 1
            
            analysis["optimization_potential"] = {
                "summarizable_tokens": summarizable_tokens,
                "redundant_entries": redundant_entries,
                "estimated_savings": summarizable_tokens * (1 - self.max_summary_ratio)
            }
            
            # Generate recommendations
            if analysis["efficiency_score"] < 0.6:
                analysis["recommendations"].append("Low efficiency detected. Consider removing low-priority entries.")
            
            if summarizable_tokens > total_tokens * 0.3:
                analysis["recommendations"].append("Large amount of content can be summarized to save tokens.")
            
            if redundant_entries > 0:
                analysis["recommendations"].append(f"Found {redundant_entries} potentially redundant entries that could be deduplicated.")
            
            if context_window.utilization_ratio > 0.9:
                analysis["recommendations"].append("Context window is nearly full. Consider optimization.")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Context efficiency analysis failed: {e}")
            return {}
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        
        return {
            **self.optimization_stats,
            "initialized": self.initialized,
            "config": {
                "enable_summarization": self.enable_summarization,
                "enable_deduplication": self.enable_deduplication,
                "enable_pruning": self.enable_pruning,
                "enable_compression": self.enable_compression,
                "summarization_model": self.summarization_model,
                "use_openai_summarization": self.use_openai_summarization
            },
            "available_models": {
                "transformers": TRANSFORMERS_AVAILABLE,
                "openai": OPENAI_AVAILABLE
            }
        }
    
    async def shutdown(self) -> None:
        """Shutdown the context optimizer."""
        
        try:
            # Clear model references
            self.summarizer = None
            
            self.initialized = False
            self.logger.info("Context optimizer shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during context optimizer shutdown: {e}")