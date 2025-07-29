"""
Intelligent context manager for preventing context poisoning.

This module provides dynamic context window management, preventing
information overload while maintaining semantic coherence and
ensuring critical information is preserved.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .context_types import (
    ContextType, ContextPriority, ContextEntry, 
    ContextWindow, ContextOptimizationResult
)
from .context_optimizer import ContextOptimizer


class ContextManager:
    """
    Intelligent context manager for dynamic window management.
    
    Prevents context poisoning by intelligently managing context window
    utilization, prioritizing important information, and optimizing
    content for maximum effectiveness.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Context window configuration
        self.max_tokens = config.get("max_tokens", 4000)
        self.target_utilization = config.get("target_utilization", 0.8)  # 80% target
        self.optimization_threshold = config.get("optimization_threshold", 0.9)  # 90% triggers optimization
        
        # Context management settings
        self.enable_summarization = config.get("enable_summarization", True)
        self.enable_compression = config.get("enable_compression", True)
        self.enable_expiration = config.get("enable_expiration", True)
        
        # Priority-based limits
        self.priority_limits = config.get("priority_limits", {
            ContextPriority.CRITICAL.value: 0.4,  # 40% max for critical
            ContextPriority.HIGH.value: 0.3,      # 30% max for high
            ContextPriority.MEDIUM.value: 0.2,    # 20% max for medium
            ContextPriority.LOW.value: 0.1,       # 10% max for low/minimal
        })
        
        # Type-based settings
        self.type_config = config.get("type_config", {
            ContextType.SYSTEM.value: {
                "default_priority": ContextPriority.CRITICAL.value,
                "can_expire": False,
                "can_summarize": False
            },
            ContextType.TASK.value: {
                "default_priority": ContextPriority.CRITICAL.value,
                "can_expire": False,
                "can_summarize": True
            },
            ContextType.CONVERSATION.value: {
                "default_priority": ContextPriority.HIGH.value,
                "can_expire": True,
                "expire_hours": 24,
                "can_summarize": True
            },
            ContextType.CODE.value: {
                "default_priority": ContextPriority.HIGH.value,
                "can_expire": True,
                "expire_hours": 12,
                "can_summarize": False  # Code shouldn't be summarized
            },
            ContextType.MEMORY.value: {
                "default_priority": ContextPriority.MEDIUM.value,
                "can_expire": True,
                "expire_hours": 6,
                "can_summarize": True
            }
        })
        
        # Core components
        self.context_window = ContextWindow(max_tokens=self.max_tokens)
        self.optimizer = ContextOptimizer(config.get("optimizer", {}))
        
        # Context registry
        self.entries_by_id: Dict[str, ContextEntry] = {}
        self.entries_by_type: Dict[ContextType, List[ContextEntry]] = defaultdict(list)
        
        # Performance tracking
        self.stats = {
            "total_entries_added": 0,
            "total_entries_removed": 0,
            "optimization_count": 0,
            "total_tokens_processed": 0,
            "avg_utilization": 0.0,
            "context_poisoning_events": 0
        }
        
        # Background tasks
        self._cleanup_task = None
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the context manager."""
        
        try:
            # Initialize optimizer
            await self.optimizer.initialize()
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.initialized = True
            self.logger.info("Context manager initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize context manager: {e}")
            return False
    
    async def add_context(
        self,
        content: str,
        context_type: ContextType,
        priority: Optional[ContextPriority] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        expires_in_hours: Optional[float] = None,
        is_persistent: bool = False
    ) -> str:
        """
        Add content to the context window.
        
        Args:
            content: Context content to add
            context_type: Type of context
            priority: Priority level (auto-determined if None)
            metadata: Additional metadata
            tags: Context tags
            expires_in_hours: Hours until expiration
            is_persistent: Whether to keep through cleanups
            
        Returns:
            Context entry ID
        """
        
        try:
            # Auto-determine priority if not specified
            if priority is None:
                priority = self._get_default_priority(context_type)
            
            # Calculate expiration
            expires_at = None
            if expires_in_hours is not None:
                expires_at = datetime.now() + timedelta(hours=expires_in_hours)
            elif self.enable_expiration:
                default_hours = self._get_default_expiration_hours(context_type)
                if default_hours:
                    expires_at = datetime.now() + timedelta(hours=default_hours)
            
            # Create context entry
            entry_id = str(uuid.uuid4())
            entry = ContextEntry(
                id=entry_id,
                content=content,
                context_type=context_type,
                priority=priority,
                created_at=datetime.now(),
                metadata=metadata or {},
                tags=tags or [],
                expires_at=expires_at,
                is_persistent=is_persistent,
                can_summarize=self._can_summarize(context_type)
            )
            
            # Check if adding would exceed limits
            if not await self._can_add_entry(entry):
                # Try optimization first
                optimization_result = await self.optimize_context()
                if not optimization_result.success or not await self._can_add_entry(entry):
                    self.logger.warning(f"Cannot add context entry: {context_type.value} would exceed limits")
                    return ""
            
            # Add entry
            self.entries_by_id[entry_id] = entry
            self.entries_by_type[context_type].append(entry)
            self.context_window.entries.append(entry)
            self.context_window.update_statistics()
            
            # Update stats
            self.stats["total_entries_added"] += 1
            self.stats["total_tokens_processed"] += entry.estimated_tokens
            
            self.logger.debug(f"Added context entry: {entry_id} ({context_type.value}, {entry.estimated_tokens} tokens)")
            
            # Trigger optimization if needed
            if self.context_window.utilization_ratio > self.optimization_threshold:
                await self.optimize_context()
            
            return entry_id
            
        except Exception as e:
            self.logger.error(f"Failed to add context: {e}")
            return ""
    
    async def get_context(self, entry_id: str, update_access: bool = True) -> Optional[ContextEntry]:
        """Get a context entry by ID."""
        
        entry = self.entries_by_id.get(entry_id)
        if entry and update_access:
            entry.update_access()
        
        return entry
    
    async def remove_context(self, entry_id: str) -> bool:
        """Remove a context entry."""
        
        try:
            entry = self.entries_by_id.get(entry_id)
            if not entry:
                return False
            
            # Remove from all collections
            del self.entries_by_id[entry_id]
            self.entries_by_type[entry.context_type].remove(entry)
            self.context_window.entries.remove(entry)
            self.context_window.update_statistics()
            
            # Update stats
            self.stats["total_entries_removed"] += 1
            
            self.logger.debug(f"Removed context entry: {entry_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove context {entry_id}: {e}")
            return False
    
    async def get_active_context(
        self,
        context_types: Optional[List[ContextType]] = None,
        max_tokens: Optional[int] = None,
        include_summaries: bool = False
    ) -> str:
        """
        Get the current active context as a formatted string.
        
        Args:
            context_types: Filter by context types
            max_tokens: Maximum tokens to include
            include_summaries: Whether to use summaries when available
            
        Returns:
            Formatted context string
        """
        
        try:
            # Get relevant entries
            entries = self.context_window.entries.copy()
            
            # Filter by type if specified
            if context_types:
                entries = [e for e in entries if e.context_type in context_types]
            
            # Sort by priority and recency
            entries.sort(key=lambda e: (e.priority.value, e.last_accessed or e.created_at), reverse=True)
            
            # Build context string
            context_parts = []
            total_tokens = 0
            max_tokens = max_tokens or self.max_tokens
            
            for entry in entries:
                content = entry.get_effective_content(use_summary=include_summaries)
                entry_tokens = entry.get_effective_tokens(use_summary=include_summaries)
                
                if total_tokens + entry_tokens > max_tokens:
                    break
                
                # Format context entry
                section_header = f"[{entry.context_type.value.upper()}]"
                if entry.metadata.get("title"):
                    section_header += f" {entry.metadata['title']}"
                
                context_parts.append(f"{section_header}\n{content}\n")
                total_tokens += entry_tokens
                
                # Update access
                entry.update_access()
            
            return "\n".join(context_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to get active context: {e}")
            return ""
    
    async def optimize_context(self) -> ContextOptimizationResult:
        """
        Optimize the context window to reduce token usage.
        
        Returns:
            Optimization result with statistics
        """
        
        try:
            start_time = datetime.now()
            
            # Capture before state
            tokens_before = self.context_window.current_tokens
            entries_before = len(self.context_window.entries)
            
            # Run optimization
            result = await self.optimizer.optimize_window(self.context_window)
            
            # Update internal state based on optimization
            self._apply_optimization_result(result)
            
            # Update window statistics
            self.context_window.update_statistics()
            self.context_window.last_optimized = datetime.now()
            self.context_window.optimization_count += 1
            
            # Update stats
            self.stats["optimization_count"] += 1
            optimization_time = (datetime.now() - start_time).total_seconds()
            result.optimization_time = optimization_time
            
            self.logger.info(f"Context optimization completed: {result.tokens_saved()} tokens saved ({result.compression_percentage():.1f}%)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Context optimization failed: {e}")
            return ContextOptimizationResult(
                tokens_before=self.context_window.current_tokens,
                tokens_after=self.context_window.current_tokens,
                entries_before=len(self.context_window.entries),
                entries_after=len(self.context_window.entries),
                success=False,
                error_message=str(e)
            )
    
    def _apply_optimization_result(self, result: ContextOptimizationResult) -> None:
        """Apply optimization changes to internal state."""
        
        # Remove entries that were removed
        for entry_id in result.entries_removed:
            if entry_id in self.entries_by_id:
                entry = self.entries_by_id[entry_id]
                del self.entries_by_id[entry_id]
                self.entries_by_type[entry.context_type].remove(entry)
                if entry in self.context_window.entries:
                    self.context_window.entries.remove(entry)
        
        # Update summarized entries
        for entry_id in result.entries_summarized:
            if entry_id in self.entries_by_id:
                entry = self.entries_by_id[entry_id]
                # The optimizer should have updated the entry's summary
                entry.compression_ratio = len(entry.content) / len(entry.summary) if entry.summary else 1.0
    
    async def search_context(
        self,
        query: str,
        context_types: Optional[List[ContextType]] = None,
        max_results: int = 10
    ) -> List[ContextEntry]:
        """
        Search for relevant context entries.
        
        Args:
            query: Search query
            context_types: Filter by context types
            max_results: Maximum results to return
            
        Returns:
            List of matching context entries
        """
        
        try:
            query_lower = query.lower()
            matches = []
            
            for entry in self.context_window.entries:
                # Filter by type if specified
                if context_types and entry.context_type not in context_types:
                    continue
                
                # Simple text matching
                if query_lower in entry.content.lower():
                    entry.update_access()
                    matches.append(entry)
                
                # Check tags
                elif any(query_lower in tag.lower() for tag in entry.tags):
                    entry.update_access()
                    matches.append(entry)
                
                # Check metadata
                elif any(query_lower in str(v).lower() for v in entry.metadata.values()):
                    entry.update_access()
                    matches.append(entry)
            
            # Sort by relevance (importance score + access count + recency)
            matches.sort(key=lambda e: (
                e.importance_score,
                e.access_count,
                e.created_at.timestamp()
            ), reverse=True)
            
            return matches[:max_results]
            
        except Exception as e:
            self.logger.error(f"Context search failed: {e}")
            return []
    
    async def update_context(
        self,
        entry_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        priority: Optional[ContextPriority] = None
    ) -> bool:
        """Update an existing context entry."""
        
        try:
            entry = self.entries_by_id.get(entry_id)
            if not entry:
                return False
            
            # Update fields
            if content is not None:
                entry.content = content
                entry.content_length = len(content)
                entry.estimated_tokens = max(1, len(content) // 4)
            
            if metadata is not None:
                entry.metadata.update(metadata)
            
            if tags is not None:
                entry.tags = tags
            
            if priority is not None:
                entry.priority = priority
            
            # Update window statistics
            self.context_window.update_statistics()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update context {entry_id}: {e}")
            return False
    
    async def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context state."""
        
        try:
            # Calculate statistics
            total_entries = len(self.context_window.entries)
            
            # Entries by type
            type_counts = {
                context_type.value: len(entries)
                for context_type, entries in self.entries_by_type.items()
            }
            
            # Entries by priority
            priority_counts = defaultdict(int)
            priority_tokens = defaultdict(int)
            
            for entry in self.context_window.entries:
                priority_counts[entry.priority.value] += 1
                priority_tokens[entry.priority.value] += entry.estimated_tokens
            
            # Age distribution
            now = datetime.now()
            age_buckets = {"<1h": 0, "1-6h": 0, "6-24h": 0, ">24h": 0}
            
            for entry in self.context_window.entries:
                age_hours = (now - entry.created_at).total_seconds() / 3600
                if age_hours < 1:
                    age_buckets["<1h"] += 1
                elif age_hours < 6:
                    age_buckets["1-6h"] += 1
                elif age_hours < 24:
                    age_buckets["6-24h"] += 1
                else:
                    age_buckets[">24h"] += 1
            
            return {
                "window_stats": {
                    "max_tokens": self.context_window.max_tokens,
                    "current_tokens": self.context_window.current_tokens,
                    "available_tokens": self.context_window.get_available_tokens(),
                    "utilization_ratio": self.context_window.utilization_ratio,
                    "total_entries": total_entries
                },
                "entries_by_type": type_counts,
                "entries_by_priority": dict(priority_counts),
                "tokens_by_priority": dict(priority_tokens),
                "age_distribution": age_buckets,
                "optimization_stats": {
                    "last_optimized": self.context_window.last_optimized.isoformat() if self.context_window.last_optimized else None,
                    "optimization_count": self.context_window.optimization_count
                },
                "system_stats": self.stats.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get context summary: {e}")
            return {}
    
    async def _can_add_entry(self, entry: ContextEntry) -> bool:
        """Check if entry can be added without exceeding limits."""
        
        # Check token limit
        if not self.context_window.can_fit(entry.estimated_tokens):
            return False
        
        # Check priority-based limits
        priority_limit = self.priority_limits.get(entry.priority.value, 0.1)
        max_tokens_for_priority = int(self.max_tokens * priority_limit)
        
        current_tokens_for_priority = sum(
            e.estimated_tokens for e in self.context_window.entries
            if e.priority == entry.priority
        )
        
        if current_tokens_for_priority + entry.estimated_tokens > max_tokens_for_priority:
            return False
        
        return True
    
    def _get_default_priority(self, context_type: ContextType) -> ContextPriority:
        """Get default priority for a context type."""
        
        type_config = self.type_config.get(context_type.value, {})
        priority_value = type_config.get("default_priority", ContextPriority.MEDIUM.value)
        
        return ContextPriority(priority_value)
    
    def _get_default_expiration_hours(self, context_type: ContextType) -> Optional[float]:
        """Get default expiration hours for a context type."""
        
        type_config = self.type_config.get(context_type.value, {})
        
        if not type_config.get("can_expire", True):
            return None
        
        return type_config.get("expire_hours")
    
    def _can_summarize(self, context_type: ContextType) -> bool:
        """Check if context type can be summarized."""
        
        type_config = self.type_config.get(context_type.value, {})
        return type_config.get("can_summarize", True)
    
    async def _cleanup_expired_entries(self) -> int:
        """Remove expired context entries."""
        
        try:
            expired_entries = [
                entry for entry in self.context_window.entries
                if entry.is_expired() and not entry.is_persistent
            ]
            
            removed_count = 0
            for entry in expired_entries:
                if await self.remove_context(entry.id):
                    removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} expired context entries")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Cleanup of expired entries failed: {e}")
            return 0
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup task."""
        
        cleanup_interval = self.config.get("cleanup_interval_minutes", 30) * 60
        
        while self.initialized:
            try:
                await asyncio.sleep(cleanup_interval)
                
                # Clean up expired entries
                await self._cleanup_expired_entries()
                
                # Update utilization statistics
                if len(self.context_window.entries) > 0:
                    current_util = self.context_window.utilization_ratio
                    avg_util = self.stats["avg_utilization"]
                    count = self.stats["optimization_count"] + 1
                    self.stats["avg_utilization"] = (avg_util * (count - 1) + current_util) / count
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    async def export_context(self, export_path: str) -> bool:
        """Export current context to a file."""
        
        try:
            import json
            
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "context_window": self.context_window.to_dict(),
                "summary": await self.get_context_summary(),
                "entries": [entry.to_dict() for entry in self.context_window.entries]
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Context exported to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Context export failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the context manager."""
        
        try:
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown optimizer
            if self.optimizer:
                await self.optimizer.shutdown()
            
            self.initialized = False
            self.logger.info("Context manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during context manager shutdown: {e}")