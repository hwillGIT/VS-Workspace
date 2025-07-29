"""
Context window implementation for managing active context.

This module provides the core ContextWindow class for managing
the active context window with intelligent token management
and optimization capabilities.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .context_types import ContextType, ContextPriority, ContextEntry


class ContextWindow:
    """
    Active context window with intelligent management.
    
    Manages the current context window state, tracks token usage,
    and provides utilities for context manipulation and analysis.
    """
    
    def __init__(self, max_tokens: int = 4000):
        self.logger = logging.getLogger(__name__)
        
        # Window constraints
        self.max_tokens = max_tokens
        self.current_tokens = 0
        
        # Context entries
        self.entries: List[ContextEntry] = []
        
        # Window statistics
        self.entries_by_type: Dict[str, int] = {}
        self.entries_by_priority: Dict[int, int] = {}
        self.utilization_ratio = 0.0
        
        # Optimization state
        self.last_optimized: Optional[datetime] = None
        self.optimization_count = 0
        
        # Performance tracking
        self.access_patterns: Dict[str, int] = defaultdict(int)
        self.type_access_counts: Dict[str, int] = defaultdict(int)
        
    def add_entry(self, entry: ContextEntry) -> bool:
        """
        Add an entry to the context window.
        
        Args:
            entry: Context entry to add
            
        Returns:
            True if entry was added successfully
        """
        
        try:
            # Check if entry would fit
            if not self.can_fit(entry.estimated_tokens):
                return False
            
            # Add entry
            self.entries.append(entry)
            self.current_tokens += entry.estimated_tokens
            
            # Update statistics
            self.update_statistics()
            
            self.logger.debug(f"Added context entry: {entry.id} ({entry.estimated_tokens} tokens)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add context entry: {e}")
            return False
    
    def remove_entry(self, entry_id: str) -> bool:
        """
        Remove an entry from the context window.
        
        Args:
            entry_id: ID of entry to remove
            
        Returns:
            True if entry was removed successfully
        """
        
        try:
            for i, entry in enumerate(self.entries):
                if entry.id == entry_id:
                    # Remove entry
                    removed_entry = self.entries.pop(i)
                    self.current_tokens -= removed_entry.estimated_tokens
                    
                    # Update statistics
                    self.update_statistics()
                    
                    self.logger.debug(f"Removed context entry: {entry_id}")
                    
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to remove context entry: {e}")
            return False
    
    def update_entry(self, entry_id: str, new_content: str) -> bool:
        """
        Update an entry's content and recalculate tokens.
        
        Args:
            entry_id: ID of entry to update
            new_content: New content for the entry
            
        Returns:
            True if entry was updated successfully
        """
        
        try:
            for entry in self.entries:
                if entry.id == entry_id:
                    # Update token count
                    old_tokens = entry.estimated_tokens
                    entry.content = new_content
                    entry.content_length = len(new_content)
                    entry.estimated_tokens = max(1, len(new_content) // 4)
                    
                    # Update total tokens
                    self.current_tokens = self.current_tokens - old_tokens + entry.estimated_tokens
                    
                    # Update statistics
                    self.update_statistics()
                    
                    self.logger.debug(f"Updated context entry: {entry_id}")
                    
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to update context entry: {e}")
            return False
    
    def get_entry(self, entry_id: str) -> Optional[ContextEntry]:
        """Get an entry by ID."""
        
        for entry in self.entries:
            if entry.id == entry_id:
                # Track access
                entry.update_access()
                self.access_patterns[entry_id] += 1
                self.type_access_counts[entry.context_type.value] += 1
                
                return entry
        
        return None
    
    def get_entries_by_type(self, context_type: ContextType) -> List[ContextEntry]:
        """Get all entries of a specific type."""
        
        entries = [entry for entry in self.entries if entry.context_type == context_type]
        
        # Track access
        for entry in entries:
            entry.update_access()
            self.access_patterns[entry.id] += 1
            self.type_access_counts[context_type.value] += 1
        
        return entries
    
    def get_entries_by_priority(self, priority: ContextPriority) -> List[ContextEntry]:
        """Get all entries of a specific priority."""
        
        entries = [entry for entry in self.entries if entry.priority == priority]
        
        # Track access
        for entry in entries:
            entry.update_access()
            self.access_patterns[entry.id] += 1
            self.type_access_counts[entry.context_type.value] += 1
        
        return entries
    
    def get_entries_sorted(self, sort_by: str = "priority") -> List[ContextEntry]:
        """
        Get entries sorted by specified criteria.
        
        Args:
            sort_by: Sort criteria ("priority", "created", "accessed", "tokens", "importance")
            
        Returns:
            Sorted list of entries
        """
        
        try:
            if sort_by == "priority":
                return sorted(self.entries, key=lambda e: e.priority.value, reverse=True)
            elif sort_by == "created":
                return sorted(self.entries, key=lambda e: e.created_at, reverse=True)
            elif sort_by == "accessed":
                return sorted(self.entries, key=lambda e: e.last_accessed or e.created_at, reverse=True)
            elif sort_by == "tokens":
                return sorted(self.entries, key=lambda e: e.estimated_tokens, reverse=True)
            elif sort_by == "importance":
                return sorted(self.entries, key=lambda e: e.importance_score, reverse=True)
            else:
                return self.entries.copy()
                
        except Exception as e:
            self.logger.error(f"Failed to sort entries: {e}")
            return self.entries.copy()
    
    def can_fit(self, tokens: int) -> bool:
        """Check if additional tokens can fit in the window."""
        return self.current_tokens + tokens <= self.max_tokens
    
    def get_available_tokens(self) -> int:
        """Get remaining token capacity."""
        return max(0, self.max_tokens - self.current_tokens)
    
    def is_over_limit(self) -> bool:
        """Check if context window exceeds token limit."""
        return self.current_tokens > self.max_tokens
    
    def get_utilization_ratio(self) -> float:
        """Get current utilization ratio (0.0 to 1.0+)."""
        return self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0
    
    def update_statistics(self) -> None:
        """Update window statistics."""
        
        try:
            # Recalculate current tokens
            self.current_tokens = sum(entry.estimated_tokens for entry in self.entries)
            self.utilization_ratio = self.get_utilization_ratio()
            
            # Count by type
            self.entries_by_type.clear()
            for entry in self.entries:
                context_type = entry.context_type.value
                self.entries_by_type[context_type] = self.entries_by_type.get(context_type, 0) + 1
            
            # Count by priority
            self.entries_by_priority.clear()
            for entry in self.entries:
                priority = entry.priority.value
                self.entries_by_priority[priority] = self.entries_by_priority.get(priority, 0) + 1
                
        except Exception as e:
            self.logger.error(f"Failed to update statistics: {e}")
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the current context window state."""
        
        try:
            # Token distribution by type
            type_tokens = defaultdict(int)
            for entry in self.entries:
                type_tokens[entry.context_type.value] += entry.estimated_tokens
            
            # Token distribution by priority
            priority_tokens = defaultdict(int)
            for entry in self.entries:
                priority_tokens[entry.priority.value] += entry.estimated_tokens
            
            # Age analysis
            now = datetime.now()
            age_buckets = {"<1h": 0, "1-6h": 0, "6-24h": 0, ">24h": 0}
            
            for entry in self.entries:
                age_hours = (now - entry.created_at).total_seconds() / 3600
                if age_hours < 1:
                    age_buckets["<1h"] += 1
                elif age_hours < 6:
                    age_buckets["1-6h"] += 1
                elif age_hours < 24:
                    age_buckets["6-24h"] += 1
                else:
                    age_buckets[">24h"] += 1
            
            # Access patterns
            most_accessed_type = max(self.type_access_counts.items(), key=lambda x: x[1])[0] if self.type_access_counts else None
            
            return {
                "window_state": {
                    "max_tokens": self.max_tokens,
                    "current_tokens": self.current_tokens,
                    "available_tokens": self.get_available_tokens(),
                    "utilization_ratio": self.utilization_ratio,
                    "total_entries": len(self.entries),
                    "is_over_limit": self.is_over_limit()
                },
                "distribution": {
                    "tokens_by_type": dict(type_tokens),
                    "tokens_by_priority": dict(priority_tokens),
                    "entries_by_type": self.entries_by_type.copy(),
                    "entries_by_priority": self.entries_by_priority.copy()
                },
                "age_analysis": age_buckets,
                "access_patterns": {
                    "most_accessed_type": most_accessed_type,
                    "total_accesses": sum(self.access_patterns.values()),
                    "unique_entries_accessed": len(self.access_patterns)
                },
                "optimization": {
                    "last_optimized": self.last_optimized.isoformat() if self.last_optimized else None,
                    "optimization_count": self.optimization_count,
                    "needs_optimization": self.utilization_ratio > 0.9
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get context summary: {e}")
            return {}
    
    def find_similar_entries(self, reference_entry: ContextEntry, similarity_threshold: float = 0.7) -> List[ContextEntry]:
        """
        Find entries similar to a reference entry.
        
        Args:
            reference_entry: Entry to compare against
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            List of similar entries
        """
        
        try:
            similar_entries = []
            
            # Get reference content words
            ref_words = set(reference_entry.content.lower().split())
            
            for entry in self.entries:
                if entry.id == reference_entry.id:
                    continue
                
                # Must be same type
                if entry.context_type != reference_entry.context_type:
                    continue
                
                # Calculate similarity
                entry_words = set(entry.content.lower().split())
                
                if not ref_words or not entry_words:
                    continue
                
                # Jaccard similarity
                intersection = len(ref_words.intersection(entry_words))
                union = len(ref_words.union(entry_words))
                
                similarity = intersection / union if union > 0 else 0
                
                if similarity >= similarity_threshold:
                    similar_entries.append(entry)
            
            return similar_entries
            
        except Exception as e:
            self.logger.error(f"Failed to find similar entries: {e}")
            return []
    
    def compact_window(self, target_utilization: float = 0.8) -> Dict[str, Any]:
        """
        Compact the window to achieve target utilization.
        
        Args:
            target_utilization: Target utilization ratio (0.0 to 1.0)
            
        Returns:
            Compaction result with statistics
        """
        
        try:
            if self.utilization_ratio <= target_utilization:
                return {"success": True, "action": "no_compaction_needed"}
            
            target_tokens = int(self.max_tokens * target_utilization)
            tokens_to_remove = self.current_tokens - target_tokens
            
            if tokens_to_remove <= 0:
                return {"success": True, "action": "no_compaction_needed"}
            
            # Sort entries by removal priority (least important first)
            entries_by_priority = sorted(
                self.entries,
                key=lambda e: (
                    e.priority.value,  # Lower priority first
                    -(e.access_count or 0),  # Less accessed first
                    e.created_at.timestamp()  # Older first
                )
            )
            
            # Remove entries until target is reached
            removed_entries = []
            tokens_removed = 0
            
            for entry in entries_by_priority:
                # Don't remove critical or persistent entries
                if (entry.priority == ContextPriority.CRITICAL or 
                    entry.is_persistent):
                    continue
                
                if tokens_removed >= tokens_to_remove:
                    break
                
                removed_entries.append(entry)
                tokens_removed += entry.estimated_tokens
                self.remove_entry(entry.id)
            
            return {
                "success": True,
                "action": "compacted",
                "entries_removed": len(removed_entries),
                "tokens_removed": tokens_removed,
                "new_utilization": self.get_utilization_ratio(),
                "removed_entry_ids": [e.id for e in removed_entries]
            }
            
        except Exception as e:
            self.logger.error(f"Window compaction failed: {e}")
            return {"success": False, "error": str(e)}
    
    def validate_window(self) -> Dict[str, Any]:
        """
        Validate window consistency and identify issues.
        
        Returns:
            Validation result with any issues found
        """
        
        try:
            issues = []
            
            # Check token calculation consistency
            calculated_tokens = sum(entry.estimated_tokens for entry in self.entries)
            if calculated_tokens != self.current_tokens:
                issues.append(f"Token count mismatch: calculated={calculated_tokens}, stored={self.current_tokens}")
                # Fix the issue
                self.current_tokens = calculated_tokens
            
            # Check for duplicate entries
            entry_ids = [entry.id for entry in self.entries]
            if len(entry_ids) != len(set(entry_ids)):
                duplicate_ids = [id for id in set(entry_ids) if entry_ids.count(id) > 1]
                issues.append(f"Duplicate entry IDs found: {duplicate_ids}")
            
            # Check for expired entries
            expired_entries = [entry for entry in self.entries if entry.is_expired()]
            if expired_entries:
                issues.append(f"Found {len(expired_entries)} expired entries")
            
            # Check for zero-token entries
            zero_token_entries = [entry for entry in self.entries if entry.estimated_tokens <= 0]
            if zero_token_entries:
                issues.append(f"Found {len(zero_token_entries)} entries with zero or negative tokens")
            
            # Check statistics consistency
            expected_type_counts = defaultdict(int)
            expected_priority_counts = defaultdict(int)
            
            for entry in self.entries:
                expected_type_counts[entry.context_type.value] += 1
                expected_priority_counts[entry.priority.value] += 1
            
            if dict(expected_type_counts) != self.entries_by_type:
                issues.append("Type count statistics are inconsistent")
                
            if dict(expected_priority_counts) != self.entries_by_priority:
                issues.append("Priority count statistics are inconsistent")
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "statistics": {
                    "total_entries": len(self.entries),
                    "total_tokens": self.current_tokens,
                    "utilization": self.utilization_ratio
                }
            }
            
        except Exception as e:
            self.logger.error(f"Window validation failed: {e}")
            return {"valid": False, "error": str(e)}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context window to dictionary representation."""
        
        return {
            "max_tokens": self.max_tokens,
            "current_tokens": self.current_tokens,
            "utilization_ratio": self.utilization_ratio,
            "total_entries": len(self.entries),
            "entries": [entry.to_dict() for entry in self.entries],
            "entries_by_type": self.entries_by_type.copy(),
            "entries_by_priority": self.entries_by_priority.copy(),
            "optimization_state": {
                "last_optimized": self.last_optimized.isoformat() if self.last_optimized else None,
                "optimization_count": self.optimization_count
            },
            "access_patterns": dict(self.access_patterns),
            "type_access_counts": dict(self.type_access_counts)
        }
    
    def clear(self) -> None:
        """Clear all entries from the context window."""
        
        self.entries.clear()
        self.current_tokens = 0
        self.entries_by_type.clear()
        self.entries_by_priority.clear()
        self.utilization_ratio = 0.0
        self.access_patterns.clear()
        self.type_access_counts.clear()
        
        self.logger.debug("Context window cleared")
    
    def __len__(self) -> int:
        """Get number of entries in the window."""
        return len(self.entries)
    
    def __contains__(self, entry_id: str) -> bool:
        """Check if entry ID exists in the window."""
        return any(entry.id == entry_id for entry in self.entries)
    
    def __str__(self) -> str:
        """String representation of the context window."""
        return f"ContextWindow({len(self.entries)} entries, {self.current_tokens}/{self.max_tokens} tokens, {self.utilization_ratio:.1%} used)"