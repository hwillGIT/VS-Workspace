"""
Context types and data structures for context engineering.

This module defines the core data structures used for intelligent
context management and optimization.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class ContextType(Enum):
    """Types of context information."""
    
    SYSTEM = "system"               # System instructions and configuration
    TASK = "task"                  # Current task description and requirements
    CONVERSATION = "conversation"   # Dialog history with user
    CODE = "code"                  # Code snippets and implementations
    DOCUMENTATION = "documentation" # Documentation and references
    ERROR = "error"                # Error messages and debugging info
    MEMORY = "memory"              # Retrieved memories and knowledge
    FEEDBACK = "feedback"          # User feedback and corrections  
    CONTEXT = "context"            # Meta-context about the situation
    TOOLS = "tools"                # Tool usage and results
    PLANNING = "planning"          # Planning and decision making
    REFLECTION = "reflection"      # Self-reflection and analysis


class ContextPriority(Enum):
    """Priority levels for context entries."""
    
    CRITICAL = 5    # Always include (system instructions, current task)
    HIGH = 4        # Include unless severe space constraints
    MEDIUM = 3      # Include if space permits
    LOW = 2         # Include only with ample space
    MINIMAL = 1     # Include only if context window is mostly empty


@dataclass
class ContextEntry:
    """A single context entry with metadata."""
    
    # Core fields
    id: str
    content: str
    context_type: ContextType
    priority: ContextPriority
    created_at: datetime
    
    # Size and token information
    content_length: int = 0
    estimated_tokens: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Context relationships
    related_entries: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # Entries this depends on
    
    # Usage tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance_score: float = 1.0  # Calculated based on usage and relevance
    
    # Temporal properties
    expires_at: Optional[datetime] = None
    is_persistent: bool = False  # Should survive context cleanups
    
    # Optimization hints
    can_summarize: bool = True   # Can be summarized when space is tight
    summary: Optional[str] = None
    compression_ratio: float = 1.0  # Original size / compressed size
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.content_length == 0:
            self.content_length = len(self.content)
        
        if self.estimated_tokens == 0:
            # Rough token estimation (1 token ≈ 4 characters for English)
            self.estimated_tokens = max(1, self.content_length // 4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "context_type": self.context_type.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "content_length": self.content_length,
            "estimated_tokens": self.estimated_tokens,
            "metadata": self.metadata,
            "tags": self.tags,
            "related_entries": self.related_entries,
            "dependencies": self.dependencies,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "importance_score": self.importance_score,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_persistent": self.is_persistent,
            "can_summarize": self.can_summarize,
            "summary": self.summary,
            "compression_ratio": self.compression_ratio
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            context_type=ContextType(data["context_type"]),
            priority=ContextPriority(data["priority"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            content_length=data.get("content_length", 0),
            estimated_tokens=data.get("estimated_tokens", 0),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            related_entries=data.get("related_entries", []),
            dependencies=data.get("dependencies", []),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            importance_score=data.get("importance_score", 1.0),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            is_persistent=data.get("is_persistent", False),
            can_summarize=data.get("can_summarize", True),
            summary=data.get("summary"),
            compression_ratio=data.get("compression_ratio", 1.0)
        )
    
    def update_access(self) -> None:
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def calculate_age_hours(self) -> float:
        """Calculate age in hours."""
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def get_effective_content(self, use_summary: bool = False) -> str:
        """Get content to use (original or summary)."""
        if use_summary and self.summary:
            return self.summary
        return self.content
    
    def get_effective_tokens(self, use_summary: bool = False) -> int:
        """Get token count for effective content."""
        if use_summary and self.summary:
            return max(1, len(self.summary) // 4)
        return self.estimated_tokens


@dataclass
class ContextWindow:
    """Represents the current context window state."""
    
    # Window constraints
    max_tokens: int
    current_tokens: int = 0
    
    # Context entries
    entries: List[ContextEntry] = field(default_factory=list)
    
    # Window statistics
    entries_by_type: Dict[str, int] = field(default_factory=dict)
    entries_by_priority: Dict[int, int] = field(default_factory=dict)
    utilization_ratio: float = 0.0
    
    # Optimization state
    last_optimized: Optional[datetime] = None
    optimization_count: int = 0
    
    def __post_init__(self):
        """Initialize computed fields."""
        self.update_statistics()
    
    def update_statistics(self) -> None:
        """Update window statistics."""
        self.current_tokens = sum(entry.estimated_tokens for entry in self.entries)
        self.utilization_ratio = self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0
        
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
    
    def get_available_tokens(self) -> int:
        """Get remaining token capacity."""
        return max(0, self.max_tokens - self.current_tokens)
    
    def can_fit(self, tokens: int) -> bool:
        """Check if additional tokens can fit."""
        return self.current_tokens + tokens <= self.max_tokens
    
    def is_over_limit(self) -> bool:
        """Check if context window exceeds limit."""
        return self.current_tokens > self.max_tokens
    
    def get_entries_by_type(self, context_type: ContextType) -> List[ContextEntry]:
        """Get all entries of a specific type."""
        return [entry for entry in self.entries if entry.context_type == context_type]
    
    def get_entries_by_priority(self, priority: ContextPriority) -> List[ContextEntry]:
        """Get all entries of a specific priority."""
        return [entry for entry in self.entries if entry.priority == priority]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "max_tokens": self.max_tokens,
            "current_tokens": self.current_tokens,
            "entries": [entry.to_dict() for entry in self.entries],
            "entries_by_type": self.entries_by_type,
            "entries_by_priority": self.entries_by_priority,
            "utilization_ratio": self.utilization_ratio,
            "last_optimized": self.last_optimized.isoformat() if self.last_optimized else None,
            "optimization_count": self.optimization_count
        }


@dataclass 
class ContextOptimizationResult:
    """Result of context optimization operation."""
    
    # Before/after statistics
    tokens_before: int
    tokens_after: int
    entries_before: int
    entries_after: int
    
    # Optimization actions taken
    entries_removed: List[str] = field(default_factory=list)
    entries_summarized: List[str] = field(default_factory=list)
    entries_compressed: List[str] = field(default_factory=list)
    
    # Performance metrics
    optimization_time: float = 0.0
    compression_ratio: float = 1.0  # tokens_before / tokens_after
    
    # Success indicators
    success: bool = True
    error_message: Optional[str] = None
    
    def tokens_saved(self) -> int:
        """Calculate tokens saved."""
        return max(0, self.tokens_before - self.tokens_after)
    
    def compression_percentage(self) -> float:
        """Calculate compression percentage."""
        if self.tokens_before == 0:
            return 0.0
        return (self.tokens_saved() / self.tokens_before) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "entries_before": self.entries_before,
            "entries_after": self.entries_after,
            "entries_removed": self.entries_removed,
            "entries_summarized": self.entries_summarized,
            "entries_compressed": self.entries_compressed,
            "optimization_time": self.optimization_time,
            "compression_ratio": self.compression_ratio,
            "tokens_saved": self.tokens_saved(),
            "compression_percentage": self.compression_percentage(),
            "success": self.success,
            "error_message": self.error_message
        }