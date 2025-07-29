"""
Memory types and data structures for the agent memory system.

This module defines the core data structures and types used
for memory management in the Self-Reflecting Claude Code Agent.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class MemoryType(Enum):
    """Types of memories that can be stored."""
    
    EPISODIC = "episodic"           # Specific experiences and events
    SEMANTIC = "semantic"           # General knowledge and facts
    PROCEDURAL = "procedural"       # How-to knowledge and skills
    WORKING = "working"             # Temporary, context-specific memory
    CONVERSATION = "conversation"   # Dialog history and context
    TASK = "task"                  # Task-specific information
    ERROR = "error"                # Error patterns and solutions
    SUCCESS = "success"            # Successful patterns and approaches
    FEEDBACK = "feedback"          # User feedback and corrections
    REFLECTION = "reflection"      # Self-reflection and insights


@dataclass
class MemoryEntry:
    """A single memory entry."""
    
    # Core fields
    id: str
    content: str
    memory_type: MemoryType
    created_at: datetime
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    importance: float = 1.0  # 0.0 to 1.0
    confidence: float = 1.0  # 0.0 to 1.0
    
    # Relationships
    related_memories: List[str] = field(default_factory=list)
    source: Optional[str] = None
    agent_id: Optional[str] = None
    
    # Usage tracking
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory entry to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "tags": self.tags,
            "importance": self.importance,
            "confidence": self.confidence,
            "related_memories": self.related_memories,
            "source": self.source,
            "agent_id": self.agent_id,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "context": self.context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create memory entry from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            importance=data.get("importance", 1.0),
            confidence=data.get("confidence", 1.0),
            related_memories=data.get("related_memories", []),
            source=data.get("source"),
            agent_id=data.get("agent_id"),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            context=data.get("context", {})
        )
    
    def update_access(self) -> None:
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def add_related_memory(self, memory_id: str) -> None:
        """Add a related memory ID if not already present."""
        if memory_id not in self.related_memories:
            self.related_memories.append(memory_id)
            self.updated_at = datetime.now()
    
    def update_importance(self, new_importance: float) -> None:
        """Update importance score."""
        self.importance = max(0.0, min(1.0, new_importance))
        self.updated_at = datetime.now()
    
    def update_confidence(self, new_confidence: float) -> None:
        """Update confidence score."""
        self.confidence = max(0.0, min(1.0, new_confidence))
        self.updated_at = datetime.now()


@dataclass
class MemoryQuery:
    """Query structure for memory retrieval."""
    
    query: str
    memory_types: Optional[List[MemoryType]] = None
    tags: Optional[List[str]] = None
    agent_id: Optional[str] = None
    min_importance: float = 0.0
    min_confidence: float = 0.0
    max_results: int = 10
    include_context: bool = True
    time_range: Optional[tuple] = None  # (start_time, end_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary."""
        return {
            "query": self.query,
            "memory_types": [mt.value for mt in self.memory_types] if self.memory_types else None,
            "tags": self.tags,
            "agent_id": self.agent_id,
            "min_importance": self.min_importance,
            "min_confidence": self.min_confidence,
            "max_results": self.max_results,
            "include_context": self.include_context,
            "time_range": [t.isoformat() for t in self.time_range] if self.time_range else None
        }


@dataclass
class MemoryStats:
    """Memory system statistics."""
    
    total_memories: int = 0
    memories_by_type: Dict[str, int] = field(default_factory=dict)
    memories_by_agent: Dict[str, int] = field(default_factory=dict)
    avg_importance: float = 0.0
    avg_confidence: float = 0.0
    total_access_count: int = 0
    most_accessed_memory_id: Optional[str] = None
    oldest_memory_date: Optional[datetime] = None
    newest_memory_date: Optional[datetime] = None
    memory_size_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_memories": self.total_memories,
            "memories_by_type": self.memories_by_type,
            "memories_by_agent": self.memories_by_agent,
            "avg_importance": self.avg_importance,
            "avg_confidence": self.avg_confidence,
            "total_access_count": self.total_access_count,
            "most_accessed_memory_id": self.most_accessed_memory_id,
            "oldest_memory_date": self.oldest_memory_date.isoformat() if self.oldest_memory_date else None,
            "newest_memory_date": self.newest_memory_date.isoformat() if self.newest_memory_date else None,
            "memory_size_mb": self.memory_size_mb
        }