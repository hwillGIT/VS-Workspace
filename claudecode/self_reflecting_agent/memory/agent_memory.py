"""
Agent memory system implementation using mem0.

This module provides persistent memory capabilities for agents,
enabling long-term learning and knowledge retention.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from .memory_types import MemoryType, MemoryEntry, MemoryQuery, MemoryStats

try:
    from mem0 import Memory
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class AgentMemory:
    """
    Agent memory system with persistent storage.
    
    Provides episodic, semantic, and procedural memory capabilities
    using mem0 for vector storage and retrieval.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.agent_id = config.get("agent_id", "default_agent")
        self.memory_dir = Path(config.get("memory_dir", "./memory"))
        self.max_memories = config.get("max_memories", 10000)
        self.cleanup_interval = config.get("cleanup_interval_hours", 24)
        
        # mem0 configuration
        self.mem0_config = config.get("mem0", {
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": f"agent_memory_{self.agent_id}",
                    "path": str(self.memory_dir / "chroma_db")
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small"
                }
            }
        })
        
        # Memory components
        self.memory_client = None
        self.local_memories: Dict[str, MemoryEntry] = {}
        
        # Memory management
        self.importance_threshold = config.get("importance_threshold", 0.1)
        self.max_working_memory = config.get("max_working_memory", 50)
        
        # Statistics
        self.stats = MemoryStats()
        
        # Background tasks
        self._cleanup_task = None
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the memory system."""
        
        try:
            # Create memory directory
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize mem0 if available
            if MEM0_AVAILABLE:
                await self._initialize_mem0()
            else:
                self.logger.warning("mem0 not available, using local memory only")
                await self._initialize_local_memory()
            
            # Load existing memories
            await self._load_memories()
            
            # Start background cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.initialized = True
            self.logger.info(f"Agent memory initialized for agent: {self.agent_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent memory: {e}")
            return False
    
    async def _initialize_mem0(self) -> None:
        """Initialize mem0 client."""
        
        try:
            self.memory_client = Memory(config=self.mem0_config)
            self.logger.info("mem0 client initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize mem0: {e}")
            # Fall back to local memory
            await self._initialize_local_memory()
    
    async def _initialize_local_memory(self) -> None:
        """Initialize local memory fallback."""
        
        try:
            # Create simple local storage structure
            self.local_memory_file = self.memory_dir / f"{self.agent_id}_memories.json"
            
            self.logger.info("Local memory fallback initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize local memory: {e}")
    
    async def _load_memories(self) -> None:
        """Load existing memories from storage."""
        
        try:
            if self.local_memory_file.exists():
                import json
                with open(self.local_memory_file, 'r') as f:
                    data = json.load(f)
                
                for memory_data in data.get("memories", []):
                    memory = MemoryEntry.from_dict(memory_data)
                    self.local_memories[memory.id] = memory
                
                self.logger.info(f"Loaded {len(self.local_memories)} memories from local storage")
            
            # Update statistics
            await self._update_stats()
            
        except Exception as e:
            self.logger.warning(f"Failed to load memories: {e}")
    
    async def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        importance: float = 1.0,
        agent_id: Optional[str] = None
    ) -> str:
        """
        Add a new memory entry.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            metadata: Additional metadata
            tags: Memory tags
            importance: Importance score (0.0 to 1.0)
            agent_id: Agent ID (defaults to system agent ID)
            
        Returns:
            Memory ID
        """
        
        try:
            # Create memory entry
            memory_id = str(uuid.uuid4())
            memory = MemoryEntry(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                created_at=datetime.now(),
                metadata=metadata or {},
                tags=tags or [],
                importance=importance,
                agent_id=agent_id or self.agent_id
            )
            
            # Add to mem0 if available
            if self.memory_client and MEM0_AVAILABLE:
                try:
                    # Add to mem0 with metadata
                    mem0_metadata = {
                        "memory_id": memory_id,
                        "memory_type": memory_type.value,
                        "agent_id": memory.agent_id,
                        "importance": importance,
                        "tags": tags or [],
                        **(metadata or {})
                    }
                    
                    self.memory_client.add(
                        content,
                        user_id=memory.agent_id,
                        metadata=mem0_metadata
                    )
                    
                except Exception as e:
                    self.logger.warning(f"Failed to add memory to mem0: {e}")
            
            # Add to local storage
            self.local_memories[memory_id] = memory
            
            # Save to disk
            await self._save_memories()
            
            # Update statistics
            await self._update_stats()
            
            self.logger.debug(f"Added memory: {memory_id} ({memory_type.value})")
            
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to add memory: {e}")
            raise
    
    async def search_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        tags: Optional[List[str]] = None,
        max_results: int = 10,
        min_importance: float = 0.0
    ) -> List[MemoryEntry]:
        """
        Search for relevant memories.
        
        Args:
            query: Search query
            memory_types: Filter by memory types
            tags: Filter by tags
            max_results: Maximum results to return
            min_importance: Minimum importance threshold
            
        Returns:
            List of relevant memory entries
        """
        
        try:
            memories = []
            
            # Search with mem0 if available
            if self.memory_client and MEM0_AVAILABLE:
                try:
                    # Create search filters
                    filters = {}
                    if memory_types:
                        filters["memory_type"] = [mt.value for mt in memory_types]
                    if tags:
                        filters["tags"] = tags
                    if min_importance > 0:
                        filters["importance"] = {"$gte": min_importance}
                    
                    # Search memories
                    results = self.memory_client.search(
                        query=query,
                        user_id=self.agent_id,
                        limit=max_results,
                        filters=filters
                    )
                    
                    # Convert results to MemoryEntry objects
                    for result in results:
                        memory_id = result.get("metadata", {}).get("memory_id")
                        if memory_id and memory_id in self.local_memories:
                            memory = self.local_memories[memory_id]
                            memory.update_access()
                            memories.append(memory)
                    
                except Exception as e:
                    self.logger.warning(f"mem0 search failed: {e}")
            
            # Fallback to local search if mem0 failed or not available
            if not memories:
                memories = await self._local_search(
                    query, memory_types, tags, max_results, min_importance
                )
            
            # Sort by relevance (importance * recency)
            memories.sort(key=lambda m: self._calculate_relevance_score(m, query), reverse=True)
            
            self.logger.debug(f"Found {len(memories)} memories for query: {query}")
            
            return memories[:max_results]
            
        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            return []
    
    async def _local_search(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]],
        tags: Optional[List[str]],
        max_results: int,
        min_importance: float
    ) -> List[MemoryEntry]:
        """Fallback local search implementation."""
        
        query_lower = query.lower()
        matches = []
        
        for memory in self.local_memories.values():
            # Apply filters
            if memory_types and memory.memory_type not in memory_types:
                continue
            
            if tags and not any(tag in memory.tags for tag in tags):
                continue
            
            if memory.importance < min_importance:
                continue
            
            # Simple text matching
            content_lower = memory.content.lower()
            if query_lower in content_lower:
                memory.update_access()
                matches.append(memory)
        
        return matches
    
    def _calculate_relevance_score(self, memory: MemoryEntry, query: str) -> float:
        """Calculate relevance score for ranking."""
        
        # Base score from importance
        score = memory.importance
        
        # Recency bonus (memories from last 7 days get bonus)
        days_old = (datetime.now() - memory.created_at).days
        if days_old < 7:
            score += 0.2 * (7 - days_old) / 7
        
        # Access count bonus
        if memory.access_count > 0:
            score += min(0.1, memory.access_count * 0.01)
        
        # Confidence factor
        score *= memory.confidence
        
        return score
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID."""
        
        memory = self.local_memories.get(memory_id)
        if memory:
            memory.update_access()
        
        return memory
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> bool:
        """Update an existing memory."""
        
        try:
            memory = self.local_memories.get(memory_id)
            if not memory:
                return False
            
            # Update fields
            if content is not None:
                memory.content = content
            
            if metadata is not None:
                memory.metadata.update(metadata)
            
            if tags is not None:
                memory.tags = tags
            
            if importance is not None:
                memory.update_importance(importance)
            
            if confidence is not None:
                memory.update_confidence(confidence)
            
            memory.updated_at = datetime.now()
            
            # Update in mem0 if available
            if self.memory_client and MEM0_AVAILABLE:
                try:
                    # mem0 doesn't have direct update, so we'd need to delete and re-add
                    # For now, just log the update
                    self.logger.debug(f"Memory {memory_id} updated (mem0 sync pending)")
                except Exception as e:
                    self.logger.warning(f"Failed to update memory in mem0: {e}")
            
            # Save changes
            await self._save_memories()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update memory {memory_id}: {e}")
            return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        
        try:
            if memory_id not in self.local_memories:
                return False
            
            # Delete from mem0 if available
            if self.memory_client and MEM0_AVAILABLE:
                try:
                    # mem0 doesn't have direct delete by ID, would need to implement
                    pass
                except Exception as e:
                    self.logger.warning(f"Failed to delete memory from mem0: {e}")
            
            # Delete from local storage
            del self.local_memories[memory_id]
            
            # Save changes
            await self._save_memories()
            
            # Update statistics
            await self._update_stats()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete memory {memory_id}: {e}")
            return False
    
    async def get_recent_memories(
        self, 
        hours: int = 24,
        memory_types: Optional[List[MemoryType]] = None,
        max_results: int = 50
    ) -> List[MemoryEntry]:
        """Get recent memories within specified time window."""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_memories = []
        
        for memory in self.local_memories.values():
            if memory.created_at >= cutoff_time:
                if memory_types is None or memory.memory_type in memory_types:
                    recent_memories.append(memory)
        
        # Sort by creation time (newest first)
        recent_memories.sort(key=lambda m: m.created_at, reverse=True)
        
        return recent_memories[:max_results]
    
    async def get_memories_by_type(
        self, 
        memory_type: MemoryType,
        max_results: int = 100
    ) -> List[MemoryEntry]:
        """Get all memories of a specific type."""
        
        memories = [
            memory for memory in self.local_memories.values()
            if memory.memory_type == memory_type
        ]
        
        # Sort by importance and recency
        memories.sort(key=lambda m: (m.importance, m.created_at), reverse=True)
        
        return memories[:max_results]
    
    async def consolidate_memories(self) -> int:
        """Consolidate similar memories to reduce redundancy."""
        
        try:
            consolidated_count = 0
            
            # Group memories by type
            memories_by_type = {}
            for memory in self.local_memories.values():
                memory_type = memory.memory_type
                if memory_type not in memories_by_type:
                    memories_by_type[memory_type] = []
                memories_by_type[memory_type].append(memory)
            
            # Look for similar memories within each type
            for memory_type, memories in memories_by_type.items():
                if len(memories) < 2:
                    continue
                
                # Simple similarity check based on content overlap
                to_consolidate = []
                for i, memory1 in enumerate(memories):
                    for memory2 in memories[i+1:]:
                        if self._memories_similar(memory1, memory2):
                            if memory1.id not in [m.id for m in to_consolidate]:
                                to_consolidate.append(memory1)
                            if memory2.id not in [m.id for m in to_consolidate]:
                                to_consolidate.append(memory2)
                
                # Consolidate similar memories
                if len(to_consolidate) >= 2:
                    consolidated_memory = await self._merge_memories(to_consolidate)
                    if consolidated_memory:
                        consolidated_count += len(to_consolidate) - 1
            
            if consolidated_count > 0:
                await self._save_memories()
                await self._update_stats()
                self.logger.info(f"Consolidated {consolidated_count} memories")
            
            return consolidated_count
            
        except Exception as e:
            self.logger.error(f"Memory consolidation failed: {e}")
            return 0
    
    def _memories_similar(self, memory1: MemoryEntry, memory2: MemoryEntry) -> bool:
        """Check if two memories are similar enough to consolidate."""
        
        # Simple similarity check
        content1_words = set(memory1.content.lower().split())
        content2_words = set(memory2.content.lower().split())
        
        if not content1_words or not content2_words:
            return False
        
        # Jaccard similarity
        intersection = len(content1_words.intersection(content2_words))
        union = len(content1_words.union(content2_words))
        
        similarity = intersection / union if union > 0 else 0
        
        # Consider similar if > 70% overlap and same type
        return (similarity > 0.7 and 
                memory1.memory_type == memory2.memory_type and
                abs((memory1.created_at - memory2.created_at).days) < 7)
    
    async def _merge_memories(self, memories: List[MemoryEntry]) -> Optional[MemoryEntry]:
        """Merge multiple similar memories into one."""
        
        try:
            if not memories:
                return None
            
            # Use the most important memory as base
            base_memory = max(memories, key=lambda m: m.importance)
            
            # Merge content
            merged_content = base_memory.content
            for memory in memories:
                if memory.id != base_memory.id:
                    merged_content += f"\n\n[Merged from {memory.id}]: {memory.content}"
            
            # Merge metadata and tags
            merged_metadata = base_memory.metadata.copy()
            merged_tags = set(base_memory.tags)
            
            for memory in memories:
                merged_metadata.update(memory.metadata)
                merged_tags.update(memory.tags)
            
            # Create new consolidated memory
            consolidated_id = await self.add_memory(
                content=merged_content,
                memory_type=base_memory.memory_type,
                metadata=merged_metadata,
                tags=list(merged_tags),
                importance=max(m.importance for m in memories),
                agent_id=base_memory.agent_id
            )
            
            # Delete original memories
            for memory in memories:
                await self.delete_memory(memory.id)
            
            return self.local_memories.get(consolidated_id)
            
        except Exception as e:
            self.logger.error(f"Failed to merge memories: {e}")
            return None
    
    async def _cleanup_old_memories(self) -> int:
        """Clean up old, low-importance memories."""
        
        try:
            if len(self.local_memories) <= self.max_memories:
                return 0
            
            # Sort memories by cleanup priority (low importance + old)
            memories_list = list(self.local_memories.values())
            memories_list.sort(key=lambda m: (
                m.importance,
                m.access_count,
                m.created_at.timestamp()
            ))
            
            # Calculate how many to remove
            excess_count = len(self.local_memories) - self.max_memories
            to_remove = memories_list[:excess_count]
            
            # Remove low-priority memories
            removed_count = 0
            for memory in to_remove:
                if memory.importance < self.importance_threshold:
                    await self.delete_memory(memory.id)
                    removed_count += 1
            
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old memories")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")
            return 0
    
    async def _save_memories(self) -> None:
        """Save memories to local storage."""
        
        try:
            import json
            
            data = {
                "agent_id": self.agent_id,
                "saved_at": datetime.now().isoformat(),
                "memories": [memory.to_dict() for memory in self.local_memories.values()]
            }
            
            with open(self.local_memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save memories: {e}")
    
    async def _update_stats(self) -> None:
        """Update memory statistics."""
        
        try:
            memories = list(self.local_memories.values())
            
            self.stats.total_memories = len(memories)
            
            if memories:
                # Memories by type
                self.stats.memories_by_type = {}
                for memory in memories:
                    memory_type = memory.memory_type.value
                    self.stats.memories_by_type[memory_type] = (
                        self.stats.memories_by_type.get(memory_type, 0) + 1
                    )
                
                # Memories by agent
                self.stats.memories_by_agent = {}
                for memory in memories:
                    agent_id = memory.agent_id or "unknown"
                    self.stats.memories_by_agent[agent_id] = (
                        self.stats.memories_by_agent.get(agent_id, 0) + 1
                    )
                
                # Average scores
                self.stats.avg_importance = sum(m.importance for m in memories) / len(memories)
                self.stats.avg_confidence = sum(m.confidence for m in memories) / len(memories)
                
                # Access statistics
                self.stats.total_access_count = sum(m.access_count for m in memories)
                most_accessed = max(memories, key=lambda m: m.access_count)
                self.stats.most_accessed_memory_id = most_accessed.id
                
                # Date ranges
                sorted_by_date = sorted(memories, key=lambda m: m.created_at)
                self.stats.oldest_memory_date = sorted_by_date[0].created_at
                self.stats.newest_memory_date = sorted_by_date[-1].created_at
                
                # Rough memory size calculation
                total_content_size = sum(len(m.content.encode('utf-8')) for m in memories)
                self.stats.memory_size_mb = total_content_size / (1024 * 1024)
            
        except Exception as e:
            self.logger.error(f"Failed to update stats: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup task."""
        
        while self.initialized:
            try:
                await asyncio.sleep(self.cleanup_interval * 3600)  # Convert hours to seconds
                
                # Perform cleanup tasks
                await self.consolidate_memories()
                await self._cleanup_old_memories()
                await self._update_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    def get_stats(self) -> MemoryStats:
        """Get memory system statistics."""
        return self.stats
    
    async def export_memories(self, export_path: str) -> bool:
        """Export all memories to a file."""
        
        try:
            import json
            
            export_data = {
                "agent_id": self.agent_id,
                "exported_at": datetime.now().isoformat(),
                "stats": self.stats.to_dict(),
                "memories": [memory.to_dict() for memory in self.local_memories.values()]
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported {len(self.local_memories)} memories to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory export failed: {e}")
            return False
    
    async def import_memories(self, import_path: str) -> int:
        """Import memories from a file."""
        
        try:
            import json
            
            with open(import_path, 'r') as f:
                data = json.load(f)
            
            imported_count = 0
            for memory_data in data.get("memories", []):
                try:
                    memory = MemoryEntry.from_dict(memory_data)
                    self.local_memories[memory.id] = memory
                    imported_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to import memory: {e}")
            
            if imported_count > 0:
                await self._save_memories()
                await self._update_stats()
            
            self.logger.info(f"Imported {imported_count} memories from: {import_path}")
            return imported_count
            
        except Exception as e:
            self.logger.error(f"Memory import failed: {e}")
            return 0
    
    async def shutdown(self) -> None:
        """Shutdown the memory system."""
        
        try:
            # Cancel cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Final save
            await self._save_memories()
            
            # Shutdown mem0 client if available
            if self.memory_client:
                # mem0 doesn't have explicit shutdown, just clear reference
                self.memory_client = None
            
            self.initialized = False
            self.logger.info("Agent memory system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during memory shutdown: {e}")