"""
Memory manager for coordinating memory operations across agents.

This module provides centralized memory management capabilities,
handling memory sharing, synchronization, and cross-agent queries.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from .agent_memory import AgentMemory
from .memory_types import MemoryType, MemoryEntry, MemoryQuery, MemoryStats


class MemoryManager:
    """
    Centralized memory manager for multiple agents.
    
    Coordinates memory operations, enables memory sharing between agents,
    and provides system-wide memory analytics and management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.memory_dir = Path(config.get("memory_dir", "./memory"))
        self.enable_sharing = config.get("enable_memory_sharing", True)
        self.shared_memory_types = set(config.get("shared_memory_types", [
            MemoryType.SEMANTIC.value,
            MemoryType.PROCEDURAL.value,
            MemoryType.ERROR.value,
            MemoryType.SUCCESS.value
        ]))
        
        # Agent memories
        self.agent_memories: Dict[str, AgentMemory] = {}
        
        # Shared memory pool
        self.shared_memories: Dict[str, MemoryEntry] = {}
        
        # Cross-agent relationships
        self.memory_relationships: Dict[str, Set[str]] = {}  # memory_id -> set of related memory_ids
        
        # System-wide statistics
        self.system_stats = {
            "total_agents": 0,
            "total_memories": 0,
            "shared_memories": 0,
            "cross_agent_queries": 0,
            "memory_synchronizations": 0
        }
        
        # Background tasks
        self._sync_task = None
        self._analytics_task = None
        
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the memory manager."""
        
        try:
            # Create memory directory
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            
            # Load shared memories
            await self._load_shared_memories()
            
            # Start background tasks
            if self.enable_sharing:
                self._sync_task = asyncio.create_task(self._synchronization_loop())
            
            self._analytics_task = asyncio.create_task(self._analytics_loop())
            
            self.initialized = True
            self.logger.info("Memory manager initialized")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory manager: {e}")
            return False
    
    async def register_agent(self, agent_id: str, config: Optional[Dict[str, Any]] = None) -> AgentMemory:
        """
        Register a new agent and create its memory system.
        
        Args:
            agent_id: Unique agent identifier
            config: Agent-specific memory configuration
            
        Returns:
            AgentMemory instance for the agent
        """
        
        try:
            if agent_id in self.agent_memories:
                self.logger.warning(f"Agent {agent_id} already registered")
                return self.agent_memories[agent_id]
            
            # Create agent-specific config
            agent_config = self.config.copy()
            if config:
                agent_config.update(config)
            
            agent_config["agent_id"] = agent_id
            agent_config["memory_dir"] = str(self.memory_dir / agent_id)
            
            # Create and initialize agent memory
            agent_memory = AgentMemory(agent_config)
            await agent_memory.initialize()
            
            # Register agent
            self.agent_memories[agent_id] = agent_memory
            self.system_stats["total_agents"] += 1
            
            self.logger.info(f"Registered agent memory: {agent_id}")
            
            return agent_memory
            
        except Exception as e:
            self.logger.error(f"Failed to register agent {agent_id}: {e}")
            raise
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent and cleanup its memory."""
        
        try:
            if agent_id not in self.agent_memories:
                return False
            
            # Shutdown agent memory
            agent_memory = self.agent_memories[agent_id]
            await agent_memory.shutdown()
            
            # Remove from registry
            del self.agent_memories[agent_id]
            self.system_stats["total_agents"] -= 1
            
            self.logger.info(f"Unregistered agent memory: {agent_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    async def add_shared_memory(
        self,
        content: str,
        memory_type: MemoryType,
        source_agent: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        importance: float = 1.0
    ) -> str:
        """
        Add a memory to the shared memory pool.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            source_agent: Agent that created this memory
            metadata: Additional metadata
            tags: Memory tags
            importance: Importance score
            
        Returns:
            Memory ID
        """
        
        try:
            # Check if memory type should be shared
            if memory_type.value not in self.shared_memory_types:
                self.logger.debug(f"Memory type {memory_type.value} not configured for sharing")
                return ""
            
            # Add to source agent's memory first
            source_memory = self.agent_memories.get(source_agent)
            if source_memory:
                memory_id = await source_memory.add_memory(
                    content=content,
                    memory_type=memory_type,
                    metadata=metadata,
                    tags=tags,
                    importance=importance,
                    agent_id=source_agent
                )
                
                # Add to shared pool
                memory_entry = await source_memory.get_memory(memory_id)
                if memory_entry:
                    self.shared_memories[memory_id] = memory_entry
                    self.system_stats["shared_memories"] += 1
                    
                    # Save shared memories
                    await self._save_shared_memories()
                    
                    self.logger.debug(f"Added shared memory: {memory_id}")
                    
                    return memory_id
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Failed to add shared memory: {e}")
            return ""
    
    async def search_cross_agent(
        self,
        query: str,
        requesting_agent: str,
        memory_types: Optional[List[MemoryType]] = None,
        max_results: int = 10,
        include_own_memories: bool = True
    ) -> List[MemoryEntry]:
        """
        Search memories across all agents.
        
        Args:
            query: Search query
            requesting_agent: Agent making the request
            memory_types: Filter by memory types
            max_results: Maximum results to return
            include_own_memories: Whether to include requesting agent's memories
            
        Returns:
            List of relevant memories from all agents
        """
        
        try:
            self.system_stats["cross_agent_queries"] += 1
            
            all_results = []
            
            # Search shared memories first
            shared_results = await self._search_shared_memories(
                query, memory_types, max_results
            )
            all_results.extend(shared_results)
            
            # Search individual agent memories
            for agent_id, agent_memory in self.agent_memories.items():
                if not include_own_memories and agent_id == requesting_agent:
                    continue
                
                try:
                    agent_results = await agent_memory.search_memories(
                        query=query,
                        memory_types=memory_types,
                        max_results=max_results // len(self.agent_memories)
                    )
                    
                    # Add agent context to results
                    for result in agent_results:
                        result.metadata["source_agent"] = agent_id
                    
                    all_results.extend(agent_results)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to search agent {agent_id}: {e}")
            
            # Remove duplicates and sort by relevance
            unique_results = self._deduplicate_memories(all_results)
            unique_results.sort(key=lambda m: m.importance, reverse=True)
            
            return unique_results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Cross-agent search failed: {e}")
            return []
    
    async def _search_shared_memories(
        self,
        query: str,
        memory_types: Optional[List[MemoryType]],
        max_results: int
    ) -> List[MemoryEntry]:
        """Search shared memory pool."""
        
        query_lower = query.lower()
        matches = []
        
        for memory in self.shared_memories.values():
            # Apply filters
            if memory_types and memory.memory_type not in memory_types:
                continue
            
            # Simple text matching
            if query_lower in memory.content.lower():
                matches.append(memory)
        
        # Sort by importance
        matches.sort(key=lambda m: m.importance, reverse=True)
        
        return matches[:max_results]
    
    def _deduplicate_memories(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """Remove duplicate memories from results."""
        
        seen_ids = set()
        seen_hashes = set()
        unique_memories = []
        
        for memory in memories:
            # Skip if ID already seen
            if memory.id in seen_ids:
                continue
            
            # Skip if content hash already seen
            content_hash = memory.metadata.get("content_hash")
            if content_hash and content_hash in seen_hashes:
                continue
            
            seen_ids.add(memory.id)
            if content_hash:
                seen_hashes.add(content_hash)
            
            unique_memories.append(memory)
        
        return unique_memories
    
    async def synchronize_agent_memories(self, agent_id: str) -> int:
        """
        Synchronize an agent's shareable memories to the shared pool.
        
        Args:
            agent_id: Agent to synchronize
            
        Returns:
            Number of memories synchronized
        """
        
        try:
            agent_memory = self.agent_memories.get(agent_id)
            if not agent_memory:
                return 0
            
            sync_count = 0
            
            # Get shareable memory types from agent
            for memory_type in self.shared_memory_types:
                try:
                    memories = await agent_memory.get_memories_by_type(
                        MemoryType(memory_type), max_results=100
                    )
                    
                    for memory in memories:
                        # Only sync if not already in shared pool
                        if memory.id not in self.shared_memories:
                            self.shared_memories[memory.id] = memory
                            sync_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to sync {memory_type} memories: {e}")
            
            if sync_count > 0:
                self.system_stats["shared_memories"] += sync_count
                self.system_stats["memory_synchronizations"] += 1
                await self._save_shared_memories()
                
                self.logger.info(f"Synchronized {sync_count} memories for agent {agent_id}")
            
            return sync_count
            
        except Exception as e:
            self.logger.error(f"Failed to synchronize agent {agent_id}: {e}")
            return 0
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system-wide memory statistics."""
        
        try:
            # Update current stats
            self.system_stats["total_memories"] = sum(
                len(agent_memory.local_memories) 
                for agent_memory in self.agent_memories.values()
            )
            
            # Collect agent-specific stats
            agent_stats = {}
            for agent_id, agent_memory in self.agent_memories.items():
                agent_stats[agent_id] = agent_memory.get_stats().to_dict()
            
            # System-wide memory types distribution
            memory_types_distribution = {}
            for agent_memory in self.agent_memories.values():
                for memory in agent_memory.local_memories.values():
                    memory_type = memory.memory_type.value
                    memory_types_distribution[memory_type] = (
                        memory_types_distribution.get(memory_type, 0) + 1
                    )
            
            return {
                "system_stats": self.system_stats.copy(),
                "agent_stats": agent_stats,
                "memory_types_distribution": memory_types_distribution,
                "shared_memory_types": list(self.shared_memory_types),
                "memory_relationships_count": len(self.memory_relationships),
                "memory_manager_config": {
                    "enable_sharing": self.enable_sharing,
                    "shared_types": list(self.shared_memory_types)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system stats: {e}")
            return {}
    
    async def create_memory_relationship(self, memory_id1: str, memory_id2: str) -> bool:
        """Create a relationship between two memories."""
        
        try:
            # Add bidirectional relationship
            if memory_id1 not in self.memory_relationships:
                self.memory_relationships[memory_id1] = set()
            if memory_id2 not in self.memory_relationships:
                self.memory_relationships[memory_id2] = set()
            
            self.memory_relationships[memory_id1].add(memory_id2)
            self.memory_relationships[memory_id2].add(memory_id1)
            
            self.logger.debug(f"Created memory relationship: {memory_id1} <-> {memory_id2}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create memory relationship: {e}")
            return False
    
    async def get_related_memories(self, memory_id: str) -> List[MemoryEntry]:
        """Get memories related to a specific memory."""
        
        try:
            related_ids = self.memory_relationships.get(memory_id, set())
            related_memories = []
            
            # Search in shared memories
            for related_id in related_ids:
                if related_id in self.shared_memories:
                    related_memories.append(self.shared_memories[related_id])
            
            # Search in agent memories
            for agent_memory in self.agent_memories.values():
                for related_id in related_ids:
                    memory = await agent_memory.get_memory(related_id)
                    if memory:
                        related_memories.append(memory)
            
            return related_memories
            
        except Exception as e:
            self.logger.error(f"Failed to get related memories: {e}")
            return []
    
    async def _load_shared_memories(self) -> None:
        """Load shared memories from storage."""
        
        try:
            shared_memory_file = self.memory_dir / "shared_memories.json"
            
            if shared_memory_file.exists():
                import json
                with open(shared_memory_file, 'r') as f:
                    data = json.load(f)
                
                for memory_data in data.get("memories", []):
                    memory = MemoryEntry.from_dict(memory_data)
                    self.shared_memories[memory.id] = memory
                
                # Load relationships
                self.memory_relationships = {
                    memory_id: set(related_ids)
                    for memory_id, related_ids in data.get("relationships", {}).items()
                }
                
                self.logger.info(f"Loaded {len(self.shared_memories)} shared memories")
            
        except Exception as e:
            self.logger.warning(f"Failed to load shared memories: {e}")
    
    async def _save_shared_memories(self) -> None:
        """Save shared memories to storage."""
        
        try:
            import json
            
            # Convert relationships to serializable format
            serializable_relationships = {
                memory_id: list(related_ids)
                for memory_id, related_ids in self.memory_relationships.items()
            }
            
            data = {
                "saved_at": datetime.now().isoformat(),
                "memories": [memory.to_dict() for memory in self.shared_memories.values()],
                "relationships": serializable_relationships
            }
            
            shared_memory_file = self.memory_dir / "shared_memories.json"
            with open(shared_memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save shared memories: {e}")
    
    async def _synchronization_loop(self) -> None:
        """Background synchronization task."""
        
        sync_interval = self.config.get("sync_interval_minutes", 30) * 60
        
        while self.initialized:
            try:
                await asyncio.sleep(sync_interval)
                
                # Synchronize all agents
                for agent_id in self.agent_memories:
                    await self.synchronize_agent_memories(agent_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Synchronization loop error: {e}")
    
    async def _analytics_loop(self) -> None:
        """Background analytics and maintenance task."""
        
        analytics_interval = self.config.get("analytics_interval_hours", 6) * 3600
        
        while self.initialized:
            try:
                await asyncio.sleep(analytics_interval)
                
                # Update system statistics
                await self.get_system_stats()
                
                # Perform maintenance tasks
                await self._cleanup_stale_relationships()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Analytics loop error: {e}")
    
    async def _cleanup_stale_relationships(self) -> None:
        """Clean up relationships to non-existent memories."""
        
        try:
            all_memory_ids = set(self.shared_memories.keys())
            
            # Add memory IDs from all agents
            for agent_memory in self.agent_memories.values():
                all_memory_ids.update(agent_memory.local_memories.keys())
            
            # Remove stale relationships
            stale_relationships = []
            for memory_id, related_ids in self.memory_relationships.items():
                if memory_id not in all_memory_ids:
                    stale_relationships.append(memory_id)
                else:
                    # Remove stale related IDs
                    valid_related_ids = {
                        rid for rid in related_ids if rid in all_memory_ids
                    }
                    self.memory_relationships[memory_id] = valid_related_ids
            
            # Remove completely stale relationships
            for memory_id in stale_relationships:
                del self.memory_relationships[memory_id]
            
            if stale_relationships:
                self.logger.info(f"Cleaned up {len(stale_relationships)} stale memory relationships")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup stale relationships: {e}")
    
    async def export_system_memory(self, export_path: str) -> bool:
        """Export all system memories to a file."""
        
        try:
            import json
            
            # Collect all memories
            all_memories = {}
            agent_exports = {}
            
            # Export agent memories
            for agent_id, agent_memory in self.agent_memories.items():
                agent_export_path = f"{export_path}_{agent_id}.json"
                await agent_memory.export_memories(agent_export_path)
                agent_exports[agent_id] = agent_export_path
            
            # Create system export
            system_export = {
                "exported_at": datetime.now().isoformat(),
                "system_stats": await self.get_system_stats(),
                "shared_memories": [memory.to_dict() for memory in self.shared_memories.values()],
                "memory_relationships": {
                    memory_id: list(related_ids)
                    for memory_id, related_ids in self.memory_relationships.items()
                },
                "agent_exports": agent_exports
            }
            
            with open(export_path, 'w') as f:
                json.dump(system_export, f, indent=2)
            
            self.logger.info(f"Exported system memory to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"System memory export failed: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the memory manager."""
        
        try:
            # Cancel background tasks
            if self._sync_task:
                self._sync_task.cancel()
                try:
                    await self._sync_task
                except asyncio.CancelledError:
                    pass
            
            if self._analytics_task:
                self._analytics_task.cancel()
                try:
                    await self._analytics_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown all agent memories
            for agent_id, agent_memory in self.agent_memories.items():
                try:
                    await agent_memory.shutdown()
                except Exception as e:
                    self.logger.error(f"Failed to shutdown agent {agent_id}: {e}")
            
            # Final save of shared memories
            await self._save_shared_memories()
            
            self.initialized = False
            self.logger.info("Memory manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during memory manager shutdown: {e}")