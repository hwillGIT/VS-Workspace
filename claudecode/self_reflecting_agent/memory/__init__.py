"""
Memory management system for the Self-Reflecting Claude Code Agent.

This module provides persistent memory capabilities using mem0
for long-term knowledge retention and agent learning.
"""

from .agent_memory import AgentMemory
from .memory_types import MemoryType, MemoryEntry
from .memory_manager import MemoryManager

__all__ = [
    "AgentMemory",
    "MemoryType", 
    "MemoryEntry",
    "MemoryManager"
]