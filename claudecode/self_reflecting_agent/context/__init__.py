"""
Context engineering framework for the Self-Reflecting Claude Code Agent.

This module provides intelligent context management to prevent context poisoning
and optimize token usage while maintaining semantic coherence.
"""

from .context_manager import ContextManager
from .context_window import ContextWindow
from .context_optimizer import ContextOptimizer
from .context_types import ContextType, ContextEntry, ContextPriority

__all__ = [
    "ContextManager",
    "ContextWindow", 
    "ContextOptimizer",
    "ContextType",
    "ContextEntry",
    "ContextPriority"
]