"""
Self-Reflecting Claude Code Agent

A sophisticated agent framework combining LangGraph orchestration with DSPy optimization
for autonomous code development with self-improvement capabilities.
"""

from .main import SelfReflectingAgent
from .agents import ManagerAgent, CoderAgent, ReviewerAgent, ResearcherAgent
from .workflows import DevelopmentWorkflow
from .rag import HybridRAG
from .memory import AgentMemory
from .context import ContextManager

__version__ = "1.0.0"
__author__ = "Claude Code Agent Team"

__all__ = [
    "SelfReflectingAgent",
    "ManagerAgent", 
    "CoderAgent",
    "ReviewerAgent", 
    "ResearcherAgent",
    "DevelopmentWorkflow",
    "HybridRAG",
    "AgentMemory",
    "ContextManager"
]