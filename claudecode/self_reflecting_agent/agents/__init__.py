"""
Core agent implementations for the Self-Reflecting Claude Code Agent system.

This module contains the four core agents:
- ManagerAgent: Orchestrates tasks and coordinates other agents  
- CoderAgent: Implements code solutions and handles development tasks
- ReviewerAgent: Reviews code quality, security, and best practices
- ResearcherAgent: Researches solutions, analyzes codebases, and gathers information
"""

from .base_agent import BaseAgent
from .manager_agent import ManagerAgent
from .coder_agent import CoderAgent
from .reviewer_agent import ReviewerAgent
from .researcher_agent import ResearcherAgent

__all__ = [
    "BaseAgent",
    "ManagerAgent", 
    "CoderAgent",
    "ReviewerAgent",
    "ResearcherAgent"
]