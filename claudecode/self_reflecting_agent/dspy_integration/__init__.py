"""
DSPy integration for optimizable agent cognition.

This module provides DSPy integration for the Self-Reflecting Claude Code Agent,
enabling optimizable prompting and continuous learning capabilities.
"""

from .dspy_manager import DSPyManager
from .signatures import AgentSignatures
from .optimization import SignatureOptimizer
from .metrics import DSPyMetrics

__all__ = [
    "DSPyManager",
    "AgentSignatures", 
    "SignatureOptimizer",
    "DSPyMetrics"
]