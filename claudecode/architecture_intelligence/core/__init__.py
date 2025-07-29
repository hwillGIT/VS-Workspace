"""
Architecture Intelligence Platform Core
Expert-level architecture framework support with intelligent orchestration
"""

from .intelligence_engine import ArchitectureIntelligenceEngine
from .pattern_miner import PatternMiner
from .recommendation_engine import RecommendationEngine
from .knowledge_graph import ArchitectureKnowledgeGraph

__version__ = "1.0.0"
__all__ = [
    "ArchitectureIntelligenceEngine",
    "PatternMiner", 
    "RecommendationEngine",
    "ArchitectureKnowledgeGraph"
]