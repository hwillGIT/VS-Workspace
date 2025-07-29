"""
Evaluation types and data structures.

This module defines the core data structures used for agent evaluation,
performance tracking, and self-improvement feedback.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


class EvaluationType(Enum):
    """Types of evaluations that can be performed."""
    
    CODE_QUALITY = "code_quality"         # Code quality assessment
    TASK_COMPLETION = "task_completion"   # Task completion evaluation
    COMMUNICATION = "communication"      # Communication effectiveness
    REASONING = "reasoning"              # Reasoning and logic quality
    CREATIVITY = "creativity"            # Creative problem solving
    EFFICIENCY = "efficiency"            # Time and resource efficiency
    ACCURACY = "accuracy"                # Correctness of outputs
    HELPFULNESS = "helpfulness"          # User satisfaction and utility
    SAFETY = "safety"                   # Safety and ethical considerations
    CONSISTENCY = "consistency"         # Consistency across interactions
    LEARNING = "learning"               # Learning and adaptation capability
    OVERALL = "overall"                 # Overall performance assessment


class EvaluationCriteria(Enum):
    """Specific criteria for evaluation."""
    
    # Code Quality Criteria
    CORRECTNESS = "correctness"
    READABILITY = "readability"
    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    SECURITY = "security"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    
    # Task Completion Criteria
    COMPLETENESS = "completeness"
    TIMELINESS = "timeliness"
    REQUIREMENT_ADHERENCE = "requirement_adherence"
    PROBLEM_SOLVING = "problem_solving"
    
    # Communication Criteria
    CLARITY = "clarity"
    CONCISENESS = "conciseness"
    RELEVANCE = "relevance"
    TONE = "tone"
    STRUCTURE = "structure"
    
    # Reasoning Criteria
    LOGICAL_FLOW = "logical_flow"
    EVIDENCE_SUPPORT = "evidence_support"
    ASSUMPTION_VALIDITY = "assumption_validity"
    CONCLUSION_SOUNDNESS = "conclusion_soundness"
    
    # Efficiency Criteria
    TIME_EFFICIENCY = "time_efficiency"
    RESOURCE_UTILIZATION = "resource_utilization"
    OPTIMIZATION = "optimization"
    
    # Safety Criteria
    BIAS_DETECTION = "bias_detection"
    HARMFUL_CONTENT = "harmful_content"
    PRIVACY_PROTECTION = "privacy_protection"
    ETHICAL_COMPLIANCE = "ethical_compliance"


@dataclass
class EvaluationResult:
    """Result of an evaluation."""
    
    # Core identification
    evaluation_id: str
    evaluation_type: EvaluationType
    evaluated_item_id: str  # ID of the item being evaluated
    evaluator_id: str       # ID of the evaluator (human, LLM, system)
    
    # Evaluation metadata
    created_at: datetime
    evaluation_context: Dict[str, Any] = field(default_factory=dict)
    
    # Scores and ratings
    overall_score: float = 0.0  # 0.0 to 1.0 or 1 to 10 depending on scale
    criteria_scores: Dict[str, float] = field(default_factory=dict)  # Detailed criterion scores
    
    # Qualitative feedback
    summary: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    detailed_feedback: str = ""
    
    # Confidence and reliability
    confidence_score: float = 1.0  # How confident is the evaluator
    reliability_score: float = 1.0  # How reliable is this evaluation
    
    # Comparative data
    baseline_score: Optional[float] = None
    improvement_over_baseline: Optional[float] = None
    
    # Supporting evidence
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    
    # Evaluation metadata
    evaluation_method: str = "unknown"  # manual, llm_judge, automated, etc.
    evaluation_duration: float = 0.0    # Time taken for evaluation
    evaluation_cost: float = 0.0        # Cost of evaluation (if applicable)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "evaluation_type": self.evaluation_type.value,
            "evaluated_item_id": self.evaluated_item_id,
            "evaluator_id": self.evaluator_id,
            "created_at": self.created_at.isoformat(),
            "evaluation_context": self.evaluation_context,
            "overall_score": self.overall_score,
            "criteria_scores": self.criteria_scores,
            "summary": self.summary,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "recommendations": self.recommendations,
            "detailed_feedback": self.detailed_feedback,
            "confidence_score": self.confidence_score,
            "reliability_score": self.reliability_score,
            "baseline_score": self.baseline_score,
            "improvement_over_baseline": self.improvement_over_baseline,
            "evidence": self.evidence,
            "examples": self.examples,
            "evaluation_method": self.evaluation_method,
            "evaluation_duration": self.evaluation_duration,
            "evaluation_cost": self.evaluation_cost
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create from dictionary."""
        return cls(
            evaluation_id=data["evaluation_id"],
            evaluation_type=EvaluationType(data["evaluation_type"]),
            evaluated_item_id=data["evaluated_item_id"],
            evaluator_id=data["evaluator_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            evaluation_context=data.get("evaluation_context", {}),
            overall_score=data.get("overall_score", 0.0),
            criteria_scores=data.get("criteria_scores", {}),
            summary=data.get("summary", ""),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            recommendations=data.get("recommendations", []),
            detailed_feedback=data.get("detailed_feedback", ""),
            confidence_score=data.get("confidence_score", 1.0),
            reliability_score=data.get("reliability_score", 1.0),
            baseline_score=data.get("baseline_score"),
            improvement_over_baseline=data.get("improvement_over_baseline"),
            evidence=data.get("evidence", []),
            examples=data.get("examples", []),
            evaluation_method=data.get("evaluation_method", "unknown"),
            evaluation_duration=data.get("evaluation_duration", 0.0),
            evaluation_cost=data.get("evaluation_cost", 0.0)
        )
    
    def get_weighted_score(self, criteria_weights: Dict[str, float]) -> float:
        """Calculate weighted score based on criteria weights."""
        
        if not self.criteria_scores or not criteria_weights:
            return self.overall_score
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for criterion, weight in criteria_weights.items():
            if criterion in self.criteria_scores:
                weighted_sum += self.criteria_scores[criterion] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else self.overall_score
    
    def is_above_threshold(self, threshold: float) -> bool:
        """Check if evaluation meets minimum threshold."""
        return self.overall_score >= threshold
    
    def get_improvement_suggestions(self) -> List[str]:
        """Get actionable improvement suggestions."""
        suggestions = []
        
        # Add recommendations
        suggestions.extend(self.recommendations)
        
        # Generate suggestions from weaknesses
        for weakness in self.weaknesses:
            suggestions.append(f"Address weakness: {weakness}")
        
        # Generate suggestions from low criterion scores
        for criterion, score in self.criteria_scores.items():
            if score < 0.7:  # Below 70%
                suggestions.append(f"Improve {criterion}: current score {score:.2f}")
        
        return suggestions


@dataclass
class EvaluationRequest:
    """Request for evaluation."""
    
    # What to evaluate
    item_to_evaluate: Any
    item_id: str
    evaluation_type: EvaluationType
    
    # Evaluation configuration
    criteria: List[EvaluationCriteria] = field(default_factory=list)
    custom_criteria: Dict[str, str] = field(default_factory=dict)  # name -> description
    
    # Context and constraints
    context: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    
    # Evaluation preferences
    detailed_feedback: bool = True
    include_examples: bool = True
    compare_to_baseline: bool = False
    baseline_data: Optional[Dict[str, Any]] = None
    
    # Evaluator preferences
    preferred_evaluator: Optional[str] = None  # "human", "llm", "automated"
    evaluation_priority: str = "normal"  # "low", "normal", "high", "urgent"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "evaluation_type": self.evaluation_type.value,
            "criteria": [c.value for c in self.criteria],
            "custom_criteria": self.custom_criteria,
            "context": self.context,
            "requirements": self.requirements,
            "constraints": self.constraints,
            "detailed_feedback": self.detailed_feedback,
            "include_examples": self.include_examples,
            "compare_to_baseline": self.compare_to_baseline,
            "baseline_data": self.baseline_data,
            "preferred_evaluator": self.preferred_evaluator,
            "evaluation_priority": self.evaluation_priority
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for tracking improvement."""
    
    # Basic metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Efficiency metrics
    response_time: float = 0.0
    throughput: float = 0.0
    resource_usage: float = 0.0
    
    # Quality metrics
    user_satisfaction: float = 0.0
    task_completion_rate: float = 0.0
    error_rate: float = 0.0
    
    # Learning metrics
    improvement_rate: float = 0.0
    consistency_score: float = 0.0
    adaptability_score: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    measurement_period: str = ""
    sample_size: int = 0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "response_time": self.response_time,
            "throughput": self.throughput,
            "resource_usage": self.resource_usage,
            "user_satisfaction": self.user_satisfaction,
            "task_completion_rate": self.task_completion_rate,
            "error_rate": self.error_rate,
            "improvement_rate": self.improvement_rate,
            "consistency_score": self.consistency_score,
            "adaptability_score": self.adaptability_score,
            "custom_metrics": self.custom_metrics,
            "measurement_period": self.measurement_period,
            "sample_size": self.sample_size,
            "confidence_interval": self.confidence_interval
        }
    
    def get_overall_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall performance score."""
        
        if weights is None:
            # Default weights
            weights = {
                "accuracy": 0.2,
                "f1_score": 0.15,
                "user_satisfaction": 0.2,
                "task_completion_rate": 0.15,
                "response_time": 0.1,  # Lower is better, so invert
                "error_rate": 0.1,     # Lower is better, so invert
                "consistency_score": 0.1
            }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric, weight in weights.items():
            if hasattr(self, metric):
                value = getattr(self, metric)
                
                # Invert metrics where lower is better
                if metric in ["response_time", "error_rate", "resource_usage"]:
                    value = 1.0 - min(1.0, value)  # Assume normalized to [0,1]
                
                weighted_sum += value * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0


@dataclass
class EvaluationBatch:
    """Batch of evaluations for efficient processing."""
    
    batch_id: str
    requests: List[EvaluationRequest]
    created_at: datetime
    
    # Batch configuration
    batch_priority: str = "normal"
    parallel_processing: bool = True
    max_concurrent: int = 5
    
    # Progress tracking
    completed_count: int = 0
    failed_count: int = 0
    results: List[EvaluationResult] = field(default_factory=list)
    
    # Batch statistics
    total_cost: float = 0.0
    total_duration: float = 0.0
    
    def get_completion_rate(self) -> float:
        """Get batch completion rate."""
        total = len(self.requests)
        return (self.completed_count + self.failed_count) / total if total > 0 else 0.0
    
    def get_success_rate(self) -> float:
        """Get batch success rate."""
        processed = self.completed_count + self.failed_count
        return self.completed_count / processed if processed > 0 else 0.0
    
    def is_complete(self) -> bool:
        """Check if batch is complete."""
        return (self.completed_count + self.failed_count) >= len(self.requests)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at.isoformat(),
            "batch_priority": self.batch_priority,
            "parallel_processing": self.parallel_processing,
            "max_concurrent": self.max_concurrent,
            "total_requests": len(self.requests),
            "completed_count": self.completed_count,
            "failed_count": self.failed_count,
            "completion_rate": self.get_completion_rate(),
            "success_rate": self.get_success_rate(),
            "total_cost": self.total_cost,
            "total_duration": self.total_duration,
            "is_complete": self.is_complete()
        }