"""
Recommendation Engine for Architecture Intelligence

Provides intelligent recommendations based on patterns, context, and goals.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class RecommendationType(Enum):
    """Types of architectural recommendations"""
    PATTERN = "pattern"
    PRACTICE = "practice"
    FRAMEWORK = "framework"
    REFACTORING = "refactoring"
    MIGRATION = "migration"


@dataclass
class Recommendation:
    """Represents an architectural recommendation"""
    type: RecommendationType
    title: str
    description: str
    rationale: str
    confidence: float
    impact: str  # high, medium, low
    effort: str  # high, medium, low
    prerequisites: List[str] = None
    risks: List[str] = None
    references: List[str] = None

    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []
        if self.risks is None:
            self.risks = []
        if self.references is None:
            self.references = []


class RecommendationEngine:
    """
    Generates intelligent architectural recommendations based on:
    - Current patterns and anti-patterns
    - Project context and constraints
    - Framework-specific guidance
    - Historical success patterns
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recommendation_rules = self._load_recommendation_rules()
    
    def _load_recommendation_rules(self) -> Dict[str, Any]:
        """Load recommendation rules and heuristics"""
        return {
            "microservices_adoption": {
                "conditions": ["monolithic_architecture", "scalability_issues", "team_size > 20"],
                "recommendation": {
                    "type": RecommendationType.MIGRATION,
                    "title": "Migrate to Microservices Architecture",
                    "impact": "high",
                    "effort": "high"
                }
            },
            "event_sourcing": {
                "conditions": ["audit_requirements", "temporal_queries", "event_driven"],
                "recommendation": {
                    "type": RecommendationType.PATTERN,
                    "title": "Implement Event Sourcing",
                    "impact": "high",
                    "effort": "medium"
                }
            },
            "api_gateway": {
                "conditions": ["multiple_services", "external_clients", "authentication_complexity"],
                "recommendation": {
                    "type": RecommendationType.PATTERN,
                    "title": "Implement API Gateway Pattern",
                    "impact": "high",
                    "effort": "medium"
                }
            }
        }
    
    async def generate_recommendations(
        self,
        context: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        goals: List[str]
    ) -> List[Recommendation]:
        """
        Generate recommendations based on context, patterns, and goals.
        
        Args:
            context: Project context including constraints and requirements
            patterns: Currently identified patterns
            goals: Project goals (scalability, maintainability, etc.)
            
        Returns:
            List of prioritized recommendations
        """
        
        recommendations = []
        
        # Analyze current state
        current_patterns = {p['name'] for p in patterns}
        
        # Check recommendation rules
        for rule_name, rule in self.recommendation_rules.items():
            if self._evaluate_conditions(rule['conditions'], context, current_patterns):
                rec_data = rule['recommendation']
                
                recommendation = Recommendation(
                    type=rec_data['type'],
                    title=rec_data['title'],
                    description=self._generate_description(rule_name, context),
                    rationale=self._generate_rationale(rule_name, context, goals),
                    confidence=self._calculate_confidence(rule_name, context),
                    impact=rec_data['impact'],
                    effort=rec_data['effort'],
                    prerequisites=self._identify_prerequisites(rule_name, current_patterns),
                    risks=self._identify_risks(rule_name, context)
                )
                
                recommendations.append(recommendation)
        
        # Sort by impact and confidence
        recommendations.sort(key=lambda r: (
            self._impact_score(r.impact),
            r.confidence
        ), reverse=True)
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _evaluate_conditions(
        self,
        conditions: List[str],
        context: Dict[str, Any],
        current_patterns: Set[str]
    ) -> bool:
        """Evaluate if conditions are met for a recommendation"""
        # Simplified evaluation - would be more sophisticated in production
        for condition in conditions:
            if ">" in condition or "<" in condition:
                # Numeric comparison
                continue
            elif condition in context.get('requirements', []):
                continue
            elif condition in context.get('constraints', []):
                continue
            else:
                return False
        return True
    
    def _generate_description(self, rule_name: str, context: Dict[str, Any]) -> str:
        """Generate detailed description for recommendation"""
        descriptions = {
            "microservices_adoption": "Decompose the monolithic application into independently deployable microservices to improve scalability and team autonomy.",
            "event_sourcing": "Implement event sourcing to maintain a complete audit trail and enable temporal queries on system state.",
            "api_gateway": "Introduce an API Gateway to centralize cross-cutting concerns like authentication, rate limiting, and request routing."
        }
        return descriptions.get(rule_name, "Implement recommended architectural pattern.")
    
    def _generate_rationale(
        self,
        rule_name: str,
        context: Dict[str, Any],
        goals: List[str]
    ) -> str:
        """Generate rationale explaining why this recommendation is relevant"""
        goal_alignment = [g for g in goals if g in ['scalability', 'maintainability', 'security']]
        return f"This recommendation addresses {', '.join(goal_alignment)} goals based on your project context."
    
    def _calculate_confidence(self, rule_name: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score for recommendation"""
        # Simplified calculation
        base_confidence = 0.7
        
        # Adjust based on context match
        if context.get('domain') in ['ecommerce', 'fintech']:
            base_confidence += 0.1
        
        # Adjust based on team size
        team_size = context.get('team_size', 10)
        if team_size > 20:
            base_confidence += 0.1
        
        return min(base_confidence, 0.95)
    
    def _identify_prerequisites(
        self,
        rule_name: str,
        current_patterns: Set[str]
    ) -> List[str]:
        """Identify prerequisites for implementing recommendation"""
        prerequisites_map = {
            "microservices_adoption": [
                "Domain boundaries identified",
                "CI/CD pipeline established",
                "Container orchestration platform"
            ],
            "event_sourcing": [
                "Event store infrastructure",
                "CQRS understanding",
                "Eventual consistency acceptance"
            ],
            "api_gateway": [
                "Service discovery mechanism",
                "Authentication service",
                "Rate limiting strategy"
            ]
        }
        return prerequisites_map.get(rule_name, [])
    
    def _identify_risks(self, rule_name: str, context: Dict[str, Any]) -> List[str]:
        """Identify risks associated with recommendation"""
        risks_map = {
            "microservices_adoption": [
                "Increased operational complexity",
                "Network latency overhead",
                "Data consistency challenges"
            ],
            "event_sourcing": [
                "Storage requirements growth",
                "Event schema evolution",
                "Complexity for simple domains"
            ],
            "api_gateway": [
                "Single point of failure",
                "Additional latency",
                "Gateway becoming a monolith"
            ]
        }
        return risks_map.get(rule_name, [])
    
    def _impact_score(self, impact: str) -> int:
        """Convert impact string to numeric score"""
        return {"high": 3, "medium": 2, "low": 1}.get(impact, 0)
    
    async def evaluate_recommendation_success(
        self,
        recommendation: Recommendation,
        implementation_result: Dict[str, Any]
    ) -> float:
        """
        Evaluate success of an implemented recommendation.
        Used for learning and improving future recommendations.
        """
        success_score = 0.5  # Base score
        
        # Check if goals were met
        if implementation_result.get('goals_met', False):
            success_score += 0.3
        
        # Check if within estimated effort
        if implementation_result.get('effort_accurate', False):
            success_score += 0.1
        
        # Check if risks were managed
        if implementation_result.get('risks_managed', False):
            success_score += 0.1
        
        return success_score