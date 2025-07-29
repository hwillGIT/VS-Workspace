"""
Base Framework Interface
Abstract base class for all architecture framework implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class AnalysisDepth(Enum):
    """Depth levels for framework analysis"""
    BASIC = "basic"  # Quick overview, key concepts
    INTERMEDIATE = "intermediate"  # Standard analysis, common patterns
    EXPERT = "expert"  # Deep dive, all artifacts, advanced techniques


@dataclass
class FrameworkArtifact:
    """Represents an artifact produced by a framework"""
    id: str
    name: str
    type: str  # diagram, document, model, etc.
    format: str  # markdown, json, mermaid, plantuml, etc.
    content: Any
    metadata: Dict[str, Any]
    framework: str
    created_at: datetime
    relationships: List[str]  # IDs of related artifacts


@dataclass
class FrameworkAnalysis:
    """Result of framework-specific analysis"""
    framework_name: str
    framework_version: str
    analysis_depth: AnalysisDepth
    current_state: Dict[str, Any]
    target_state: Optional[Dict[str, Any]]
    gaps: List[Dict[str, Any]]
    findings: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    artifacts: List[FrameworkArtifact]
    patterns: List[Dict[str, Any]]
    anti_patterns: List[Dict[str, Any]]
    compliance: Dict[str, Any]
    metrics: Dict[str, Any]
    confidence_score: float
    analysis_metadata: Dict[str, Any]


class BaseFramework(ABC):
    """
    Abstract base class for architecture framework implementations
    All frameworks must implement these methods for deep expertise
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.get_framework_name()
        self.version = self.get_framework_version()
        self.capabilities = self.get_capabilities()
        self.artifacts_registry = {}
        self.patterns_catalog = {}
        self.techniques_registry = {}
        
    @abstractmethod
    def get_framework_name(self) -> str:
        """Return the official name of the framework"""
        pass
    
    @abstractmethod
    def get_framework_version(self) -> str:
        """Return the version of the framework being implemented"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this framework provides"""
        pass
    
    @abstractmethod
    async def analyze(
        self,
        context: Dict[str, Any],
        depth: AnalysisDepth = AnalysisDepth.INTERMEDIATE
    ) -> FrameworkAnalysis:
        """
        Perform framework-specific analysis
        
        Args:
            context: Architecture context and requirements
            depth: Analysis depth level
            
        Returns:
            Framework-specific analysis results
        """
        pass
    
    @abstractmethod
    async def assess_current_state(
        self,
        context: Dict[str, Any],
        artifacts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Assess the current state using framework methodology
        
        Args:
            context: Current architecture context
            artifacts: Existing architecture artifacts
            
        Returns:
            Current state assessment
        """
        pass
    
    @abstractmethod
    async def define_target_state(
        self,
        context: Dict[str, Any],
        current_state: Dict[str, Any],
        goals: List[str]
    ) -> Dict[str, Any]:
        """
        Define target state using framework methodology
        
        Args:
            context: Architecture context
            current_state: Current state assessment
            goals: Business and technical goals
            
        Returns:
            Target state definition
        """
        pass
    
    @abstractmethod
    async def perform_gap_analysis(
        self,
        current_state: Dict[str, Any],
        target_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Perform gap analysis between current and target states
        
        Args:
            current_state: Current architecture state
            target_state: Target architecture state
            
        Returns:
            List of identified gaps
        """
        pass
    
    @abstractmethod
    async def generate_artifacts(
        self,
        analysis: FrameworkAnalysis,
        artifact_types: Optional[List[str]] = None
    ) -> List[FrameworkArtifact]:
        """
        Generate framework-specific artifacts
        
        Args:
            analysis: Framework analysis results
            artifact_types: Specific artifact types to generate (None = all)
            
        Returns:
            List of generated artifacts
        """
        pass
    
    @abstractmethod
    async def detect_patterns(
        self,
        context: Dict[str, Any],
        scope: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect architectural patterns using framework perspective
        
        Args:
            context: Architecture context
            scope: Specific scope for pattern detection
            
        Returns:
            List of detected patterns
        """
        pass
    
    @abstractmethod
    async def detect_anti_patterns(
        self,
        context: Dict[str, Any],
        scope: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect anti-patterns and architectural smells
        
        Args:
            context: Architecture context
            scope: Specific scope for anti-pattern detection
            
        Returns:
            List of detected anti-patterns
        """
        pass
    
    @abstractmethod
    async def generate_recommendations(
        self,
        analysis: FrameworkAnalysis,
        priorities: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate framework-specific recommendations
        
        Args:
            analysis: Framework analysis results
            priorities: Priority areas for recommendations
            
        Returns:
            List of recommendations
        """
        pass
    
    @abstractmethod
    async def check_compliance(
        self,
        context: Dict[str, Any],
        standards: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Check compliance with framework standards and best practices
        
        Args:
            context: Architecture context
            standards: Specific standards to check against
            
        Returns:
            Compliance check results
        """
        pass
    
    @abstractmethod
    async def create_roadmap(
        self,
        current_state: Dict[str, Any],
        target_state: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create implementation roadmap using framework methodology
        
        Args:
            current_state: Current architecture state
            target_state: Target architecture state
            constraints: Time, budget, resource constraints
            
        Returns:
            Implementation roadmap
        """
        pass
    
    # Common helper methods that frameworks can override or extend
    
    async def validate_context(self, context: Dict[str, Any]) -> bool:
        """Validate that context has required information for framework"""
        required_fields = self.get_required_context_fields()
        for field in required_fields:
            if field not in context:
                raise ValueError(f"Missing required context field: {field}")
        return True
    
    def get_required_context_fields(self) -> List[str]:
        """Return list of required context fields for this framework"""
        return ["project_name", "domain"]
    
    async def get_artifact_templates(self) -> Dict[str, Any]:
        """Return available artifact templates for this framework"""
        return self.artifacts_registry
    
    async def get_pattern_catalog(self) -> Dict[str, Any]:
        """Return the pattern catalog for this framework"""
        return self.patterns_catalog
    
    async def get_techniques(self) -> Dict[str, Any]:
        """Return available techniques and methods for this framework"""
        return self.techniques_registry
    
    def calculate_confidence_score(
        self,
        analysis_completeness: float,
        data_quality: float,
        pattern_matches: int
    ) -> float:
        """Calculate confidence score for analysis results"""
        # Base implementation - frameworks can override
        weights = {
            "completeness": 0.4,
            "quality": 0.3,
            "patterns": 0.3
        }
        
        pattern_score = min(pattern_matches / 10, 1.0)  # Normalize to 0-1
        
        score = (
            weights["completeness"] * analysis_completeness +
            weights["quality"] * data_quality +
            weights["patterns"] * pattern_score
        )
        
        return round(score, 2)
    
    def format_artifact(
        self,
        content: Any,
        format: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FrameworkArtifact:
        """Helper to create properly formatted artifacts"""
        artifact_id = f"{self.name}_{datetime.now().timestamp()}"
        
        return FrameworkArtifact(
            id=artifact_id,
            name=metadata.get("name", "Unnamed Artifact") if metadata else "Unnamed Artifact",
            type=metadata.get("type", "document") if metadata else "document",
            format=format,
            content=content,
            metadata=metadata or {},
            framework=self.name,
            created_at=datetime.now(),
            relationships=metadata.get("relationships", []) if metadata else []
        )
    
    async def export_analysis(
        self,
        analysis: FrameworkAnalysis,
        format: str = "json"
    ) -> str:
        """Export analysis results in specified format"""
        if format == "json":
            import json
            return json.dumps(analysis.__dict__, indent=2, default=str)
        elif format == "markdown":
            return self._export_as_markdown(analysis)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_as_markdown(self, analysis: FrameworkAnalysis) -> str:
        """Export analysis as markdown"""
        md = f"# {analysis.framework_name} Analysis\n\n"
        md += f"**Version:** {analysis.framework_version}\n"
        md += f"**Depth:** {analysis.analysis_depth.value}\n"
        md += f"**Confidence:** {analysis.confidence_score}\n\n"
        
        if analysis.findings:
            md += "## Key Findings\n\n"
            for finding in analysis.findings[:5]:
                md += f"- {finding.get('title', 'Finding')}: {finding.get('description', '')}\n"
            md += "\n"
        
        if analysis.recommendations:
            md += "## Recommendations\n\n"
            for i, rec in enumerate(analysis.recommendations[:5], 1):
                md += f"{i}. **{rec.get('title', 'Recommendation')}**\n"
                md += f"   {rec.get('description', '')}\n\n"
        
        if analysis.patterns:
            md += "## Patterns Identified\n\n"
            for pattern in analysis.patterns:
                md += f"- **{pattern.get('name', 'Pattern')}**: {pattern.get('description', '')}\n"
        
        return md


class FrameworkRegistry:
    """Registry for managing framework implementations"""
    
    def __init__(self):
        self.frameworks: Dict[str, BaseFramework] = {}
        
    def register(self, framework_name: str, framework_class: type):
        """Register a framework implementation"""
        if not issubclass(framework_class, BaseFramework):
            raise ValueError(f"{framework_class} must inherit from BaseFramework")
        
        self.frameworks[framework_name] = framework_class
        
    def get(self, framework_name: str) -> Optional[BaseFramework]:
        """Get a framework implementation"""
        framework_class = self.frameworks.get(framework_name)
        if framework_class:
            return framework_class()
        return None
    
    def list_frameworks(self) -> List[str]:
        """List all registered frameworks"""
        return list(self.frameworks.keys())
    
    def get_all_capabilities(self) -> Dict[str, List[str]]:
        """Get capabilities of all registered frameworks"""
        capabilities = {}
        for name, framework_class in self.frameworks.items():
            framework = framework_class()
            capabilities[name] = framework.get_capabilities()
        return capabilities