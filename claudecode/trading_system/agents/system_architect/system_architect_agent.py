"""
System Architect Agent - Master Coordinator

This agent orchestrates all architectural concerns including SOLID principles,
design patterns, complexity management, security, performance, and documentation.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

from ..base.agent import BaseAgent
from .solid_principles_agent import SOLIDPrinciplesAgent
from .design_patterns_agent import DesignPatternsAgent
from .complexity_analyzer import ComplexityAnalyzer
from .security_audit_agent import SecurityAuditAgent
from .performance_audit_agent import PerformanceAuditAgent
from .prd_generator import PRDGenerator
from .adr_manager import ADRManager
from .documentation_agent import DocumentationAgent
from .architecture_diagram_manager import ArchitectureDiagramManager


@dataclass
class ArchitecturalAssessment:
    """Comprehensive architectural assessment results"""
    solid_score: float
    complexity_score: float
    security_score: float
    performance_score: float
    documentation_score: float
    overall_score: float
    recommendations: List[str]
    critical_issues: List[str]
    refactoring_priorities: List[Dict[str, Any]]


class SystemArchitectAgent(BaseAgent):
    """
    Master System Architect Agent
    
    Coordinates all architectural concerns and provides comprehensive
    system analysis, refactoring recommendations, and documentation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SystemArchitect", config)
        self.project_root = Path(config.get('project_root', '.'))
        
        # Initialize specialized agents
        self.solid_agent = SOLIDPrinciplesAgent(config)
        self.patterns_agent = DesignPatternsAgent(config)
        self.complexity_analyzer = ComplexityAnalyzer(config)
        self.security_agent = SecurityAuditAgent(config)
        self.performance_agent = PerformanceAuditAgent(config)
        self.prd_generator = PRDGenerator(config)
        self.adr_manager = ADRManager(config)
        self.docs_agent = DocumentationAgent(config)
        self.diagram_manager = ArchitectureDiagramManager(config)
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze_system_architecture(self, target_path: Optional[str] = None) -> ArchitecturalAssessment:
        """
        Perform comprehensive architectural analysis of the system
        
        Args:
            target_path: Specific path to analyze (defaults to entire project)
            
        Returns:
            ArchitecturalAssessment with comprehensive results
        """
        analysis_path = Path(target_path) if target_path else self.project_root
        self.logger.info(f"Starting comprehensive architectural analysis of {analysis_path}")
        
        # Run all analyses in parallel
        solid_results = await self.solid_agent.analyze_solid_compliance(str(analysis_path))
        complexity_results = await self.complexity_analyzer.analyze_complexity(str(analysis_path))
        security_results = await self.security_agent.audit_security(str(analysis_path))
        performance_results = await self.performance_agent.analyze_performance(str(analysis_path))
        docs_results = await self.docs_agent.assess_documentation_quality(str(analysis_path))
        
        # Calculate overall scores
        solid_score = solid_results.get('overall_score', 0.0)
        complexity_score = complexity_results.get('overall_score', 0.0)
        security_score = security_results.get('overall_score', 0.0)
        performance_score = performance_results.get('overall_score', 0.0)
        documentation_score = docs_results.get('overall_score', 0.0)
        
        overall_score = (
            solid_score * 0.25 +
            complexity_score * 0.20 +
            security_score * 0.25 +
            performance_score * 0.15 +
            documentation_score * 0.15
        )
        
        # Aggregate recommendations and issues
        recommendations = []
        critical_issues = []
        refactoring_priorities = []
        
        for result in [solid_results, complexity_results, security_results, 
                      performance_results, docs_results]:
            recommendations.extend(result.get('recommendations', []))
            critical_issues.extend(result.get('critical_issues', []))
            refactoring_priorities.extend(result.get('refactoring_priorities', []))
        
        # Prioritize refactoring based on impact and effort
        refactoring_priorities.sort(key=lambda x: (x.get('impact', 0), -x.get('effort', 0)), reverse=True)
        
        assessment = ArchitecturalAssessment(
            solid_score=solid_score,
            complexity_score=complexity_score,
            security_score=security_score,
            performance_score=performance_score,
            documentation_score=documentation_score,
            overall_score=overall_score,
            recommendations=recommendations[:10],  # Top 10 recommendations
            critical_issues=critical_issues,
            refactoring_priorities=refactoring_priorities[:5]  # Top 5 priorities
        )
        
        await self._save_assessment_report(assessment)
        return assessment
    
    async def create_architecture_documentation(self, component: str) -> Dict[str, Any]:
        """
        Create comprehensive architecture documentation for a component
        
        Args:
            component: Component name or path to document
            
        Returns:
            Documentation artifacts created
        """
        self.logger.info(f"Creating architecture documentation for {component}")
        
        artifacts = {}
        
        # Generate PRD
        prd = await self.prd_generator.generate_prd(component)
        artifacts['prd'] = prd
        
        # Create architectural diagrams
        diagrams = await self.diagram_manager.generate_component_diagrams(component)
        artifacts['diagrams'] = diagrams
        
        # Generate API documentation
        api_docs = await self.docs_agent.generate_api_documentation(component)
        artifacts['api_docs'] = api_docs
        
        # Create design pattern documentation
        pattern_docs = await self.patterns_agent.document_patterns_used(component)
        artifacts['pattern_docs'] = pattern_docs
        
        return artifacts
    
    async def execute_refactoring_plan(self, assessment: ArchitecturalAssessment) -> Dict[str, Any]:
        """
        Execute a prioritized refactoring plan based on assessment
        
        Args:
            assessment: Architectural assessment with refactoring priorities
            
        Returns:
            Refactoring execution results
        """
        self.logger.info("Executing prioritized refactoring plan")
        
        results = {
            'completed_tasks': [],
            'failed_tasks': [],
            'improvements': {}
        }
        
        # Execute high-priority refactoring tasks
        for priority in assessment.refactoring_priorities:
            task_type = priority.get('type')
            target = priority.get('target')
            
            try:
                if task_type == 'solid_violation':
                    result = await self.solid_agent.fix_solid_violation(target, priority)
                elif task_type == 'complexity_reduction':
                    result = await self.complexity_analyzer.reduce_complexity(target, priority)
                elif task_type == 'security_fix':
                    result = await self.security_agent.fix_security_issue(target, priority)
                elif task_type == 'performance_optimization':
                    result = await self.performance_agent.optimize_performance(target, priority)
                elif task_type == 'pattern_application':
                    result = await self.patterns_agent.apply_design_pattern(target, priority)
                
                results['completed_tasks'].append({
                    'task': priority,
                    'result': result
                })
                
            except Exception as e:
                self.logger.error(f"Failed to execute refactoring task {priority}: {e}")
                results['failed_tasks'].append({
                    'task': priority,
                    'error': str(e)
                })
        
        # Re-analyze to measure improvements
        post_refactor_assessment = await self.analyze_system_architecture()
        results['improvements'] = {
            'solid_improvement': post_refactor_assessment.solid_score - assessment.solid_score,
            'complexity_improvement': post_refactor_assessment.complexity_score - assessment.complexity_score,
            'security_improvement': post_refactor_assessment.security_score - assessment.security_score,
            'performance_improvement': post_refactor_assessment.performance_score - assessment.performance_score,
            'overall_improvement': post_refactor_assessment.overall_score - assessment.overall_score
        }
        
        return results
    
    async def create_adr(self, decision_title: str, context: str, decision: str, 
                        consequences: str) -> str:
        """
        Create an Architecture Decision Record
        
        Args:
            decision_title: Title of the architectural decision
            context: Context and background
            decision: The decision made
            consequences: Consequences and trade-offs
            
        Returns:
            Path to created ADR
        """
        return await self.adr_manager.create_adr(
            decision_title, context, decision, consequences
        )
    
    async def update_system_diagrams(self) -> Dict[str, str]:
        """
        Update all system architecture diagrams
        
        Returns:
            Paths to updated diagrams
        """
        return await self.diagram_manager.update_all_diagrams()
    
    async def generate_architecture_report(self) -> str:
        """
        Generate comprehensive architecture health report
        
        Returns:
            Path to generated report
        """
        assessment = await self.analyze_system_architecture()
        
        report_content = f"""
# System Architecture Health Report

## Executive Summary
- **Overall Architecture Score**: {assessment.overall_score:.2f}/10.0
- **Critical Issues**: {len(assessment.critical_issues)}
- **Refactoring Priorities**: {len(assessment.refactoring_priorities)}

## Detailed Scores
- **SOLID Principles Compliance**: {assessment.solid_score:.2f}/10.0
- **Complexity Management**: {assessment.complexity_score:.2f}/10.0
- **Security Posture**: {assessment.security_score:.2f}/10.0
- **Performance Efficiency**: {assessment.performance_score:.2f}/10.0
- **Documentation Quality**: {assessment.documentation_score:.2f}/10.0

## Critical Issues
{chr(10).join(f"- {issue}" for issue in assessment.critical_issues)}

## Top Recommendations
{chr(10).join(f"- {rec}" for rec in assessment.recommendations)}

## Refactoring Priorities
{chr(10).join(f"- {priority.get('description', 'N/A')} (Impact: {priority.get('impact', 'N/A')}, Effort: {priority.get('effort', 'N/A')})" for priority in assessment.refactoring_priorities)}

---
*Generated by System Architect Agent*
"""
        
        report_path = self.project_root / "docs" / "architecture_health_report.md"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Architecture health report saved to {report_path}")
        return str(report_path)
    
    async def _save_assessment_report(self, assessment: ArchitecturalAssessment) -> None:
        """Save detailed assessment results to JSON file"""
        assessment_data = {
            'solid_score': assessment.solid_score,
            'complexity_score': assessment.complexity_score,
            'security_score': assessment.security_score,
            'performance_score': assessment.performance_score,
            'documentation_score': assessment.documentation_score,
            'overall_score': assessment.overall_score,
            'recommendations': assessment.recommendations,
            'critical_issues': assessment.critical_issues,
            'refactoring_priorities': assessment.refactoring_priorities
        }
        
        assessment_path = self.project_root / "docs" / "architecture_assessment.json"
        assessment_path.parent.mkdir(exist_ok=True)
        
        with open(assessment_path, 'w') as f:
            json.dump(assessment_data, f, indent=2)
        
        self.logger.info(f"Detailed assessment saved to {assessment_path}")