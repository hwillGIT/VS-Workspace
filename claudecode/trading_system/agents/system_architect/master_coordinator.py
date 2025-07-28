"""
Master Coordinator for System Architect Suite

Orchestrates all architect agents and provides unified interface for comprehensive
system analysis, planning, and optimization.
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import time

from .architecture_diagram_manager import ArchitectureDiagramManager
from .dependency_analysis_agent import DependencyAnalysisAgent
from .code_metrics_dashboard import CodeMetricsDashboard
from .migration_planning_agent import MigrationPlanningAgent
from .system_architect_agent import SystemArchitectAgent
from .solid_principles_agent import SOLIDPrinciplesAgent
from .design_patterns_agent import DesignPatternsAgent
from .complexity_analyzer import ComplexityAnalyzer
from .performance_audit_agent import PerformanceAuditAgent
from .security_audit_agent import SecurityAuditAgent
from .adr_manager import ADRManager
from .documentation_agent import DocumentationAgent
from ...core.base.agent import BaseAgent


@dataclass
class AnalysisSession:
    """Represents a complete analysis session"""
    session_id: str
    project_path: str
    timestamp: datetime
    config: Dict[str, Any]
    results: Dict[str, Any] = field(default_factory=dict)
    execution_times: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class ArchitectureInsight:
    """High-level architectural insight derived from analysis"""
    category: str  # 'complexity', 'dependency', 'security', 'performance', 'quality'
    severity: str  # 'info', 'warning', 'critical'
    title: str
    description: str
    evidence: List[str]
    recommendations: List[str]
    affected_components: List[str]
    confidence: float  # 0.0 to 1.0


@dataclass
class SystemHealthReport:
    """Overall system health assessment"""
    overall_score: float  # 0-100
    health_status: str  # 'excellent', 'good', 'fair', 'poor', 'critical'
    key_strengths: List[str]
    critical_issues: List[str]
    improvement_priorities: List[Dict[str, Any]]
    trend_analysis: Dict[str, str]
    benchmark_comparison: Dict[str, Any]


class MasterCoordinator(BaseAgent):
    """
    Master Coordinator for System Architect Suite
    
    Orchestrates all architect agents to provide:
    - Unified system analysis
    - Cross-agent data correlation
    - Prioritized recommendations
    - Comprehensive reporting
    - Session management
    - Performance optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MasterCoordinator", config.get('coordinator', {}))
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.enable_parallel_execution = config.get('enable_parallel_execution', True)
        self.cache_results = config.get('cache_results', True)
        self.max_concurrent_agents = config.get('max_concurrent_agents', 4)
        self.cross_validation_enabled = config.get('cross_validation', True)
        
        # Initialize all agents
        self.agents = self._initialize_agents(config)
        
        # Session management
        self.active_sessions: Dict[str, AnalysisSession] = {}
        self.result_cache: Dict[str, Dict[str, Any]] = {}
        
        # Analysis weights for scoring
        self.analysis_weights = {
            'complexity': 0.20,
            'dependencies': 0.15,
            'security': 0.25,
            'performance': 0.20,
            'quality': 0.20
        }
    
    def _initialize_agents(self, config: Dict[str, Any]) -> Dict[str, BaseAgent]:
        """Initialize all architect agents"""
        agents = {}
        
        try:
            # Core analysis agents
            agents['system_architect'] = SystemArchitectAgent(config)
            agents['architecture_diagram'] = ArchitectureDiagramManager(config)
            agents['dependency_analysis'] = DependencyAnalysisAgent(config)
            agents['code_metrics'] = CodeMetricsDashboard(config)
            agents['migration_planning'] = MigrationPlanningAgent(config)
            
            # Specialized analysis agents
            agents['solid_principles'] = SOLIDPrinciplesAgent(config)
            agents['design_patterns'] = DesignPatternsAgent(config)
            agents['complexity_analyzer'] = ComplexityAnalyzer(config)
            agents['performance_audit'] = PerformanceAuditAgent(config)
            agents['security_audit'] = SecurityAuditAgent(config)
            
            # Documentation and decision agents
            agents['adr_manager'] = ADRManager(config)
            agents['documentation'] = DocumentationAgent(config)
            
            self.logger.info(f"Initialized {len(agents)} architect agents")
            return agents
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            return {}
    
    async def analyze_system(self, project_path: str, analysis_scope: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform comprehensive system analysis
        
        Args:
            project_path: Path to the project to analyze
            analysis_scope: 'quick', 'standard', 'comprehensive', 'deep'
            
        Returns:
            Complete analysis results with insights and recommendations
        """
        session_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        session = AnalysisSession(
            session_id=session_id,
            project_path=project_path,
            timestamp=datetime.utcnow(),
            config={'analysis_scope': analysis_scope}
        )
        
        self.active_sessions[session_id] = session
        session.status = "running"
        
        try:
            self.logger.info(f"Starting system analysis: {session_id}")
            
            # Determine analysis pipeline based on scope
            pipeline = self._get_analysis_pipeline(analysis_scope)
            
            # Execute analysis pipeline
            results = await self._execute_analysis_pipeline(pipeline, project_path, session)
            
            # Cross-validate results if enabled
            if self.cross_validation_enabled:
                results = await self._cross_validate_results(results, session)
            
            # Generate insights
            insights = await self._generate_insights(results, session)
            
            # Create health report
            health_report = await self._generate_health_report(results, insights, session)
            
            # Compile final results
            final_results = {
                'session_id': session_id,
                'analysis_scope': analysis_scope,
                'project_path': project_path,
                'timestamp': session.timestamp.isoformat(),
                'execution_times': session.execution_times,
                'raw_results': results,
                'insights': [self._insight_to_dict(i) for i in insights],
                'health_report': self._health_report_to_dict(health_report),
                'recommendations': await self._generate_unified_recommendations(results, insights),
                'summary': await self._generate_executive_summary(health_report, insights),
                'next_steps': await self._generate_next_steps(health_report, insights),
                'metadata': {
                    'agents_used': list(results.keys()),
                    'total_execution_time': sum(session.execution_times.values()),
                    'errors': session.errors,
                    'warnings': session.warnings
                }
            }
            
            session.results = final_results
            session.status = "completed"
            
            self.logger.info(f"Completed system analysis: {session_id}")
            return final_results
            
        except Exception as e:
            session.status = "failed"
            session.errors.append(str(e))
            self.logger.error(f"Analysis failed: {e}")
            raise
    
    def _get_analysis_pipeline(self, scope: str) -> List[str]:
        """Get analysis pipeline based on scope"""
        pipelines = {
            'quick': [
                'code_metrics',
                'dependency_analysis'
            ],
            'standard': [
                'code_metrics',
                'dependency_analysis',
                'security_audit',
                'complexity_analyzer'
            ],
            'comprehensive': [
                'system_architect',
                'code_metrics',
                'dependency_analysis',
                'architecture_diagram',
                'security_audit',
                'performance_audit',
                'solid_principles',
                'design_patterns',
                'complexity_analyzer'
            ],
            'deep': [
                'system_architect',
                'code_metrics',
                'dependency_analysis',
                'architecture_diagram',
                'migration_planning',
                'security_audit',
                'performance_audit',
                'solid_principles',
                'design_patterns',
                'complexity_analyzer',
                'adr_manager',
                'documentation'
            ]
        }
        
        return pipelines.get(scope, pipelines['standard'])
    
    async def _execute_analysis_pipeline(self, pipeline: List[str], 
                                        project_path: str, 
                                        session: AnalysisSession) -> Dict[str, Any]:
        """Execute the analysis pipeline"""
        results = {}
        
        if self.enable_parallel_execution:
            # Group agents by dependency requirements
            independent_agents = ['code_metrics', 'security_audit', 'performance_audit', 'complexity_analyzer']
            dependent_agents = ['architecture_diagram', 'migration_planning', 'system_architect']
            
            # Execute independent agents in parallel
            independent_tasks = []
            for agent_name in pipeline:
                if agent_name in independent_agents and agent_name in self.agents:
                    task = self._execute_agent_analysis(agent_name, project_path, session)
                    independent_tasks.append((agent_name, task))
            
            # Execute independent analyses
            if independent_tasks:
                independent_results = await asyncio.gather(
                    *[task for _, task in independent_tasks],
                    return_exceptions=True
                )
                
                for (agent_name, _), result in zip(independent_tasks, independent_results):
                    if isinstance(result, Exception):
                        session.errors.append(f"{agent_name}: {str(result)}")
                        self.logger.error(f"Agent {agent_name} failed: {result}")
                    else:
                        results[agent_name] = result
            
            # Execute dependent agents sequentially
            for agent_name in pipeline:
                if agent_name in dependent_agents and agent_name in self.agents:
                    try:
                        result = await self._execute_agent_analysis(agent_name, project_path, session, results)
                        results[agent_name] = result
                    except Exception as e:
                        session.errors.append(f"{agent_name}: {str(e)}")
                        self.logger.error(f"Agent {agent_name} failed: {e}")
        
        else:
            # Sequential execution
            for agent_name in pipeline:
                if agent_name in self.agents:
                    try:
                        result = await self._execute_agent_analysis(agent_name, project_path, session, results)
                        results[agent_name] = result
                    except Exception as e:
                        session.errors.append(f"{agent_name}: {str(e)}")
                        self.logger.error(f"Agent {agent_name} failed: {e}")
        
        return results
    
    async def _execute_agent_analysis(self, agent_name: str, project_path: str, 
                                     session: AnalysisSession, 
                                     previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute analysis for a specific agent"""
        start_time = time.time()
        
        try:
            agent = self.agents[agent_name]
            
            # Check cache first
            cache_key = f"{agent_name}_{project_path}_{hash(str(session.config))}"
            if self.cache_results and cache_key in self.result_cache:
                self.logger.info(f"Using cached result for {agent_name}")
                return self.result_cache[cache_key]
            
            # Execute agent-specific analysis
            if agent_name == 'system_architect':
                result = await agent.assess_system_architecture(project_path)
            elif agent_name == 'architecture_diagram':
                result = await agent.generate_architecture_diagrams(project_path)
            elif agent_name == 'dependency_analysis':
                result = await agent.analyze_dependencies(project_path)
            elif agent_name == 'code_metrics':
                result = await agent.generate_dashboard(project_path)
            elif agent_name == 'migration_planning':
                # Use previous results to inform migration planning
                source_config = {'project_path': project_path}
                if previous_results and 'code_metrics' in previous_results:
                    metrics = previous_results['code_metrics']
                    source_config['current_complexity'] = metrics.get('project_metrics', {}).get('overall_complexity', 0)
                
                target_config = {'python_version': '3.11.5'}  # Default target
                result = await agent.create_migration_plan('version_upgrade', source_config, target_config)
            elif agent_name == 'security_audit':
                result = await agent.audit_security(project_path)
            elif agent_name == 'performance_audit':
                result = await agent.analyze_performance(project_path)
            elif agent_name == 'solid_principles':
                result = await agent.analyze_solid_compliance(project_path)
            elif agent_name == 'design_patterns':
                result = await agent.identify_patterns(project_path)
            elif agent_name == 'complexity_analyzer':
                result = await agent.analyze_complexity(project_path)
            elif agent_name == 'adr_manager':
                result = await agent.generate_adr_analysis(project_path)
            elif agent_name == 'documentation':
                result = await agent.analyze_documentation(project_path)
            else:
                raise ValueError(f"Unknown agent: {agent_name}")
            
            # Cache result
            if self.cache_results:
                self.result_cache[cache_key] = result
            
            return result
            
        finally:
            execution_time = time.time() - start_time
            session.execution_times[agent_name] = execution_time
            self.logger.info(f"Agent {agent_name} completed in {execution_time:.2f}s")
    
    async def _cross_validate_results(self, results: Dict[str, Any], 
                                     session: AnalysisSession) -> Dict[str, Any]:
        """Cross-validate results between agents"""
        # Complexity validation
        if 'code_metrics' in results and 'complexity_analyzer' in results:
            metrics_complexity = results['code_metrics'].get('project_metrics', {}).get('overall_complexity', 0)
            analyzer_complexity = results['complexity_analyzer'].get('overall_complexity', 0)
            
            if abs(metrics_complexity - analyzer_complexity) > 5:
                session.warnings.append("Complexity measurements differ significantly between agents")
        
        # Security validation
        if 'security_audit' in results and 'code_metrics' in results:
            security_hotspots = results['security_audit'].get('vulnerabilities', [])
            metrics_hotspots = sum(f.get('security_hotspots', 0) for f in results['code_metrics'].get('file_metrics', []))
            
            if len(security_hotspots) != metrics_hotspots:
                session.warnings.append("Security hotspot counts differ between agents")
        
        # Dependency validation
        if 'dependency_analysis' in results and 'architecture_diagram' in results:
            dep_nodes = len(results['dependency_analysis'].get('dependency_graph', {}).get('nodes', []))
            arch_components = len(results['architecture_diagram'].get('components', []))
            
            # Should be roughly similar (allowing for different granularity)
            if abs(dep_nodes - arch_components) > max(dep_nodes, arch_components) * 0.5:
                session.warnings.append("Significant difference in component/dependency counts")
        
        return results
    
    async def _generate_insights(self, results: Dict[str, Any], 
                                session: AnalysisSession) -> List[ArchitectureInsight]:
        """Generate high-level insights from analysis results"""
        insights = []
        
        # Complexity insights
        if 'code_metrics' in results:
            metrics = results['code_metrics']
            overall_complexity = metrics.get('project_metrics', {}).get('overall_complexity', 0)
            
            if overall_complexity > 15:
                insights.append(ArchitectureInsight(
                    category='complexity',
                    severity='critical',
                    title='High System Complexity',
                    description=f'Overall system complexity ({overall_complexity:.1f}) exceeds recommended thresholds',
                    evidence=[
                        f"Average cyclomatic complexity: {overall_complexity:.1f}",
                        f"Files with high complexity: {len([f for f in metrics.get('file_metrics', []) if f.get('cyclomatic_complexity', 0) > 15])}"
                    ],
                    recommendations=[
                        'Refactor complex functions using Extract Method pattern',
                        'Consider breaking large classes into smaller, focused classes',
                        'Implement complexity monitoring in CI/CD pipeline'
                    ],
                    affected_components=[f['file_path'] for f in metrics.get('file_metrics', []) if f.get('cyclomatic_complexity', 0) > 15],
                    confidence=0.9
                ))
        
        # Security insights
        if 'security_audit' in results:
            security = results['security_audit']
            critical_vulns = [v for v in security.get('vulnerabilities', []) if v.get('severity') == 'critical']
            
            if critical_vulns:
                insights.append(ArchitectureInsight(
                    category='security',
                    severity='critical',
                    title='Critical Security Vulnerabilities',
                    description=f'Found {len(critical_vulns)} critical security vulnerabilities',
                    evidence=[v.get('description', 'Unknown vulnerability') for v in critical_vulns[:3]],
                    recommendations=[
                        'Address critical vulnerabilities immediately',
                        'Implement security code review process',
                        'Add automated security scanning to CI/CD'
                    ],
                    affected_components=[v.get('file_path', 'Unknown') for v in critical_vulns],
                    confidence=0.95
                ))
        
        # Dependency insights
        if 'dependency_analysis' in results:
            deps = results['dependency_analysis']
            circular_deps = deps.get('circular_dependencies', [])
            
            if circular_deps:
                critical_cycles = [c for c in circular_deps if c.get('severity') in ['high', 'critical']]
                if critical_cycles:
                    insights.append(ArchitectureInsight(
                        category='dependency',
                        severity='warning',
                        title='Circular Dependencies Detected',
                        description=f'Found {len(circular_deps)} circular dependencies, {len(critical_cycles)} are critical',
                        evidence=[f"Cycle: {' -> '.join(c.get('nodes', [])[:3])}" for c in critical_cycles[:3]],
                        recommendations=[
                            'Break circular dependencies using dependency injection',
                            'Extract shared functionality to common modules',
                            'Use interfaces to decouple components'
                        ],
                        affected_components=[node for c in critical_cycles for node in c.get('nodes', [])],
                        confidence=0.85
                    ))
        
        # Performance insights
        if 'performance_audit' in results:
            perf = results['performance_audit']
            critical_issues = perf.get('critical_issues', [])
            
            if critical_issues:
                insights.append(ArchitectureInsight(
                    category='performance',
                    severity='warning',
                    title='Performance Bottlenecks Identified',
                    description=f'Identified {len(critical_issues)} critical performance issues',
                    evidence=[issue.get('description', 'Unknown issue') for issue in critical_issues[:3]],
                    recommendations=[
                        'Optimize algorithmic complexity in identified functions',
                        'Implement caching for expensive operations',
                        'Consider async/await for I/O operations'
                    ],
                    affected_components=[issue.get('file_path', 'Unknown') for issue in critical_issues],
                    confidence=0.8
                ))
        
        # Design quality insights
        if 'solid_principles' in results:
            solid = results['solid_principles']
            violations = solid.get('violations', [])
            serious_violations = [v for v in violations if v.get('severity') in ['high', 'critical']]
            
            if serious_violations:
                insights.append(ArchitectureInsight(
                    category='quality',
                    severity='warning',
                    title='SOLID Principle Violations',
                    description=f'Found {len(serious_violations)} serious SOLID principle violations',
                    evidence=[v.get('description', 'Unknown violation') for v in serious_violations[:3]],
                    recommendations=[
                        'Refactor classes to follow Single Responsibility Principle',
                        'Use dependency injection to reduce coupling',
                        'Extract interfaces to improve flexibility'
                    ],
                    affected_components=[v.get('file_path', 'Unknown') for v in serious_violations],
                    confidence=0.75
                ))
        
        return insights
    
    async def _generate_health_report(self, results: Dict[str, Any], 
                                     insights: List[ArchitectureInsight],
                                     session: AnalysisSession) -> SystemHealthReport:
        """Generate overall system health report"""
        # Calculate overall score
        scores = {}
        
        # Complexity score (inverse of complexity)
        if 'code_metrics' in results:
            complexity = results['code_metrics'].get('project_metrics', {}).get('overall_complexity', 10)
            scores['complexity'] = max(0, 100 - (complexity - 5) * 10)  # Scale around baseline of 5
        
        # Security score
        if 'security_audit' in results:
            vulnerabilities = results['security_audit'].get('vulnerabilities', [])
            critical_count = len([v for v in vulnerabilities if v.get('severity') == 'critical'])
            high_count = len([v for v in vulnerabilities if v.get('severity') == 'high'])
            scores['security'] = max(0, 100 - critical_count * 20 - high_count * 10)
        
        # Performance score
        if 'performance_audit' in results:
            perf_score = results['performance_audit'].get('overall_score', 5)
            scores['performance'] = perf_score * 10  # Convert to 0-100 scale
        
        # Quality score
        if 'solid_principles' in results:
            compliance_score = results['solid_principles'].get('overall_compliance', 70)
            scores['quality'] = compliance_score
        
        # Dependency score
        if 'dependency_analysis' in results:
            circular_deps = len(results['dependency_analysis'].get('circular_dependencies', []))
            scores['dependencies'] = max(0, 100 - circular_deps * 15)
        
        # Calculate weighted overall score
        overall_score = 0
        total_weight = 0
        for category, score in scores.items():
            weight = self.analysis_weights.get(category, 0.2)
            overall_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score = overall_score / total_weight
        else:
            overall_score = 50  # Default neutral score
        
        # Determine health status
        if overall_score >= 90:
            health_status = 'excellent'
        elif overall_score >= 75:
            health_status = 'good'
        elif overall_score >= 60:
            health_status = 'fair'
        elif overall_score >= 40:
            health_status = 'poor'
        else:
            health_status = 'critical'
        
        # Identify strengths and issues
        key_strengths = []
        critical_issues = []
        
        for category, score in scores.items():
            if score >= 80:
                key_strengths.append(f"Strong {category} metrics (score: {score:.1f})")
            elif score < 40:
                critical_issues.append(f"Poor {category} metrics (score: {score:.1f})")
        
        # Add insight-based issues
        critical_insights = [i for i in insights if i.severity == 'critical']
        for insight in critical_insights:
            critical_issues.append(insight.title)
        
        # Generate improvement priorities
        improvement_priorities = []
        for insight in sorted(insights, key=lambda x: (x.severity == 'critical', x.confidence), reverse=True):
            if insight.severity in ['critical', 'warning']:
                improvement_priorities.append({
                    'category': insight.category,
                    'title': insight.title,
                    'severity': insight.severity,
                    'priority': 'high' if insight.severity == 'critical' else 'medium',
                    'effort': 'high' if len(insight.affected_components) > 10 else 'medium',
                    'impact': 'high' if insight.confidence > 0.8 else 'medium'
                })
        
        return SystemHealthReport(
            overall_score=overall_score,
            health_status=health_status,
            key_strengths=key_strengths[:5],  # Top 5 strengths
            critical_issues=critical_issues[:5],  # Top 5 issues
            improvement_priorities=improvement_priorities[:10],  # Top 10 priorities
            trend_analysis={'overall': 'stable'},  # Would need historical data
            benchmark_comparison={'industry_average': 65.0}  # Would need benchmark data
        )
    
    async def _generate_unified_recommendations(self, results: Dict[str, Any], 
                                              insights: List[ArchitectureInsight]) -> List[Dict[str, Any]]:
        """Generate unified recommendations across all analyses"""
        recommendations = []
        
        # Collect all recommendations from insights
        for insight in insights:
            for rec in insight.recommendations:
                recommendations.append({
                    'category': insight.category,
                    'title': rec,
                    'severity': insight.severity,
                    'confidence': insight.confidence,
                    'affected_components': insight.affected_components[:5],  # Limit for readability
                    'source': 'insight_analysis'
                })
        
        # Add agent-specific recommendations
        for agent_name, result in results.items():
            if 'recommendations' in result and isinstance(result['recommendations'], list):
                for rec in result['recommendations']:
                    recommendations.append({
                        'category': 'general',
                        'title': rec if isinstance(rec, str) else rec.get('title', 'Unknown'),
                        'severity': 'info',
                        'confidence': 0.7,
                        'source': agent_name
                    })
        
        # Deduplicate and prioritize
        unique_recommendations = []
        seen_titles = set()
        
        for rec in sorted(recommendations, key=lambda x: (x['severity'] == 'critical', x['confidence']), reverse=True):
            title = rec['title'].lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_recommendations.append(rec)
                if len(unique_recommendations) >= 20:  # Limit to top 20
                    break
        
        return unique_recommendations
    
    async def _generate_executive_summary(self, health_report: SystemHealthReport, 
                                         insights: List[ArchitectureInsight]) -> Dict[str, Any]:
        """Generate executive summary"""
        critical_insights = [i for i in insights if i.severity == 'critical']
        warning_insights = [i for i in insights if i.severity == 'warning']
        
        return {
            'overall_health': {
                'score': health_report.overall_score,
                'status': health_report.health_status,
                'summary': f"System health is {health_report.health_status} with a score of {health_report.overall_score:.1f}/100"
            },
            'key_findings': {
                'critical_issues': len(critical_insights),
                'warnings': len(warning_insights),
                'strengths': len(health_report.key_strengths),
                'top_issue': critical_insights[0].title if critical_insights else 'No critical issues found'
            },
            'immediate_actions': [
                insight.recommendations[0] for insight in critical_insights[:3]
                if insight.recommendations
            ],
            'business_impact': {
                'risk_level': 'high' if len(critical_insights) > 0 else 'medium' if len(warning_insights) > 2 else 'low',
                'maintenance_burden': 'high' if health_report.overall_score < 60 else 'medium' if health_report.overall_score < 80 else 'low',
                'development_velocity': 'impacted' if len(critical_insights) > 1 else 'stable'
            }
        }
    
    async def _generate_next_steps(self, health_report: SystemHealthReport, 
                                  insights: List[ArchitectureInsight]) -> List[Dict[str, Any]]:
        """Generate actionable next steps"""
        next_steps = []
        
        # Immediate actions for critical issues
        critical_insights = [i for i in insights if i.severity == 'critical']
        for i, insight in enumerate(critical_insights[:3], 1):
            next_steps.append({
                'phase': 'immediate',
                'priority': 'critical',
                'title': f"Address {insight.title}",
                'description': insight.description,
                'actions': insight.recommendations[:2],
                'timeline': '1-2 weeks',
                'owner': 'Development Team'
            })
        
        # Short-term improvements
        warning_insights = [i for i in insights if i.severity == 'warning']
        for i, insight in enumerate(warning_insights[:2], len(critical_insights) + 1):
            next_steps.append({
                'phase': 'short_term',
                'priority': 'high',
                'title': f"Improve {insight.category.title()}",
                'description': insight.description,
                'actions': insight.recommendations[:2],
                'timeline': '1 month',
                'owner': 'Development Team'
            })
        
        # Long-term strategic improvements
        if health_report.overall_score < 80:
            next_steps.append({
                'phase': 'long_term',
                'priority': 'medium',
                'title': 'Establish Architecture Governance',
                'description': 'Implement ongoing architecture monitoring and governance processes',
                'actions': [
                    'Set up automated architecture quality gates',
                    'Establish regular architecture review meetings',
                    'Create architecture decision record (ADR) process'
                ],
                'timeline': '3-6 months',
                'owner': 'Architecture Team'
            })
        
        return next_steps
    
    # Utility methods for data conversion
    def _insight_to_dict(self, insight: ArchitectureInsight) -> Dict[str, Any]:
        """Convert insight to dictionary"""
        return {
            'category': insight.category,
            'severity': insight.severity,
            'title': insight.title,
            'description': insight.description,
            'evidence': insight.evidence,
            'recommendations': insight.recommendations,
            'affected_components': insight.affected_components,
            'confidence': insight.confidence
        }
    
    def _health_report_to_dict(self, report: SystemHealthReport) -> Dict[str, Any]:
        """Convert health report to dictionary"""
        return {
            'overall_score': round(report.overall_score, 1),
            'health_status': report.health_status,
            'key_strengths': report.key_strengths,
            'critical_issues': report.critical_issues,
            'improvement_priorities': report.improvement_priorities,
            'trend_analysis': report.trend_analysis,
            'benchmark_comparison': report.benchmark_comparison
        }
    
    # Session management methods
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of an analysis session"""
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        return {
            'session_id': session.session_id,
            'status': session.status,
            'progress': len([t for t in session.execution_times.values()]),
            'errors': session.errors,
            'warnings': session.warnings,
            'execution_times': session.execution_times
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all analysis sessions"""
        return [
            {
                'session_id': session.session_id,
                'project_path': session.project_path,
                'timestamp': session.timestamp.isoformat(),
                'status': session.status
            }
            for session in self.active_sessions.values()
        ]
    
    def clear_cache(self) -> None:
        """Clear result cache"""
        self.result_cache.clear()
        self.logger.info("Result cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.result_cache),
            'cache_enabled': self.cache_results,
            'memory_usage': sum(len(str(result)) for result in self.result_cache.values())
        }
    
    # Export and reporting methods
    async def export_analysis_report(self, session_id: str, format_type: str = 'json') -> str:
        """Export analysis report in specified format"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        if session.status != 'completed':
            raise ValueError(f"Session {session_id} is not completed")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == 'json':
            filename = f"architecture_analysis_{session_id}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(session.results, f, indent=2, default=str)
        
        elif format_type == 'html':
            filename = f"architecture_analysis_{session_id}_{timestamp}.html"
            await self._generate_html_report(session.results, filename)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return filename
    
    async def _generate_html_report(self, results: Dict[str, Any], filename: str) -> None:
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Architecture Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .critical {{ border-left-color: #d32f2f; }}
                .warning {{ border-left-color: #f57c00; }}
                .good {{ border-left-color: #388e3c; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>System Architecture Analysis Report</h1>
                <p><strong>Project:</strong> {results.get('project_path', 'Unknown')}</p>
                <p><strong>Generated:</strong> {results.get('timestamp', 'Unknown')}</p>
                <p><strong>Session ID:</strong> {results.get('session_id', 'Unknown')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Overall Health Score:</strong> 
                    {results.get('health_report', {}).get('overall_score', 0):.1f}/100
                </div>
                <div class="metric">
                    <strong>Status:</strong> 
                    {results.get('health_report', {}).get('health_status', 'Unknown').title()}
                </div>
                <div class="metric">
                    <strong>Critical Issues:</strong> 
                    {len([i for i in results.get('insights', []) if i.get('severity') == 'critical'])}
                </div>
            </div>
            
            <div class="section">
                <h2>Key Insights</h2>
        """
        
        # Add insights
        for insight in results.get('insights', [])[:10]:
            severity_class = insight.get('severity', 'info')
            html_content += f"""
                <div class="section {severity_class}">
                    <h3>{insight.get('title', 'Unknown')}</h3>
                    <p><strong>Category:</strong> {insight.get('category', 'Unknown').title()}</p>
                    <p><strong>Severity:</strong> {insight.get('severity', 'Unknown').title()}</p>
                    <p>{insight.get('description', 'No description available')}</p>
                    <p><strong>Recommendations:</strong></p>
                    <ul>
            """
            for rec in insight.get('recommendations', [])[:3]:
                html_content += f"<li>{rec}</li>"
            html_content += "</ul></div>"
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Next Steps</h2>
                <ol>
        """
        
        # Add next steps
        for step in results.get('next_steps', [])[:5]:
            html_content += f"""
                <li>
                    <strong>{step.get('title', 'Unknown')}</strong> 
                    ({step.get('timeline', 'Unknown timeline')})
                    <p>{step.get('description', 'No description')}</p>
                </li>
            """
        
        html_content += """
                </ol>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)


# Convenience function for quick analysis
async def analyze_project(project_path: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze a project with default configuration
    
    Args:
        project_path: Path to the project to analyze
        config: Optional configuration (uses defaults if not provided)
        
    Returns:
        Complete analysis results
    """
    if config is None:
        config = {
            'enable_parallel_execution': True,
            'cache_results': True,
            'cross_validation': True
        }
    
    coordinator = MasterCoordinator(config)
    return await coordinator.analyze_system(project_path, 'comprehensive')