"""
Migration Planning Agent

This agent provides comprehensive migration planning capabilities including version migration,
technology stack transitions, architecture migrations, and database schema migrations.
"""

import ast
import os
import json
import logging
import asyncio
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum
import importlib.util

from ...core.base.agent import BaseAgent


class MigrationType(Enum):
    """Types of migrations"""
    VERSION_UPGRADE = "version_upgrade"
    FRAMEWORK_MIGRATION = "framework_migration"
    ARCHITECTURE_MIGRATION = "architecture_migration"
    DATABASE_MIGRATION = "database_migration"
    CLOUD_MIGRATION = "cloud_migration"
    LANGUAGE_MIGRATION = "language_migration"
    DEPENDENCY_MIGRATION = "dependency_migration"


@dataclass
class MigrationStep:
    """Represents a single migration step"""
    step_id: str
    name: str
    description: str
    category: str  # 'preparation', 'execution', 'validation', 'rollback'
    dependencies: List[str]  # List of step IDs this depends on
    estimated_hours: float
    complexity: str  # 'low', 'medium', 'high', 'critical'
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    automation_level: str  # 'manual', 'semi_automated', 'automated'
    required_skills: List[str]
    resources_needed: List[str]
    validation_criteria: List[str]
    rollback_procedure: str
    affected_components: List[str]


@dataclass
class MigrationRisk:
    """Represents a migration risk"""
    risk_id: str
    category: str  # 'technical', 'business', 'operational', 'security'
    severity: str  # 'low', 'medium', 'high', 'critical'
    probability: float  # 0.0 to 1.0
    impact: str
    description: str
    mitigation_strategy: str
    contingency_plan: str
    monitoring_indicators: List[str]
    responsible_team: str


@dataclass
class MigrationCompatibility:
    """Compatibility analysis results"""
    component: str
    current_version: str
    target_version: str
    compatibility_status: str  # 'compatible', 'minor_issues', 'major_issues', 'incompatible'
    breaking_changes: List[str]
    required_changes: List[str]
    estimated_effort: float
    alternative_options: List[str]


@dataclass
class MigrationPlan:
    """Complete migration plan"""
    plan_id: str
    name: str
    migration_type: MigrationType
    description: str
    current_state: Dict[str, Any]
    target_state: Dict[str, Any]
    steps: List[MigrationStep]
    risks: List[MigrationRisk]
    timeline: Dict[str, Any]
    resources_required: Dict[str, Any]
    success_criteria: List[str]
    rollback_plan: List[MigrationStep]
    testing_strategy: Dict[str, Any]
    communication_plan: Dict[str, Any]
    cost_estimate: Dict[str, float]
    created_at: datetime
    estimated_completion: datetime


@dataclass
class DependencyAnalysis:
    """Analysis of dependency changes"""
    package_name: str
    current_version: str
    target_version: str
    change_type: str  # 'major', 'minor', 'patch'
    breaking_changes: List[str]
    new_features: List[str]
    deprecated_features: List[str]
    security_fixes: List[str]
    migration_complexity: str
    update_priority: str  # 'critical', 'high', 'medium', 'low'


class MigrationPlanningAgent(BaseAgent):
    """
    Migration Planning Agent
    
    Provides comprehensive migration planning including:
    - Version upgrade planning (Python, frameworks, libraries)
    - Architecture migration strategies
    - Database schema migration planning
    - Cloud migration roadmaps
    - Risk assessment and mitigation
    - Timeline and resource estimation
    - Rollback planning
    - Testing and validation strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("MigrationPlanning", config.get('migration_planning', {}))
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.risk_tolerance = config.get('risk_tolerance', 'medium')  # 'low', 'medium', 'high'
        self.migration_window = config.get('migration_window_hours', 8)
        self.rollback_time_limit = config.get('rollback_time_limit_hours', 2)
        self.parallel_execution = config.get('allow_parallel_execution', True)
        
        # Migration templates and patterns
        self.migration_templates = self._load_migration_templates()
        self.compatibility_matrix = self._load_compatibility_matrix()
        self.risk_patterns = self._load_risk_patterns()
        
    async def create_migration_plan(self, migration_type: str, source_config: Dict[str, Any], 
                                   target_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive migration plan
        
        Args:
            migration_type: Type of migration (version_upgrade, framework_migration, etc.)
            source_config: Current system configuration
            target_config: Target system configuration
            
        Returns:
            Complete migration plan
        """
        self.logger.info(f"Creating migration plan: {migration_type}")
        
        migration_enum = MigrationType(migration_type)
        
        # Analyze current state
        current_analysis = await self._analyze_current_state(source_config)
        
        # Analyze target state
        target_analysis = await self._analyze_target_state(target_config)
        
        # Perform compatibility analysis
        compatibility = await self._analyze_compatibility(source_config, target_config, migration_enum)
        
        # Generate migration steps
        steps = await self._generate_migration_steps(migration_enum, current_analysis, target_analysis, compatibility)
        
        # Assess risks
        risks = await self._assess_migration_risks(steps, compatibility, migration_enum)
        
        # Calculate timeline and resources
        timeline = self._calculate_timeline(steps)
        resources = self._estimate_resources(steps, risks)
        
        # Create rollback plan
        rollback_steps = self._create_rollback_plan(steps, current_analysis)
        
        # Generate testing strategy
        testing_strategy = self._create_testing_strategy(migration_enum, steps, target_analysis)
        
        # Create communication plan
        communication_plan = self._create_communication_plan(timeline, risks)
        
        # Estimate costs
        cost_estimate = self._estimate_costs(resources, timeline)
        
        # Create migration plan
        plan = MigrationPlan(
            plan_id=f"migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=f"{migration_type.replace('_', ' ').title()} Migration",
            migration_type=migration_enum,
            description=f"Migration from {source_config.get('version', 'current')} to {target_config.get('version', 'target')}",
            current_state=current_analysis,
            target_state=target_analysis,
            steps=steps,
            risks=risks,
            timeline=timeline,
            resources_required=resources,
            success_criteria=self._define_success_criteria(target_analysis, migration_enum),
            rollback_plan=rollback_steps,
            testing_strategy=testing_strategy,
            communication_plan=communication_plan,
            cost_estimate=cost_estimate,
            created_at=datetime.utcnow(),
            estimated_completion=datetime.utcnow() + timedelta(hours=timeline.get('total_hours', 24))
        )
        
        return {
            'migration_plan': self._plan_to_dict(plan),
            'compatibility_analysis': [self._compatibility_to_dict(c) for c in compatibility],
            'risk_assessment': [self._risk_to_dict(r) for r in risks],
            'step_dependencies': self._generate_dependency_graph(steps),
            'resource_allocation': self._create_resource_allocation(resources, timeline),
            'milestone_tracking': self._create_milestone_tracking(steps),
            'quality_gates': self._define_quality_gates(migration_enum),
            'monitoring_plan': self._create_monitoring_plan(steps, risks)
        }
    
    async def _analyze_current_state(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the current system state"""
        analysis = {
            'python_version': self._detect_python_version(),
            'dependencies': await self._analyze_dependencies(config.get('project_path', '.')),
            'architecture': await self._analyze_architecture(config.get('project_path', '.')),
            'database_schema': await self._analyze_database_schema(config.get('database_config', {})),
            'infrastructure': config.get('infrastructure', {}),
            'performance_baseline': await self._collect_performance_baseline(config.get('project_path', '.')),
            'security_posture': await self._assess_security_posture(config.get('project_path', '.')),
            'test_coverage': await self._assess_test_coverage(config.get('project_path', '.')),
            'technical_debt': await self._assess_technical_debt(config.get('project_path', '.'))
        }
        
        return analysis
    
    async def _analyze_target_state(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the target system state"""
        return {
            'python_version': config.get('python_version'),
            'target_dependencies': config.get('dependencies', {}),
            'target_architecture': config.get('architecture', {}),
            'target_database': config.get('database_config', {}),
            'target_infrastructure': config.get('infrastructure', {}),
            'performance_requirements': config.get('performance_requirements', {}),
            'security_requirements': config.get('security_requirements', {}),
            'compliance_requirements': config.get('compliance_requirements', [])
        }
    
    async def _analyze_compatibility(self, source: Dict[str, Any], target: Dict[str, Any], 
                                   migration_type: MigrationType) -> List[MigrationCompatibility]:
        """Analyze compatibility between source and target"""
        compatibility_results = []
        
        if migration_type == MigrationType.VERSION_UPGRADE:
            # Python version compatibility
            python_compat = self._check_python_compatibility(
                source.get('python_version'), target.get('python_version')
            )
            compatibility_results.append(python_compat)
            
            # Dependency compatibility
            dep_compat = await self._check_dependency_compatibility(
                source.get('dependencies', {}), target.get('target_dependencies', {})
            )
            compatibility_results.extend(dep_compat)
            
        elif migration_type == MigrationType.FRAMEWORK_MIGRATION:
            # Framework compatibility
            framework_compat = await self._check_framework_compatibility(
                source.get('architecture', {}), target.get('target_architecture', {})
            )
            compatibility_results.extend(framework_compat)
            
        elif migration_type == MigrationType.DATABASE_MIGRATION:
            # Database compatibility
            db_compat = await self._check_database_compatibility(
                source.get('database_schema', {}), target.get('target_database', {})
            )
            compatibility_results.append(db_compat)
            
        return compatibility_results
    
    def _check_python_compatibility(self, current_version: str, target_version: str) -> MigrationCompatibility:
        """Check Python version compatibility"""
        if not current_version or not target_version:
            return MigrationCompatibility(
                component="Python",
                current_version=current_version or "unknown",
                target_version=target_version or "unknown",
                compatibility_status="unknown",
                breaking_changes=[],
                required_changes=[],
                estimated_effort=0.0,
                alternative_options=[]
            )
        
        # Parse version numbers
        current_parts = [int(x) for x in current_version.split('.')]
        target_parts = [int(x) for x in target_version.split('.')]
        
        breaking_changes = []
        required_changes = []
        compatibility_status = "compatible"
        
        # Major version change
        if target_parts[0] > current_parts[0]:
            compatibility_status = "major_issues"
            breaking_changes.extend([
                "Major version upgrade may break compatibility",
                "Review deprecated features and syntax changes",
                "Update import statements and method calls"
            ])
            required_changes.extend([
                "Update all dependency versions",
                "Refactor deprecated code patterns",
                "Update CI/CD configurations"
            ])
        
        # Minor version changes
        elif target_parts[1] > current_parts[1]:
            if target_parts[1] - current_parts[1] > 2:
                compatibility_status = "minor_issues"
                breaking_changes.append("Multiple minor version jump may introduce incompatibilities")
        
        return MigrationCompatibility(
            component="Python",
            current_version=current_version,
            target_version=target_version,
            compatibility_status=compatibility_status,
            breaking_changes=breaking_changes,
            required_changes=required_changes,
            estimated_effort=self._estimate_python_migration_effort(current_parts, target_parts),
            alternative_options=self._suggest_python_alternatives(current_version, target_version)
        )
    
    async def _check_dependency_compatibility(self, current_deps: Dict[str, str], 
                                            target_deps: Dict[str, str]) -> List[MigrationCompatibility]:
        """Check dependency compatibility"""
        compatibility_results = []
        
        all_deps = set(current_deps.keys()) | set(target_deps.keys())
        
        for dep_name in all_deps:
            current_version = current_deps.get(dep_name, "not_installed")
            target_version = target_deps.get(dep_name, "to_remove")
            
            if current_version == "not_installed":
                # New dependency
                compat = MigrationCompatibility(
                    component=dep_name,
                    current_version=current_version,
                    target_version=target_version,
                    compatibility_status="compatible",
                    breaking_changes=[],
                    required_changes=[f"Install {dep_name}=={target_version}"],
                    estimated_effort=1.0,
                    alternative_options=[]
                )
            elif target_version == "to_remove":
                # Dependency removal
                compat = MigrationCompatibility(
                    component=dep_name,
                    current_version=current_version,
                    target_version=target_version,
                    compatibility_status="major_issues",
                    breaking_changes=[f"Removing {dep_name} may break existing functionality"],
                    required_changes=[f"Refactor code that depends on {dep_name}"],
                    estimated_effort=8.0,
                    alternative_options=self._suggest_dependency_alternatives(dep_name)
                )
            else:
                # Version change
                compat = await self._analyze_dependency_version_change(dep_name, current_version, target_version)
            
            compatibility_results.append(compat)
        
        return compatibility_results
    
    async def _analyze_dependency_version_change(self, package: str, current: str, target: str) -> MigrationCompatibility:
        """Analyze a specific dependency version change"""
        # This would integrate with package registries for detailed analysis
        # For now, provide heuristic-based analysis
        
        try:
            current_parts = [int(x) for x in current.split('.')]
            target_parts = [int(x) for x in target.split('.')]
            
            compatibility_status = "compatible"
            breaking_changes = []
            required_changes = []
            effort = 1.0
            
            # Major version change
            if target_parts[0] > current_parts[0]:
                compatibility_status = "major_issues"
                breaking_changes.append(f"Major version upgrade of {package}")
                required_changes.append(f"Review {package} documentation for breaking changes")
                effort = 16.0
                
            # Minor version change with significant gap
            elif target_parts[1] > current_parts[1] and (target_parts[1] - current_parts[1]) > 3:
                compatibility_status = "minor_issues"
                breaking_changes.append(f"Significant minor version jump in {package}")
                effort = 4.0
            
            return MigrationCompatibility(
                component=package,
                current_version=current,
                target_version=target,
                compatibility_status=compatibility_status,
                breaking_changes=breaking_changes,
                required_changes=required_changes,
                estimated_effort=effort,
                alternative_options=[]
            )
            
        except ValueError:
            # Non-standard version format
            return MigrationCompatibility(
                component=package,
                current_version=current,
                target_version=target,
                compatibility_status="unknown",
                breaking_changes=[f"Non-standard version format for {package}"],
                required_changes=[f"Manual review required for {package}"],
                estimated_effort=8.0,
                alternative_options=[]
            )
    
    async def _generate_migration_steps(self, migration_type: MigrationType, 
                                       current: Dict[str, Any], target: Dict[str, Any],
                                       compatibility: List[MigrationCompatibility]) -> List[MigrationStep]:
        """Generate detailed migration steps"""
        steps = []
        
        # Add preparation steps
        steps.extend(self._generate_preparation_steps(migration_type, current, target))
        
        # Add pre-migration validation steps
        steps.extend(self._generate_validation_steps("pre_migration", current, target))
        
        # Add main migration steps based on type
        if migration_type == MigrationType.VERSION_UPGRADE:
            steps.extend(await self._generate_version_upgrade_steps(compatibility))
        elif migration_type == MigrationType.FRAMEWORK_MIGRATION:
            steps.extend(await self._generate_framework_migration_steps(current, target))
        elif migration_type == MigrationType.DATABASE_MIGRATION:
            steps.extend(await self._generate_database_migration_steps(current, target))
        elif migration_type == MigrationType.ARCHITECTURE_MIGRATION:
            steps.extend(await self._generate_architecture_migration_steps(current, target))
        
        # Add post-migration validation steps
        steps.extend(self._generate_validation_steps("post_migration", current, target))
        
        # Add cleanup steps
        steps.extend(self._generate_cleanup_steps(migration_type))
        
        return steps
    
    def _generate_preparation_steps(self, migration_type: MigrationType, 
                                   current: Dict[str, Any], target: Dict[str, Any]) -> List[MigrationStep]:
        """Generate preparation steps"""
        steps = []
        
        # Backup current system
        steps.append(MigrationStep(
            step_id="prep_001",
            name="Create System Backup",
            description="Create comprehensive backup of current system",
            category="preparation",
            dependencies=[],
            estimated_hours=2.0,
            complexity="medium",
            risk_level="low",
            automation_level="automated",
            required_skills=["DevOps", "System Administration"],
            resources_needed=["Backup Storage", "Backup Scripts"],
            validation_criteria=["Backup integrity verified", "Recovery test passed"],
            rollback_procedure="Restore from backup if needed",
            affected_components=["Database", "Application Code", "Configuration"]
        ))
        
        # Environment setup
        steps.append(MigrationStep(
            step_id="prep_002",
            name="Setup Migration Environment",
            description="Prepare isolated environment for migration testing",
            category="preparation",
            dependencies=[],
            estimated_hours=4.0,
            complexity="medium",
            risk_level="low",
            automation_level="semi_automated",
            required_skills=["DevOps", "Cloud Infrastructure"],
            resources_needed=["Test Environment", "Migration Tools"],
            validation_criteria=["Environment matches production", "All tools installed"],
            rollback_procedure="Destroy test environment",
            affected_components=["Infrastructure", "Deployment Pipeline"]
        ))
        
        # Dependency analysis
        steps.append(MigrationStep(
            step_id="prep_003",
            name="Analyze Dependencies",
            description="Perform detailed analysis of all system dependencies",
            category="preparation",
            dependencies=[],
            estimated_hours=8.0,
            complexity="high",
            risk_level="medium",
            automation_level="semi_automated",
            required_skills=["Software Architecture", "Python Development"],
            resources_needed=["Dependency Analysis Tools", "Documentation"],
            validation_criteria=["All dependencies catalogued", "Compatibility matrix created"],
            rollback_procedure="No rollback needed for analysis",
            affected_components=["All Application Components"]
        ))
        
        return steps
    
    def _generate_validation_steps(self, phase: str, current: Dict[str, Any], 
                                  target: Dict[str, Any]) -> List[MigrationStep]:
        """Generate validation steps"""
        steps = []
        
        if phase == "pre_migration":
            steps.append(MigrationStep(
                step_id="val_001",
                name="Pre-Migration Health Check",
                description="Verify system health before migration",
                category="validation",
                dependencies=["prep_001", "prep_002"],
                estimated_hours=2.0,
                complexity="low",
                risk_level="low",
                automation_level="automated",
                required_skills=["Testing", "System Monitoring"],
                resources_needed=["Monitoring Tools", "Test Scripts"],
                validation_criteria=["All health checks pass", "Performance baseline established"],
                rollback_procedure="Abort migration if health checks fail",
                affected_components=["All System Components"]
            ))
        
        elif phase == "post_migration":
            steps.append(MigrationStep(
                step_id="val_002",
                name="Post-Migration Validation",
                description="Comprehensive validation of migrated system",
                category="validation",
                dependencies=["exec_*"],  # Depends on all execution steps
                estimated_hours=6.0,
                complexity="high",
                risk_level="medium",
                automation_level="semi_automated",
                required_skills=["QA Testing", "System Validation"],
                resources_needed=["Test Suite", "Validation Scripts"],
                validation_criteria=["All tests pass", "Performance targets met", "No regression detected"],
                rollback_procedure="Initiate rollback if validation fails",
                affected_components=["All System Components"]
            ))
        
        return steps
    
    async def _generate_version_upgrade_steps(self, compatibility: List[MigrationCompatibility]) -> List[MigrationStep]:
        """Generate version upgrade specific steps"""
        steps = []
        
        # Python version upgrade
        python_compat = next((c for c in compatibility if c.component == "Python"), None)
        if python_compat:
            steps.append(MigrationStep(
                step_id="exec_001",
                name="Upgrade Python Version",
                description=f"Upgrade Python from {python_compat.current_version} to {python_compat.target_version}",
                category="execution",
                dependencies=["val_001"],
                estimated_hours=python_compat.estimated_effort,
                complexity=self._get_complexity_from_compatibility(python_compat),
                risk_level=self._get_risk_from_compatibility(python_compat),
                automation_level="semi_automated",
                required_skills=["Python Development", "System Administration"],
                resources_needed=["Python Installer", "Virtual Environment"],
                validation_criteria=["Python version verified", "Basic imports work"],
                rollback_procedure="Restore previous Python version",
                affected_components=["Python Runtime", "All Python Code"]
            ))
        
        # Dependency upgrades
        dependency_compat = [c for c in compatibility if c.component != "Python"]
        for i, dep_compat in enumerate(dependency_compat, 2):
            steps.append(MigrationStep(
                step_id=f"exec_{i:03d}",
                name=f"Update {dep_compat.component}",
                description=f"Update {dep_compat.component} from {dep_compat.current_version} to {dep_compat.target_version}",
                category="execution",
                dependencies=["exec_001"] if python_compat else ["val_001"],
                estimated_hours=dep_compat.estimated_effort,
                complexity=self._get_complexity_from_compatibility(dep_compat),
                risk_level=self._get_risk_from_compatibility(dep_compat),
                automation_level="automated" if dep_compat.estimated_effort < 2 else "semi_automated",
                required_skills=["Python Development", "Package Management"],
                resources_needed=["Package Manager", "Testing Tools"],
                validation_criteria=[f"{dep_compat.component} version verified", "No import errors"],
                rollback_procedure=f"Revert {dep_compat.component} to previous version",
                affected_components=[f"Components using {dep_compat.component}"]
            ))
        
        return steps
    
    async def _assess_migration_risks(self, steps: List[MigrationStep], 
                                     compatibility: List[MigrationCompatibility],
                                     migration_type: MigrationType) -> List[MigrationRisk]:
        """Assess migration risks"""
        risks = []
        
        # Technical risks
        high_complexity_steps = [s for s in steps if s.complexity in ['high', 'critical']]
        if high_complexity_steps:
            risks.append(MigrationRisk(
                risk_id="risk_001",
                category="technical",
                severity="high",
                probability=0.7,
                impact="Migration may fail or require significant additional time",
                description=f"Multiple high-complexity steps ({len(high_complexity_steps)}) increase technical risk",
                mitigation_strategy="Allocate experienced team members, conduct thorough testing",
                contingency_plan="Have rollback plan ready, extend migration window",
                monitoring_indicators=["Step completion time", "Error rates", "Performance metrics"],
                responsible_team="Development Team"
            ))
        
        # Compatibility risks
        incompatible_components = [c for c in compatibility if c.compatibility_status in ['major_issues', 'incompatible']]
        if incompatible_components:
            risks.append(MigrationRisk(
                risk_id="risk_002",
                category="technical",
                severity="critical",
                probability=0.9,
                impact="System functionality may break after migration",
                description=f"Incompatible components detected: {[c.component for c in incompatible_components]}",
                mitigation_strategy="Address compatibility issues before migration, prepare workarounds",
                contingency_plan="Rollback to previous versions, implement temporary fixes",
                monitoring_indicators=["Component functionality", "Integration test results"],
                responsible_team="Development Team"
            ))
        
        # Timeline risks
        total_hours = sum(step.estimated_hours for step in steps)
        if total_hours > self.migration_window * 2:
            risks.append(MigrationRisk(
                risk_id="risk_003",
                category="operational",
                severity="medium",
                probability=0.6,
                impact="Migration may exceed planned downtime window",
                description=f"Estimated time ({total_hours}h) exceeds comfortable window",
                mitigation_strategy="Parallelize steps where possible, pre-stage components",
                contingency_plan="Extend maintenance window or rollback",
                monitoring_indicators=["Step progress", "Time remaining"],
                responsible_team="DevOps Team"
            ))
        
        # Business risks
        if migration_type in [MigrationType.ARCHITECTURE_MIGRATION, MigrationType.FRAMEWORK_MIGRATION]:
            risks.append(MigrationRisk(
                risk_id="risk_004",
                category="business",
                severity="medium",
                probability=0.4,
                impact="Business operations may be disrupted",
                description="Major architectural changes may affect business processes",
                mitigation_strategy="Comprehensive user acceptance testing, gradual rollout",
                contingency_plan="Quick rollback, manual process fallbacks",
                monitoring_indicators=["User complaints", "Business metrics", "System availability"],
                responsible_team="Product Team"
            ))
        
        return risks
    
    def _calculate_timeline(self, steps: List[MigrationStep]) -> Dict[str, Any]:
        """Calculate migration timeline"""
        # Build dependency graph
        step_deps = {step.step_id: step.dependencies for step in steps}
        step_hours = {step.step_id: step.estimated_hours for step in steps}
        
        # Calculate critical path
        critical_path = self._find_critical_path(steps, step_deps, step_hours)
        
        # Calculate parallel execution opportunities
        parallel_groups = self._find_parallel_groups(steps, step_deps) if self.parallel_execution else []
        
        # Calculate total time considering parallelization
        if parallel_groups:
            total_hours = max(sum(step_hours[step_id] for step_id in group) for group in parallel_groups)
        else:
            total_hours = sum(step.estimated_hours for step in steps)
        
        # Add buffer time based on risk
        buffer_percentage = 0.3  # 30% buffer
        total_with_buffer = total_hours * (1 + buffer_percentage)
        
        return {
            'total_hours': total_hours,
            'total_with_buffer': total_with_buffer,
            'critical_path': critical_path,
            'parallel_groups': parallel_groups,
            'milestone_dates': self._calculate_milestone_dates(steps, total_with_buffer),
            'dependencies': step_deps
        }
    
    def _find_critical_path(self, steps: List[MigrationStep], dependencies: Dict[str, List[str]], 
                           hours: Dict[str, float]) -> List[str]:
        """Find the critical path through migration steps"""
        # Simple critical path calculation
        # In practice, would use more sophisticated algorithms
        
        # Topological sort to find execution order
        in_degree = {step.step_id: 0 for step in steps}
        for step in steps:
            for dep in step.dependencies:
                if dep in in_degree:
                    in_degree[step.step_id] += 1
        
        # Find path with maximum total time
        max_path = []
        max_time = 0
        
        def find_path(step_id, current_path, current_time):
            nonlocal max_path, max_time
            
            current_path = current_path + [step_id]
            current_time += hours.get(step_id, 0)
            
            # Find next steps
            next_steps = [s.step_id for s in steps if step_id in s.dependencies]
            
            if not next_steps:
                # End of path
                if current_time > max_time:
                    max_time = current_time
                    max_path = current_path[:]
            else:
                for next_step in next_steps:
                    find_path(next_step, current_path, current_time)
        
        # Start from steps with no dependencies
        start_steps = [step.step_id for step in steps if not step.dependencies]
        for start_step in start_steps:
            find_path(start_step, [], 0)
        
        return max_path
    
    def _find_parallel_groups(self, steps: List[MigrationStep], dependencies: Dict[str, List[str]]) -> List[List[str]]:
        """Find groups of steps that can be executed in parallel"""
        # Group steps by their dependency level
        levels = {}
        
        def get_level(step_id):
            if step_id in levels:
                return levels[step_id]
            
            step = next(s for s in steps if s.step_id == step_id)
            if not step.dependencies:
                levels[step_id] = 0
                return 0
            
            max_dep_level = max(get_level(dep) for dep in step.dependencies if dep in [s.step_id for s in steps])
            levels[step_id] = max_dep_level + 1
            return levels[step_id]
        
        # Calculate levels for all steps
        for step in steps:
            get_level(step.step_id)
        
        # Group by level
        parallel_groups = []
        level_groups = {}
        for step_id, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(step_id)
        
        # Convert to list of parallel groups
        for level in sorted(level_groups.keys()):
            if len(level_groups[level]) > 1:
                parallel_groups.append(level_groups[level])
        
        return parallel_groups
    
    def _calculate_milestone_dates(self, steps: List[MigrationStep], total_hours: float) -> Dict[str, str]:
        """Calculate milestone dates"""
        start_date = datetime.utcnow()
        
        milestones = {}
        
        # Preparation complete
        prep_steps = [s for s in steps if s.category == 'preparation']
        prep_hours = sum(s.estimated_hours for s in prep_steps)
        milestones['preparation_complete'] = (start_date + timedelta(hours=prep_hours)).isoformat()
        
        # Execution start
        milestones['execution_start'] = (start_date + timedelta(hours=prep_hours)).isoformat()
        
        # Execution complete
        exec_steps = [s for s in steps if s.category == 'execution']
        exec_hours = sum(s.estimated_hours for s in exec_steps)
        milestones['execution_complete'] = (start_date + timedelta(hours=prep_hours + exec_hours)).isoformat()
        
        # Validation complete
        milestones['validation_complete'] = (start_date + timedelta(hours=total_hours)).isoformat()
        
        return milestones
    
    def _estimate_resources(self, steps: List[MigrationStep], risks: List[MigrationRisk]) -> Dict[str, Any]:
        """Estimate required resources"""
        # Collect all required skills
        all_skills = set()
        for step in steps:
            all_skills.update(step.required_skills)
        
        # Collect all resources
        all_resources = set()
        for step in steps:
            all_resources.update(step.resources_needed)
        
        # Estimate team size based on complexity and timeline
        total_hours = sum(step.estimated_hours for step in steps)
        high_risk_count = len([r for r in risks if r.severity in ['high', 'critical']])
        
        base_team_size = 3
        if total_hours > 40:
            base_team_size += 2
        if high_risk_count > 2:
            base_team_size += 1
        
        return {
            'team_size': base_team_size,
            'required_skills': list(all_skills),
            'required_resources': list(all_resources),
            'specialist_hours': {
                'DevOps': sum(s.estimated_hours for s in steps if 'DevOps' in s.required_skills),
                'Development': sum(s.estimated_hours for s in steps if 'Python Development' in s.required_skills),
                'QA': sum(s.estimated_hours for s in steps if 'Testing' in s.required_skills),
                'Architecture': sum(s.estimated_hours for s in steps if 'Software Architecture' in s.required_skills)
            }
        }
    
    def _create_rollback_plan(self, steps: List[MigrationStep], current_state: Dict[str, Any]) -> List[MigrationStep]:
        """Create rollback plan"""
        rollback_steps = []
        
        # General rollback steps
        rollback_steps.append(MigrationStep(
            step_id="rollback_001",
            name="Initiate Emergency Rollback",
            description="Immediately stop migration and begin rollback procedures",
            category="rollback",
            dependencies=[],
            estimated_hours=0.5,
            complexity="high",
            risk_level="critical",
            automation_level="semi_automated",
            required_skills=["DevOps", "System Administration"],
            resources_needed=["Backup Systems", "Rollback Scripts"],
            validation_criteria=["Migration stopped", "System isolated"],
            rollback_procedure="Manual intervention required",
            affected_components=["All Systems"]
        ))
        
        rollback_steps.append(MigrationStep(
            step_id="rollback_002",
            name="Restore from Backup",
            description="Restore system to pre-migration state from backup",
            category="rollback",
            dependencies=["rollback_001"],
            estimated_hours=2.0,
            complexity="medium",
            risk_level="medium",
            automation_level="automated",
            required_skills=["DevOps", "Database Administration"],
            resources_needed=["Backup Files", "Restore Scripts"],
            validation_criteria=["Backup restored successfully", "Data integrity verified"],
            rollback_procedure="Manual backup restoration if automated fails",
            affected_components=["Database", "Application Files", "Configuration"]
        ))
        
        rollback_steps.append(MigrationStep(
            step_id="rollback_003",
            name="Validate Rollback",
            description="Verify system functionality after rollback",
            category="rollback",
            dependencies=["rollback_002"],
            estimated_hours=1.0,
            complexity="medium",
            risk_level="low",
            automation_level="semi_automated",
            required_skills=["QA Testing", "System Monitoring"],
            resources_needed=["Test Scripts", "Monitoring Tools"],
            validation_criteria=["All functionality restored", "Performance acceptable"],
            rollback_procedure="Manual testing if automated validation fails",
            affected_components=["All System Components"]
        ))
        
        return rollback_steps
    
    def _create_testing_strategy(self, migration_type: MigrationType, 
                                steps: List[MigrationStep], target: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive testing strategy"""
        return {
            'pre_migration_testing': {
                'unit_tests': "Run full unit test suite",
                'integration_tests': "Execute integration test suite",
                'performance_tests': "Establish performance baseline",
                'security_tests': "Run security vulnerability scans"
            },
            'migration_testing': {
                'step_validation': "Validate each migration step",
                'rollback_testing': "Test rollback procedures",
                'error_handling': "Test error scenarios and recovery"
            },
            'post_migration_testing': {
                'smoke_tests': "Basic functionality verification",
                'regression_tests': "Full regression test suite",
                'performance_validation': "Verify performance targets met",
                'user_acceptance': "User acceptance testing",
                'load_testing': "Production load simulation"
            },
            'test_environments': [
                "Development environment testing",
                "Staging environment validation",
                "Production-like environment verification"
            ],
            'test_data': {
                'backup_test_data': "Backup current test data",
                'migration_test_data': "Prepare migration-specific test data",
                'validation_datasets': "Create validation datasets"
            }
        }
    
    def _create_communication_plan(self, timeline: Dict[str, Any], risks: List[MigrationRisk]) -> Dict[str, Any]:
        """Create communication plan"""
        return {
            'stakeholders': {
                'executive': "C-level executives and senior management",
                'technical': "Development, DevOps, and QA teams",
                'business': "Product managers and business users",
                'support': "Customer support and operations teams"
            },
            'communication_schedule': {
                'pre_migration': [
                    "Initial announcement (2 weeks before)",
                    "Detailed plan review (1 week before)",
                    "Final readiness check (1 day before)"
                ],
                'during_migration': [
                    "Migration start notification",
                    "Hourly progress updates",
                    "Issue escalation alerts",
                    "Completion notification"
                ],
                'post_migration': [
                    "Success confirmation",
                    "Performance report (24 hours after)",
                    "Lessons learned summary (1 week after)"
                ]
            },
            'escalation_matrix': {
                'low_risk': "Team lead notification",
                'medium_risk': "Manager and stakeholder notification",
                'high_risk': "Executive notification and emergency procedures",
                'critical_risk': "All hands escalation and potential rollback"
            },
            'channels': {
                'primary': "Email and Slack notifications",
                'emergency': "Phone calls and emergency contacts",
                'documentation': "Shared documentation and status pages"
            }
        }
    
    def _estimate_costs(self, resources: Dict[str, Any], timeline: Dict[str, Any]) -> Dict[str, float]:
        """Estimate migration costs"""
        # Simplified cost estimation
        hourly_rates = {
            'DevOps': 150.0,
            'Development': 120.0,
            'QA': 100.0,
            'Architecture': 180.0,
            'Management': 200.0
        }
        
        costs = {}
        
        # Personnel costs
        for role, hours in resources.get('specialist_hours', {}).items():
            rate = hourly_rates.get(role, 100.0)
            costs[f'{role.lower()}_personnel'] = hours * rate
        
        # Infrastructure costs
        total_hours = timeline.get('total_with_buffer', 24)
        costs['infrastructure'] = total_hours * 10.0  # $10/hour for test environments
        
        # Tool and license costs
        costs['tools_licenses'] = 5000.0  # Estimated tool costs
        
        # Contingency
        subtotal = sum(costs.values())
        costs['contingency'] = subtotal * 0.2  # 20% contingency
        
        costs['total'] = sum(costs.values())
        
        return costs
    
    def _define_success_criteria(self, target: Dict[str, Any], migration_type: MigrationType) -> List[str]:
        """Define success criteria for migration"""
        criteria = [
            "All migration steps completed successfully",
            "System functionality verified through testing",
            "Performance meets or exceeds baseline metrics",
            "Zero data loss or corruption",
            "All user acceptance tests pass"
        ]
        
        if migration_type == MigrationType.VERSION_UPGRADE:
            criteria.extend([
                "Target versions installed and verified",
                "All dependencies updated successfully",
                "Backward compatibility maintained where required"
            ])
        elif migration_type == MigrationType.DATABASE_MIGRATION:
            criteria.extend([
                "Database schema migration completed",
                "Data integrity verified",
                "Query performance within acceptable limits"
            ])
        
        return criteria
    
    # Helper methods for detection and analysis
    def _detect_python_version(self) -> str:
        """Detect current Python version"""
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    async def _analyze_dependencies(self, project_path: str) -> Dict[str, str]:
        """Analyze current project dependencies"""
        dependencies = {}
        
        # Look for requirements.txt
        req_file = Path(project_path) / "requirements.txt"
        if req_file.exists():
            with open(req_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '==' in line:
                            name, version = line.split('==', 1)
                            dependencies[name.strip()] = version.strip()
                        elif '>=' in line:
                            name = line.split('>=')[0].strip()
                            dependencies[name] = "unknown"
        
        return dependencies
    
    async def _analyze_architecture(self, project_path: str) -> Dict[str, Any]:
        """Analyze current architecture"""
        # Simplified architecture analysis
        path = Path(project_path)
        
        return {
            'structure': 'monolithic' if not any(path.glob('**/microservice*')) else 'microservices',
            'web_framework': self._detect_web_framework(path),
            'database': self._detect_database_usage(path),
            'async_usage': self._detect_async_usage(path)
        }
    
    def _detect_web_framework(self, path: Path) -> str:
        """Detect web framework in use"""
        frameworks = {
            'django': ['django', 'manage.py'],
            'flask': ['flask', 'app.py'],
            'fastapi': ['fastapi', 'main.py'],
            'tornado': ['tornado'],
            'pyramid': ['pyramid']
        }
        
        for framework, indicators in frameworks.items():
            for indicator in indicators:
                if list(path.rglob(f'*{indicator}*')):
                    return framework
        
        return 'unknown'
    
    def _detect_database_usage(self, path: Path) -> List[str]:
        """Detect database usage"""
        databases = []
        
        db_indicators = {
            'postgresql': ['psycopg2', 'postgresql', 'postgres'],
            'mysql': ['mysql', 'pymysql', 'mysqlclient'],
            'sqlite': ['sqlite3', 'sqlite'],
            'mongodb': ['pymongo', 'mongodb'],
            'redis': ['redis', 'redis-py']
        }
        
        # Check requirements and imports
        for py_file in path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for db, indicators in db_indicators.items():
                    for indicator in indicators:
                        if indicator in content.lower():
                            if db not in databases:
                                databases.append(db)
            except:
                continue
        
        return databases
    
    def _detect_async_usage(self, path: Path) -> bool:
        """Detect async/await usage"""
        for py_file in path.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'async def' in content or 'await ' in content:
                    return True
            except:
                continue
        
        return False
    
    # Additional analysis methods would be implemented here...
    async def _analyze_database_schema(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze database schema - placeholder"""
        return {'tables': [], 'indexes': [], 'constraints': []}
    
    async def _collect_performance_baseline(self, project_path: str) -> Dict[str, Any]:
        """Collect performance baseline - placeholder"""
        return {'response_time': 0, 'throughput': 0, 'memory_usage': 0}
    
    async def _assess_security_posture(self, project_path: str) -> Dict[str, Any]:
        """Assess security posture - placeholder"""
        return {'vulnerabilities': 0, 'security_score': 8.5}
    
    async def _assess_test_coverage(self, project_path: str) -> Dict[str, Any]:
        """Assess test coverage - placeholder"""
        return {'coverage_percentage': 75.0, 'test_count': 150}
    
    async def _assess_technical_debt(self, project_path: str) -> Dict[str, Any]:
        """Assess technical debt - placeholder"""
        return {'debt_hours': 120, 'debt_ratio': 0.15}
    
    # Utility methods
    def _get_complexity_from_compatibility(self, compat: MigrationCompatibility) -> str:
        """Get complexity level from compatibility status"""
        if compat.compatibility_status == 'incompatible':
            return 'critical'
        elif compat.compatibility_status == 'major_issues':
            return 'high'
        elif compat.compatibility_status == 'minor_issues':
            return 'medium'
        else:
            return 'low'
    
    def _get_risk_from_compatibility(self, compat: MigrationCompatibility) -> str:
        """Get risk level from compatibility status"""
        if compat.compatibility_status == 'incompatible':
            return 'critical'
        elif compat.compatibility_status == 'major_issues':
            return 'high'
        elif compat.compatibility_status == 'minor_issues':
            return 'medium'
        else:
            return 'low'
    
    def _estimate_python_migration_effort(self, current: List[int], target: List[int]) -> float:
        """Estimate effort for Python migration"""
        if target[0] > current[0]:  # Major version
            return 40.0
        elif target[1] > current[1] + 2:  # Multiple minor versions
            return 16.0
        elif target[1] > current[1]:  # Single minor version
            return 8.0
        else:  # Patch version
            return 2.0
    
    def _suggest_python_alternatives(self, current: str, target: str) -> List[str]:
        """Suggest Python version alternatives"""
        return [
            f"Consider intermediate version between {current} and {target}",
            "Evaluate container-based deployment for version isolation",
            "Consider gradual migration using feature flags"
        ]
    
    def _suggest_dependency_alternatives(self, package: str) -> List[str]:
        """Suggest alternatives for a dependency"""
        alternatives = {
            'requests': ['httpx', 'aiohttp'],
            'flask': ['fastapi', 'django'],
            'django': ['fastapi', 'flask'],
            'pandas': ['polars', 'dask'],
            'numpy': ['jax', 'cupy']
        }
        
        return alternatives.get(package, ['Research modern alternatives'])
    
    def _load_migration_templates(self) -> Dict[str, Any]:
        """Load migration templates"""
        return {
            'python_upgrade': {'base_steps': 10, 'risk_multiplier': 1.2},
            'framework_migration': {'base_steps': 15, 'risk_multiplier': 1.8},
            'database_migration': {'base_steps': 12, 'risk_multiplier': 1.5}
        }
    
    def _load_compatibility_matrix(self) -> Dict[str, Any]:
        """Load compatibility matrix"""
        return {
            'python_versions': {
                '3.8': {'compatible': ['3.9', '3.10'], 'issues': ['3.11', '3.12']},
                '3.9': {'compatible': ['3.10', '3.11'], 'issues': ['3.12']},
                '3.10': {'compatible': ['3.11', '3.12'], 'issues': []}
            }
        }
    
    def _load_risk_patterns(self) -> Dict[str, Any]:
        """Load risk patterns"""
        return {
            'high_risk_packages': ['numpy', 'pandas', 'django', 'flask'],
            'breaking_change_indicators': ['major version', 'deprecated', 'removed'],
            'complexity_indicators': ['async', 'multiprocessing', 'c extensions']
        }
    
    # Conversion methods
    def _plan_to_dict(self, plan: MigrationPlan) -> Dict[str, Any]:
        """Convert migration plan to dictionary"""
        return {
            'plan_id': plan.plan_id,
            'name': plan.name,
            'migration_type': plan.migration_type.value,
            'description': plan.description,
            'current_state': plan.current_state,
            'target_state': plan.target_state,
            'steps': [self._step_to_dict(s) for s in plan.steps],
            'risks': [self._risk_to_dict(r) for r in plan.risks],
            'timeline': plan.timeline,
            'resources_required': plan.resources_required,
            'success_criteria': plan.success_criteria,
            'rollback_plan': [self._step_to_dict(s) for s in plan.rollback_plan],
            'testing_strategy': plan.testing_strategy,
            'communication_plan': plan.communication_plan,
            'cost_estimate': plan.cost_estimate,
            'created_at': plan.created_at.isoformat(),
            'estimated_completion': plan.estimated_completion.isoformat()
        }
    
    def _step_to_dict(self, step: MigrationStep) -> Dict[str, Any]:
        """Convert migration step to dictionary"""
        return {
            'step_id': step.step_id,
            'name': step.name,
            'description': step.description,
            'category': step.category,
            'dependencies': step.dependencies,
            'estimated_hours': step.estimated_hours,
            'complexity': step.complexity,
            'risk_level': step.risk_level,
            'automation_level': step.automation_level,
            'required_skills': step.required_skills,
            'resources_needed': step.resources_needed,
            'validation_criteria': step.validation_criteria,
            'rollback_procedure': step.rollback_procedure,
            'affected_components': step.affected_components
        }
    
    def _risk_to_dict(self, risk: MigrationRisk) -> Dict[str, Any]:
        """Convert migration risk to dictionary"""
        return {
            'risk_id': risk.risk_id,
            'category': risk.category,
            'severity': risk.severity,
            'probability': risk.probability,
            'impact': risk.impact,
            'description': risk.description,
            'mitigation_strategy': risk.mitigation_strategy,
            'contingency_plan': risk.contingency_plan,
            'monitoring_indicators': risk.monitoring_indicators,
            'responsible_team': risk.responsible_team
        }
    
    def _compatibility_to_dict(self, compat: MigrationCompatibility) -> Dict[str, Any]:
        """Convert compatibility analysis to dictionary"""
        return {
            'component': compat.component,
            'current_version': compat.current_version,
            'target_version': compat.target_version,
            'compatibility_status': compat.compatibility_status,
            'breaking_changes': compat.breaking_changes,
            'required_changes': compat.required_changes,
            'estimated_effort': compat.estimated_effort,
            'alternative_options': compat.alternative_options
        }
    
    # Additional utility methods
    def _generate_dependency_graph(self, steps: List[MigrationStep]) -> Dict[str, Any]:
        """Generate dependency graph visualization data"""
        nodes = [{'id': step.step_id, 'name': step.name, 'category': step.category} for step in steps]
        edges = []
        
        for step in steps:
            for dep in step.dependencies:
                edges.append({'source': dep, 'target': step.step_id})
        
        return {'nodes': nodes, 'edges': edges}
    
    def _create_resource_allocation(self, resources: Dict[str, Any], timeline: Dict[str, Any]) -> Dict[str, Any]:
        """Create resource allocation plan"""
        return {
            'team_assignments': {
                'lead': 'DevOps Lead',
                'developers': resources.get('team_size', 3) - 1,
                'qa': 1,
                'backup': 1
            },
            'schedule': {
                'preparation': '40% of total time',
                'execution': '40% of total time',
                'validation': '20% of total time'
            },
            'availability_requirements': {
                'core_team': '100% during migration window',
                'support_team': '50% during migration window',
                'stakeholders': 'On-call during migration'
            }
        }
    
    def _create_milestone_tracking(self, steps: List[MigrationStep]) -> List[Dict[str, Any]]:
        """Create milestone tracking structure"""
        milestones = []
        
        categories = ['preparation', 'execution', 'validation', 'rollback']
        for category in categories:
            category_steps = [s for s in steps if s.category == category]
            if category_steps:
                milestones.append({
                    'name': f'{category.title()} Complete',
                    'category': category,
                    'step_count': len(category_steps),
                    'estimated_hours': sum(s.estimated_hours for s in category_steps),
                    'completion_criteria': f'All {category} steps completed successfully'
                })
        
        return milestones
    
    def _define_quality_gates(self, migration_type: MigrationType) -> List[Dict[str, Any]]:
        """Define quality gates for migration"""
        gates = [
            {
                'name': 'Pre-Migration Gate',
                'criteria': [
                    'All preparation steps completed',
                    'Backup verified',
                    'Team readiness confirmed',
                    'Rollback plan tested'
                ],
                'blocking': True
            },
            {
                'name': 'Post-Migration Gate',
                'criteria': [
                    'All migration steps completed',
                    'System functionality verified',
                    'Performance acceptable',
                    'No critical issues detected'
                ],
                'blocking': True
            }
        ]
        
        if migration_type == MigrationType.DATABASE_MIGRATION:
            gates.append({
                'name': 'Data Integrity Gate',
                'criteria': [
                    'Data migration completed',
                    'Data integrity verified',
                    'No data loss detected',
                    'Query performance acceptable'
                ],
                'blocking': True
            })
        
        return gates
    
    def _create_monitoring_plan(self, steps: List[MigrationStep], risks: List[MigrationRisk]) -> Dict[str, Any]:
        """Create monitoring plan for migration"""
        return {
            'metrics_to_monitor': [
                'System availability',
                'Response time',
                'Error rates',
                'Memory usage',
                'CPU utilization',
                'Database performance'
            ],
            'monitoring_frequency': {
                'real_time': ['System availability', 'Error rates'],
                'every_5_minutes': ['Response time', 'Resource usage'],
                'every_15_minutes': ['Database performance'],
                'hourly': ['Overall health check']
            },
            'alert_thresholds': {
                'critical': 'System unavailable > 1 minute',
                'warning': 'Response time > 2x baseline',
                'info': 'Resource usage > 80%'
            },
            'dashboards': [
                'Migration Progress Dashboard',
                'System Health Dashboard',
                'Performance Monitoring Dashboard'
            ]
        }
    
    async def validate_migration_plan(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a migration plan for completeness and feasibility
        
        Args:
            plan_data: Migration plan data to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check for required fields
        required_fields = ['steps', 'timeline', 'risks', 'rollback_plan']
        for field in required_fields:
            if field not in plan_data:
                validation_results['errors'].append(f"Missing required field: {field}")
                validation_results['valid'] = False
        
        # Validate steps
        if 'steps' in plan_data:
            steps = plan_data['steps']
            
            # Check for circular dependencies
            if self._has_circular_dependencies(steps):
                validation_results['errors'].append("Circular dependencies detected in migration steps")
                validation_results['valid'] = False
            
            # Check for orphaned steps
            orphaned = self._find_orphaned_steps(steps)
            if orphaned:
                validation_results['warnings'].append(f"Orphaned steps found: {orphaned}")
        
        # Validate timeline
        if 'timeline' in plan_data:
            timeline = plan_data['timeline']
            if timeline.get('total_hours', 0) > 72:
                validation_results['warnings'].append("Migration timeline exceeds 72 hours - consider breaking into phases")
        
        # Validate risks
        if 'risks' in plan_data:
            risks = plan_data['risks']
            critical_risks = [r for r in risks if r.get('severity') == 'critical']
            if len(critical_risks) > 3:
                validation_results['warnings'].append(f"High number of critical risks ({len(critical_risks)}) - consider risk mitigation")
        
        return validation_results
    
    def _has_circular_dependencies(self, steps: List[Dict[str, Any]]) -> bool:
        """Check for circular dependencies in steps"""
        # Build adjacency list
        graph = {}
        for step in steps:
            step_id = step['step_id']
            dependencies = step.get('dependencies', [])
            graph[step_id] = dependencies
        
        # DFS to detect cycles
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            if node not in visited:
                visited.add(node)
                rec_stack.add(node)
                
                for neighbor in graph.get(node, []):
                    if neighbor not in graph:
                        continue
                    if neighbor not in visited:
                        if has_cycle(neighbor):
                            return True
                    elif neighbor in rec_stack:
                        return True
            
            rec_stack.discard(node)
            return False
        
        for step_id in graph:
            if step_id not in visited:
                if has_cycle(step_id):
                    return True
        
        return False
    
    def _find_orphaned_steps(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Find steps that are not dependencies of any other step"""
        step_ids = {step['step_id'] for step in steps}
        referenced_ids = set()
        
        for step in steps:
            for dep in step.get('dependencies', []):
                referenced_ids.add(dep)
        
        # Find steps that are referenced but don't exist
        orphaned = referenced_ids - step_ids
        return list(orphaned)