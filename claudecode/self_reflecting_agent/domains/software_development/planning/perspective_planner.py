"""
Multi-Perspective Planning for Software Development

Integrates parallel planning concepts with domain-specific agents to provide
comprehensive project planning from multiple expert perspectives.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from ....agents.base_agent import BaseAgent


@dataclass
class PlanningPerspective:
    """Represents a planning perspective configuration."""
    perspective_id: str
    name: str
    agent_name: str  # Which domain agent to use
    focus_areas: List[str]
    priority: int
    constraints: Dict[str, Any]
    prompt_template: Dict[str, Any]
    timeout: int = 300


@dataclass 
class PlanningContext:
    """Contains the context for a planning session."""
    project_type: str
    problem_description: str
    requirements: Dict[str, Any]
    constraints: Dict[str, Any]
    stakeholders: List[str]
    timeline: Optional[str] = None
    budget: Optional[str] = None
    target_audience: Optional[Dict[str, Any]] = None


@dataclass
class PerspectivePlan:
    """Represents a plan from a specific perspective."""
    perspective_id: str
    perspective_name: str
    agent_name: str
    plan_content: Dict[str, Any]
    confidence_score: float
    execution_time: float
    recommendations: List[str]
    concerns: List[str]
    dependencies: List[str]
    created_at: datetime


@dataclass
class SynthesizedPlan:
    """Final synthesized plan from all perspectives."""
    project_type: str
    executive_summary: str
    technical_specification: Dict[str, Any]
    implementation_roadmap: Dict[str, Any]
    perspective_plans: List[PerspectivePlan]
    conflict_resolutions: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    success_criteria: List[str]
    created_at: datetime


class PerspectivePlanner:
    """
    Multi-perspective planner for software development projects.
    
    Coordinates multiple domain agents to plan projects from different
    expert perspectives, then synthesizes into a unified plan.
    """
    
    def __init__(self, domain_manager, config: Dict[str, Any]):
        self.domain_manager = domain_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Planning configuration
        self.max_concurrent_perspectives = config.get("max_concurrent_perspectives", 5)
        self.synthesis_strategy = config.get("synthesis_strategy", "consensus_weighted")
        self.conflict_resolution = config.get("conflict_resolution", "expert_priority")
        
        # Loaded perspectives and templates
        self.available_perspectives: Dict[str, PlanningPerspective] = {}
        self.planning_templates: Dict[str, Dict[str, Any]] = {}
        
        self._load_default_perspectives()
    
    def _load_default_perspectives(self):
        """Load default planning perspectives for software development."""
        
        # Architecture Perspective
        self.available_perspectives["architecture"] = PlanningPerspective(
            perspective_id="architecture",
            name="System Architecture",
            agent_name="architect",
            focus_areas=[
                "system_design", "scalability_analysis", "architecture_patterns",
                "technical_debt_assessment", "component_design"
            ],
            priority=1,
            constraints={"scalability_required": True, "maintainability_focus": True},
            prompt_template={
                "base_instruction": "You are a system architect planning the overall architecture for this project.",
                "focus_instruction": "Design a scalable, maintainable system architecture that meets all requirements.",
                "specific_areas": [
                    "System architecture design and component interaction",
                    "Scalability planning for expected load and growth",
                    "Technology stack selection and justification",
                    "Design patterns and architectural patterns to use",
                    "Technical debt prevention and code organization"
                ],
                "output_requirements": [
                    "High-level system architecture diagram",
                    "Component specifications and interactions", 
                    "Technology stack recommendations with rationale",
                    "Scalability strategy and implementation plan",
                    "Architecture decision records (ADRs)"
                ]
            }
        )
        
        # Security Perspective
        self.available_perspectives["security"] = PlanningPerspective(
            perspective_id="security",
            name="Security & Compliance",
            agent_name="security_auditor",
            focus_areas=[
                "vulnerability_assessment", "secure_coding_practices",
                "compliance_analysis", "threat_modeling", "data_protection"
            ],
            priority=1,
            constraints={"security_mandatory": True, "compliance_required": True},
            prompt_template={
                "base_instruction": "You are a security expert ensuring comprehensive security planning.",
                "focus_instruction": "Design security framework covering all attack vectors and compliance requirements.",
                "specific_areas": [
                    "Threat modeling and security risk assessment",
                    "Authentication and authorization design",
                    "Data protection and encryption strategies",
                    "Input validation and security testing",
                    "Compliance requirements (GDPR, SOC2, etc.)"
                ],
                "output_requirements": [
                    "Security architecture specification",
                    "Threat model and risk assessment",
                    "Authentication/authorization implementation plan",
                    "Data protection and privacy compliance strategy",
                    "Security testing and monitoring framework"
                ]
            }
        )
        
        # Performance Perspective
        self.available_perspectives["performance"] = PlanningPerspective(
            perspective_id="performance",
            name="Performance & Optimization",
            agent_name="performance_auditor",
            focus_areas=[
                "performance_profiling", "optimization_recommendations",
                "resource_usage_analysis", "bottleneck_identification", "caching_strategies"
            ],
            priority=2,
            constraints={"performance_targets": True, "resource_efficiency": True},
            prompt_template={
                "base_instruction": "You are a performance specialist optimizing system performance.",
                "focus_instruction": "Design performance optimization strategy covering all system layers.",
                "specific_areas": [
                    "Performance requirements analysis and benchmarking",
                    "Database optimization and query performance",
                    "Caching strategies (application, database, CDN)",
                    "Resource usage optimization (CPU, memory, I/O)",
                    "Performance monitoring and alerting systems"
                ],
                "output_requirements": [
                    "Performance optimization strategy",
                    "Database and query optimization plan",
                    "Caching architecture and implementation",
                    "Resource monitoring and alerting framework",
                    "Performance testing and validation plan"
                ]
            }
        )
        
        # Code Quality Perspective
        self.available_perspectives["code_quality"] = PlanningPerspective(
            perspective_id="code_quality",
            name="Code Quality & Maintainability",
            agent_name="design_patterns_expert",
            focus_areas=[
                "pattern_identification", "refactoring_guidance",
                "code_organization", "maintainability", "testing_strategy"
            ],
            priority=2,
            constraints={"maintainability_required": True, "testing_mandatory": True},
            prompt_template={
                "base_instruction": "You are a code quality expert ensuring maintainable, well-designed code.",
                "focus_instruction": "Plan code organization, patterns, and quality practices for long-term maintainability.",
                "specific_areas": [
                    "Code organization and project structure",
                    "Design patterns and architectural patterns to implement",
                    "Testing strategy (unit, integration, e2e)",
                    "Code review processes and quality gates",
                    "Documentation and knowledge sharing practices"
                ],
                "output_requirements": [
                    "Code organization and structure plan",
                    "Design patterns implementation guide",
                    "Comprehensive testing strategy",
                    "Code quality standards and review process",
                    "Documentation and knowledge management plan"
                ]
            }
        )
        
        # DevOps Perspective
        self.available_perspectives["devops"] = PlanningPerspective(
            perspective_id="devops",
            name="DevOps & Deployment",
            agent_name="migration_planner",  # Using migration planner for deployment planning
            focus_areas=[
                "deployment_strategies", "ci_cd_pipeline", "infrastructure_management",
                "monitoring_logging", "backup_recovery"
            ],
            priority=2,
            constraints={"automation_required": True, "reliability_standards": True},
            prompt_template={
                "base_instruction": "You are a DevOps expert planning deployment and operations.",
                "focus_instruction": "Design automated, reliable deployment and operations framework.",
                "specific_areas": [
                    "CI/CD pipeline design and automation",
                    "Infrastructure as code and environment management",
                    "Containerization and orchestration strategy",
                    "Monitoring, logging, and alerting systems",
                    "Backup, disaster recovery, and incident response"
                ],
                "output_requirements": [
                    "CI/CD pipeline specification and implementation",
                    "Infrastructure as code templates and management",
                    "Containerization and deployment strategy",
                    "Monitoring and observability framework",
                    "Disaster recovery and incident response procedures"
                ]
            }
        )
    
    async def create_multi_perspective_plan(
        self,
        context: PlanningContext,
        selected_perspectives: Optional[List[str]] = None
    ) -> SynthesizedPlan:
        """
        Create a comprehensive plan using multiple perspectives.
        
        Args:
            context: Planning context with requirements and constraints
            selected_perspectives: Optional list of perspective IDs to use
            
        Returns:
            Synthesized plan combining all perspectives
        """
        
        try:
            self.logger.info(f"Starting multi-perspective planning for {context.project_type}")
            
            # Determine which perspectives to use
            perspectives_to_use = selected_perspectives or list(self.available_perspectives.keys())
            
            # Execute planning from all perspectives in parallel
            perspective_plans = await self._execute_parallel_planning(context, perspectives_to_use)
            
            # Synthesize plans into unified approach
            synthesized_plan = await self._synthesize_plans(context, perspective_plans)
            
            # Validate the synthesized plan
            validation_results = await self._validate_synthesized_plan(synthesized_plan)
            synthesized_plan.validation_results = validation_results
            
            self.logger.info(f"Multi-perspective planning completed with {len(perspective_plans)} perspectives")
            return synthesized_plan
            
        except Exception as e:
            self.logger.error(f"Multi-perspective planning failed: {e}")
            raise
    
    async def _execute_parallel_planning(
        self,
        context: PlanningContext,
        perspective_ids: List[str]
    ) -> List[PerspectivePlan]:
        """Execute planning from multiple perspectives in parallel."""
        
        # Create planning tasks
        planning_tasks = []
        for perspective_id in perspective_ids:
            if perspective_id in self.available_perspectives:
                perspective = self.available_perspectives[perspective_id]
                task = self._plan_from_perspective(context, perspective)
                planning_tasks.append((perspective_id, task))
        
        # Execute all planning tasks concurrently
        results = []
        if planning_tasks:
            task_results = await asyncio.gather(
                *[task for _, task in planning_tasks],
                return_exceptions=True
            )
            
            for (perspective_id, _), result in zip(planning_tasks, task_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Planning failed for perspective {perspective_id}: {result}")
                else:
                    results.append(result)
        
        return results
    
    async def _plan_from_perspective(
        self,
        context: PlanningContext,
        perspective: PlanningPerspective
    ) -> PerspectivePlan:
        """Plan from a specific perspective using the corresponding domain agent."""
        
        start_time = datetime.now()
        
        try:
            # Get the domain agent for this perspective
            agent = self.domain_manager.get_domain_agent("software_development", perspective.agent_name)
            
            if not agent:
                raise ValueError(f"Agent {perspective.agent_name} not available for perspective {perspective.perspective_id}")
            
            # Prepare planning prompt
            planning_prompt = self._generate_perspective_prompt(context, perspective)
            
            # Execute planning using the agent
            planning_result = await agent.process_task(
                task=planning_prompt,
                context={
                    "perspective": perspective.perspective_id,
                    "focus_areas": perspective.focus_areas,
                    "constraints": perspective.constraints,
                    "planning_context": asdict(context)
                }
            )
            
            # Extract plan components
            plan_content = self._extract_plan_content(planning_result, perspective)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return PerspectivePlan(
                perspective_id=perspective.perspective_id,
                perspective_name=perspective.name,
                agent_name=perspective.agent_name,
                plan_content=plan_content,
                confidence_score=plan_content.get("confidence_score", 0.8),
                execution_time=execution_time,
                recommendations=plan_content.get("recommendations", []),
                concerns=plan_content.get("concerns", []),
                dependencies=plan_content.get("dependencies", []),
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Perspective planning failed for {perspective.perspective_id}: {e}")
            
            # Return minimal plan with error information
            return PerspectivePlan(
                perspective_id=perspective.perspective_id,
                perspective_name=perspective.name,
                agent_name=perspective.agent_name,
                plan_content={"error": str(e), "status": "failed"},
                confidence_score=0.0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                recommendations=[],
                concerns=[f"Planning failed: {str(e)}"],
                dependencies=[],
                created_at=datetime.now()
            )
    
    def _generate_perspective_prompt(
        self,
        context: PlanningContext,
        perspective: PlanningPerspective
    ) -> str:
        """Generate a planning prompt for a specific perspective."""
        
        template = perspective.prompt_template
        
        prompt = f"""
{template['base_instruction']}

PROJECT CONTEXT:
- Project Type: {context.project_type}
- Problem Description: {context.problem_description}
- Timeline: {context.timeline or 'Not specified'}
- Budget: {context.budget or 'Not specified'}

REQUIREMENTS:
{self._format_requirements(context.requirements)}

CONSTRAINTS:
{self._format_constraints(context.constraints)}

PERSPECTIVE FOCUS:
{template['focus_instruction']}

SPECIFIC AREAS TO ADDRESS:
{self._format_list(template['specific_areas'])}

REQUIRED OUTPUTS:
{self._format_list(template['output_requirements'])}

Please provide a comprehensive plan from your perspective that addresses all the specific areas and delivers the required outputs. Include:
1. Detailed recommendations for your area of expertise
2. Potential concerns or risks you identify
3. Dependencies on other perspectives or external factors
4. Confidence level in your recommendations (0.0 to 1.0)
"""
        
        return prompt
    
    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format requirements for the prompt."""
        formatted = []
        for category, reqs in requirements.items():
            formatted.append(f"- {category.title()}: {reqs}")
        return "\n".join(formatted)
    
    def _format_constraints(self, constraints: Dict[str, Any]) -> str:
        """Format constraints for the prompt."""
        formatted = []
        for category, cons in constraints.items():
            formatted.append(f"- {category.title()}: {cons}")
        return "\n".join(formatted)
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list for the prompt."""
        return "\n".join(f"- {item}" for item in items)
    
    def _extract_plan_content(
        self,
        planning_result: Dict[str, Any],
        perspective: PlanningPerspective
    ) -> Dict[str, Any]:
        """Extract structured plan content from agent result."""
        
        # This would parse the agent's response and extract structured information
        # For now, return the result as-is with some basic parsing
        
        return {
            "raw_result": planning_result,
            "recommendations": planning_result.get("recommendations", []),
            "concerns": planning_result.get("concerns", []),
            "dependencies": planning_result.get("dependencies", []),
            "confidence_score": planning_result.get("confidence_score", 0.8),
            "technical_details": planning_result.get("technical_details", {}),
            "implementation_notes": planning_result.get("implementation_notes", "")
        }
    
    async def _synthesize_plans(
        self,
        context: PlanningContext,
        perspective_plans: List[PerspectivePlan]
    ) -> SynthesizedPlan:
        """Synthesize multiple perspective plans into a unified plan."""
        
        try:
            # Create executive summary
            executive_summary = self._create_executive_summary(context, perspective_plans)
            
            # Create technical specification
            technical_spec = self._create_technical_specification(perspective_plans)
            
            # Create implementation roadmap
            roadmap = self._create_implementation_roadmap(perspective_plans)
            
            # Identify and resolve conflicts
            conflict_resolutions = self._resolve_conflicts(perspective_plans)
            
            # Extract success criteria
            success_criteria = self._extract_success_criteria(perspective_plans)
            
            return SynthesizedPlan(
                project_type=context.project_type,
                executive_summary=executive_summary,
                technical_specification=technical_spec,
                implementation_roadmap=roadmap,
                perspective_plans=perspective_plans,
                conflict_resolutions=conflict_resolutions,
                validation_results={},  # Will be filled by validation
                success_criteria=success_criteria,
                created_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Plan synthesis failed: {e}")
            raise
    
    def _create_executive_summary(
        self,
        context: PlanningContext,
        plans: List[PerspectivePlan]
    ) -> str:
        """Create executive summary from all perspective plans."""
        
        summary_parts = [
            f"Project: {context.project_type}",
            f"Problem: {context.problem_description}",
            f"Perspectives Analyzed: {len(plans)}",
            "",
            "Key Findings:"
        ]
        
        for plan in plans:
            if plan.recommendations:
                summary_parts.append(f"- {plan.perspective_name}: {plan.recommendations[0] if plan.recommendations else 'No specific recommendations'}")
        
        return "\n".join(summary_parts)
    
    def _create_technical_specification(self, plans: List[PerspectivePlan]) -> Dict[str, Any]:
        """Create unified technical specification."""
        
        spec = {
            "architecture": {},
            "security": {},
            "performance": {},
            "quality": {},
            "deployment": {}
        }
        
        for plan in plans:
            if plan.perspective_id == "architecture":
                spec["architecture"] = plan.plan_content.get("technical_details", {})
            elif plan.perspective_id == "security":
                spec["security"] = plan.plan_content.get("technical_details", {})
            elif plan.perspective_id == "performance":
                spec["performance"] = plan.plan_content.get("technical_details", {})
            elif plan.perspective_id == "code_quality":
                spec["quality"] = plan.plan_content.get("technical_details", {})
            elif plan.perspective_id == "devops":
                spec["deployment"] = plan.plan_content.get("technical_details", {})
        
        return spec
    
    def _create_implementation_roadmap(self, plans: List[PerspectivePlan]) -> Dict[str, Any]:
        """Create implementation roadmap from all plans."""
        
        roadmap = {
            "phases": [
                "Project Setup and Architecture",
                "Core Development",
                "Security Implementation", 
                "Performance Optimization",
                "Quality Assurance",
                "Deployment and Operations"
            ],
            "dependencies": [],
            "milestones": [],
            "timeline": "To be determined based on project scope"
        }
        
        # Extract dependencies from all plans
        for plan in plans:
            roadmap["dependencies"].extend(plan.dependencies)
        
        return roadmap
    
    def _resolve_conflicts(self, plans: List[PerspectivePlan]) -> List[Dict[str, Any]]:
        """Identify and resolve conflicts between perspective plans."""
        
        conflicts = []
        
        # This would implement sophisticated conflict detection and resolution
        # For now, return basic conflict information
        
        return conflicts
    
    def _extract_success_criteria(self, plans: List[PerspectivePlan]) -> List[str]:
        """Extract success criteria from all perspective plans."""
        
        criteria = []
        
        for plan in plans:
            plan_criteria = plan.plan_content.get("success_criteria", [])
            criteria.extend(plan_criteria)
        
        # Remove duplicates
        return list(set(criteria))
    
    async def _validate_synthesized_plan(self, plan: SynthesizedPlan) -> Dict[str, Any]:
        """Validate the synthesized plan for completeness and consistency."""
        
        validation = {
            "completeness_score": 0.0,
            "consistency_score": 0.0,
            "feasibility_score": 0.0,
            "overall_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # Validate completeness
        if plan.technical_specification and plan.implementation_roadmap:
            validation["completeness_score"] = 0.8
        
        # Validate consistency
        if len(plan.conflict_resolutions) == 0:
            validation["consistency_score"] = 0.9
        
        # Validate feasibility
        if plan.perspective_plans:
            avg_confidence = sum(p.confidence_score for p in plan.perspective_plans) / len(plan.perspective_plans)
            validation["feasibility_score"] = avg_confidence
        
        # Calculate overall score
        validation["overall_score"] = (
            validation["completeness_score"] * 0.4 +
            validation["consistency_score"] * 0.3 +
            validation["feasibility_score"] * 0.3
        )
        
        return validation
    
    def list_available_perspectives(self) -> List[str]:
        """List all available planning perspectives."""
        return list(self.available_perspectives.keys())
    
    def get_perspective_info(self, perspective_id: str) -> Optional[PlanningPerspective]:
        """Get information about a specific perspective."""
        return self.available_perspectives.get(perspective_id)