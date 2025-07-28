"""
Parallel Planning Engine
Core engine for executing multi-perspective planning workflows.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging

# Import from the main parallel agent system
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from parallel_agent_launcher import AgentTask, AgentResult, ParallelAgentLauncher

logger = logging.getLogger(__name__)


@dataclass
class PlanningPerspective:
    """Represents a planning perspective configuration."""
    perspective_id: str
    name: str
    agent_type: str
    focus_areas: List[str]
    priority: int
    constraints: Dict[str, Any]
    template_path: Optional[str] = None


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


@dataclass
class PerspectivePlan:
    """Represents a plan from a specific perspective."""
    perspective_id: str
    perspective_name: str
    summary: str
    key_decisions: List[str]
    implementation_steps: List[Dict[str, Any]]
    risks: List[Dict[str, Any]]
    assumptions: List[str]
    dependencies: List[str]
    resources_required: Dict[str, Any]
    timeline_estimate: Optional[str] = None
    confidence_level: float = 0.0


@dataclass
class UnifiedPlan:
    """Represents the synthesized unified plan."""
    project_summary: str
    implementation_approach: str
    architecture_overview: str
    development_phases: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    timeline: Dict[str, Any]
    quality_assurance: Dict[str, Any]
    deployment_strategy: Dict[str, Any]
    monitoring_strategy: Dict[str, Any]
    success_criteria: List[str]
    assumptions: List[str]
    constraints: List[str]
    next_steps: List[str]


class ParallelPlanner:
    """Main class for orchestrating parallel planning workflows."""
    
    def __init__(self, config_path: Optional[str] = None, max_concurrent_agents: int = 5):
        self.max_concurrent_agents = max_concurrent_agents
        self.launcher = ParallelAgentLauncher(max_concurrent_agents)
        
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._get_default_config()
        
        # Planning state
        self.context: Optional[PlanningContext] = None
        self.perspectives: List[PlanningPerspective] = []
        self.perspective_plans: Dict[str, PerspectivePlan] = {}
        self.unified_plan: Optional[UnifiedPlan] = None
        self.conflicts: List[Dict[str, Any]] = []
        
        # Initialize perspectives from config
        self._initialize_perspectives()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load planning configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default planning configuration."""
        return {
            "project_type": "software_development",
            "perspectives": [
                {
                    "perspective_id": "technical",
                    "name": "Technical Architecture",
                    "agent_type": "code-architect",
                    "focus_areas": ["architecture", "technology_stack", "implementation_approach"],
                    "priority": 1,
                    "constraints": {}
                },
                {
                    "perspective_id": "security",
                    "name": "Security & Compliance",
                    "agent_type": "code-reviewer",
                    "focus_areas": ["security_controls", "compliance", "threat_modeling"],
                    "priority": 1,
                    "constraints": {}
                },
                {
                    "perspective_id": "performance",
                    "name": "Performance & Scalability",
                    "agent_type": "general-purpose",
                    "focus_areas": ["scalability", "performance", "resource_optimization"],
                    "priority": 2,
                    "constraints": {}
                },
                {
                    "perspective_id": "operational",
                    "name": "Operations & DevOps",
                    "agent_type": "code-architect",
                    "focus_areas": ["deployment", "monitoring", "maintenance"],
                    "priority": 2,
                    "constraints": {}
                }
            ],
            "synthesis": {
                "strategy": "comprehensive",
                "conflict_resolution": "prioritize_security_and_performance",
                "validation_criteria": ["completeness", "consistency", "feasibility"]
            }
        }
    
    def _initialize_perspectives(self):
        """Initialize planning perspectives from configuration."""
        self.perspectives = []
        
        for perspective_config in self.config.get("perspectives", []):
            perspective = PlanningPerspective(
                perspective_id=perspective_config["perspective_id"],
                name=perspective_config["name"],
                agent_type=perspective_config["agent_type"],
                focus_areas=perspective_config["focus_areas"],
                priority=perspective_config["priority"],
                constraints=perspective_config.get("constraints", {}),
                template_path=perspective_config.get("template_path")
            )
            self.perspectives.append(perspective)
    
    def set_planning_context(self, context: PlanningContext):
        """Set the planning context for this session."""
        self.context = context
        logger.info(f"Planning context set for {context.project_type}: {context.problem_description[:100]}...")
    
    async def execute_parallel_planning(self, progress_callback=None) -> Dict[str, Any]:
        """Execute the complete parallel planning workflow."""
        if not self.context:
            raise ValueError("Planning context must be set before executing planning")
        
        logger.info("Starting parallel planning workflow")
        
        # Phase 1: Parallel perspective planning
        await self._execute_perspective_planning(progress_callback)
        
        # Phase 2: Plan synthesis
        await self._execute_plan_synthesis()
        
        # Phase 3: Conflict detection and resolution
        await self._execute_conflict_resolution()
        
        # Phase 4: Plan validation
        await self._execute_plan_validation()
        
        return self._generate_planning_results()
    
    async def _execute_perspective_planning(self, progress_callback=None):
        """Execute planning from all perspectives in parallel."""
        logger.info(f"Executing perspective planning for {len(self.perspectives)} perspectives")
        
        # Create tasks for each perspective
        perspective_tasks = []
        for perspective in self.perspectives:
            task = self._create_perspective_task(perspective)
            perspective_tasks.append(task)
        
        # Add tasks to launcher
        self.launcher.add_batch_tasks(perspective_tasks)
        
        # Execute in parallel
        results = await self.launcher.execute_parallel(progress_callback)
        
        # Process results into perspective plans
        for task_id, result in results.items():
            if result.status == "success":
                perspective_id = task_id.replace("perspective_", "")
                self.perspective_plans[perspective_id] = self._parse_perspective_result(
                    perspective_id, result.result
                )
            else:
                logger.error(f"Perspective planning failed for {task_id}: {result.error}")
    
    def _create_perspective_task(self, perspective: PlanningPerspective) -> AgentTask:
        """Create an agent task for a planning perspective."""
        
        # Build focused prompt for this perspective
        prompt = self._build_perspective_prompt(perspective)
        
        # Create task inputs
        inputs = {
            "perspective": perspective.perspective_id,
            "focus_areas": perspective.focus_areas,
            "context": asdict(self.context) if self.context else {},
            "constraints": perspective.constraints
        }
        
        return AgentTask(
            task_id=f"perspective_{perspective.perspective_id}",
            agent_type=perspective.agent_type,
            description=f"Plan from {perspective.name} perspective",
            prompt=prompt,
            inputs=inputs,
            priority=perspective.priority,
            timeout=600,  # 10 minutes for planning tasks
            retry_count=1
        )
    
    def _build_perspective_prompt(self, perspective: PlanningPerspective) -> str:
        """Build a detailed prompt for a specific planning perspective."""
        base_prompt = f"""
You are a specialized planning agent focusing on {perspective.name}.

## Project Context:
- Project Type: {self.context.project_type}
- Problem Description: {self.context.problem_description}
- Requirements: {json.dumps(self.context.requirements, indent=2) if self.context.requirements else 'None specified'}
- Constraints: {json.dumps(self.context.constraints, indent=2) if self.context.constraints else 'None specified'}

## Your Perspective Focus:
{', '.join(perspective.focus_areas)}

## Planning Instructions:
Please create a detailed plan from your specialized perspective. Your plan should include:

1. **Summary**: Brief overview of your perspective on the solution
2. **Key Decisions**: Critical decisions from your perspective
3. **Implementation Steps**: Detailed steps specific to your domain
4. **Risks**: Risks and challenges from your perspective
5. **Assumptions**: Key assumptions you're making
6. **Dependencies**: Dependencies on other perspectives or external factors
7. **Resources Required**: Resources needed from your perspective
8. **Timeline Estimate**: Rough timeline for your aspects
9. **Success Criteria**: How to measure success from your perspective

## Output Format:
Please structure your response as a detailed plan covering all the above sections.
Focus specifically on {perspective.name} aspects while keeping the overall project context in mind.
"""
        
        # Add perspective-specific guidance
        if perspective.perspective_id == "technical":
            base_prompt += """
## Technical Focus:
- Architecture patterns and design decisions
- Technology stack selection and rationale
- Implementation approach and methodology
- Code organization and structure
- Development tools and frameworks
- Integration patterns and APIs
"""
        elif perspective.perspective_id == "security":
            base_prompt += """
## Security Focus:
- Threat modeling and risk assessment
- Security controls and measures
- Authentication and authorization
- Data protection and privacy
- Compliance requirements
- Security testing and validation
"""
        elif perspective.perspective_id == "performance":
            base_prompt += """
## Performance Focus:
- Scalability requirements and approach
- Performance targets and metrics
- Resource optimization strategies
- Caching and data management
- Load handling and capacity planning
- Performance testing and monitoring
"""
        elif perspective.perspective_id == "operational":
            base_prompt += """
## Operational Focus:
- Deployment strategy and infrastructure
- Monitoring and alerting
- Backup and disaster recovery
- Maintenance and updates
- DevOps practices and automation
- Operational procedures and runbooks
"""
        
        return base_prompt
    
    def _parse_perspective_result(self, perspective_id: str, result: Dict[str, Any]) -> PerspectivePlan:
        """Parse agent result into a structured perspective plan."""
        
        # Extract structured information from the result
        # This is a simplified parser - in reality, you'd need more sophisticated parsing
        
        return PerspectivePlan(
            perspective_id=perspective_id,
            perspective_name=next(p.name for p in self.perspectives if p.perspective_id == perspective_id),
            summary=result.get("summary", f"Plan from {perspective_id} perspective"),
            key_decisions=result.get("key_decisions", []),
            implementation_steps=result.get("implementation_steps", []),
            risks=result.get("risks", []),
            assumptions=result.get("assumptions", []),
            dependencies=result.get("dependencies", []),
            resources_required=result.get("resources_required", {}),
            timeline_estimate=result.get("timeline_estimate"),
            confidence_level=result.get("confidence_level", 0.7)
        )
    
    async def _execute_plan_synthesis(self):
        """Synthesize individual perspective plans into a unified plan."""
        logger.info("Executing plan synthesis")
        
        # Create synthesis task
        synthesis_task = AgentTask(
            task_id="plan_synthesis",
            agent_type="code-architect",
            description="Synthesize multiple perspective plans into unified plan",
            prompt=self._build_synthesis_prompt(),
            inputs={
                "perspective_plans": {pid: asdict(plan) for pid, plan in self.perspective_plans.items()},
                "context": asdict(self.context) if self.context else {},
                "synthesis_strategy": self.config.get("synthesis", {})
            },
            priority=1,
            timeout=900,  # 15 minutes for synthesis
            retry_count=1
        )
        
        # Execute synthesis
        launcher = ParallelAgentLauncher(1)  # Single agent for synthesis
        launcher.add_task(synthesis_task)
        results = await launcher.execute_parallel()
        
        # Process synthesis result
        synthesis_result = results.get("plan_synthesis")
        if synthesis_result and synthesis_result.status == "success":
            self.unified_plan = self._parse_synthesis_result(synthesis_result.result)
        else:
            logger.error(f"Plan synthesis failed: {synthesis_result.error if synthesis_result else 'Unknown error'}")
    
    def _build_synthesis_prompt(self) -> str:
        """Build prompt for plan synthesis."""
        
        plans_summary = "\n\n".join([
            f"## {plan.perspective_name} Plan:\n{plan.summary}"
            for plan in self.perspective_plans.values()
        ])
        
        return f"""
You are a senior architect tasked with synthesizing multiple specialized planning perspectives into a unified, actionable implementation plan.

## Project Context:
- Project Type: {self.context.project_type}
- Problem Description: {self.context.problem_description}

## Perspective Plans to Synthesize:
{plans_summary}

## Synthesis Instructions:
Create a comprehensive, unified implementation plan that:

1. **Integrates all perspectives** - Combine insights from technical, security, performance, and operational perspectives
2. **Resolves conflicts** - Identify and resolve any conflicting recommendations
3. **Prioritizes actions** - Create a logical sequence of implementation phases
4. **Maintains coherence** - Ensure the plan is internally consistent and feasible

## Required Output Structure:
1. **Project Summary** - High-level overview of the unified approach
2. **Implementation Approach** - Overall strategy and methodology
3. **Architecture Overview** - Integrated architecture considering all perspectives
4. **Development Phases** - Phased implementation plan with clear milestones
5. **Risk Assessment** - Consolidated risk analysis with mitigation strategies
6. **Resource Requirements** - Integrated resource planning (people, technology, budget)
7. **Timeline** - Realistic timeline considering all perspective requirements
8. **Quality Assurance** - Testing, validation, and quality measures
9. **Deployment Strategy** - Comprehensive deployment and rollout plan
10. **Monitoring Strategy** - Operational monitoring and success measurement
11. **Success Criteria** - Clear, measurable success criteria
12. **Assumptions** - Key assumptions underlying the plan
13. **Constraints** - Acknowledged constraints and limitations
14. **Next Steps** - Immediate next actions to begin implementation

Focus on creating a practical, actionable plan that balances all perspective requirements.
"""
    
    def _parse_synthesis_result(self, result: Dict[str, Any]) -> UnifiedPlan:
        """Parse synthesis result into structured unified plan."""
        
        return UnifiedPlan(
            project_summary=result.get("project_summary", ""),
            implementation_approach=result.get("implementation_approach", ""),
            architecture_overview=result.get("architecture_overview", ""),
            development_phases=result.get("development_phases", []),
            risk_assessment=result.get("risk_assessment", {}),
            resource_requirements=result.get("resource_requirements", {}),
            timeline=result.get("timeline", {}),
            quality_assurance=result.get("quality_assurance", {}),
            deployment_strategy=result.get("deployment_strategy", {}),
            monitoring_strategy=result.get("monitoring_strategy", {}),
            success_criteria=result.get("success_criteria", []),
            assumptions=result.get("assumptions", []),
            constraints=result.get("constraints", []),
            next_steps=result.get("next_steps", [])
        )
    
    async def _execute_conflict_resolution(self):
        """Detect and resolve conflicts between perspectives."""
        logger.info("Executing conflict detection and resolution")
        
        # Simple conflict detection (in practice, this would be more sophisticated)
        conflicts = []
        
        # Check for timeline conflicts
        timelines = [plan.timeline_estimate for plan in self.perspective_plans.values() if plan.timeline_estimate]
        if len(set(timelines)) > 1:
            conflicts.append({
                "type": "timeline_conflict",
                "description": "Different perspectives have conflicting timeline estimates",
                "perspectives": list(self.perspective_plans.keys()),
                "values": timelines
            })
        
        # Check for resource conflicts
        # (This would be more sophisticated in practice)
        
        self.conflicts = conflicts
        logger.info(f"Detected {len(conflicts)} conflicts")
    
    async def _execute_plan_validation(self):
        """Validate the unified plan for completeness and consistency."""
        logger.info("Executing plan validation")
        
        validation_criteria = self.config.get("synthesis", {}).get("validation_criteria", [])
        
        # Simple validation (in practice, this would be more comprehensive)
        if self.unified_plan:
            validation_issues = []
            
            # Check completeness
            if not self.unified_plan.implementation_approach:
                validation_issues.append("Missing implementation approach")
            
            if not self.unified_plan.development_phases:
                validation_issues.append("Missing development phases")
            
            # Check consistency
            # (Add consistency checks here)
            
            if validation_issues:
                logger.warning(f"Plan validation found issues: {validation_issues}")
        else:
            logger.error("No unified plan available for validation")
    
    def _generate_planning_results(self) -> Dict[str, Any]:
        """Generate comprehensive planning results."""
        
        return {
            "context": asdict(self.context) if self.context else {},
            "perspective_plans": {
                pid: asdict(plan) for pid, plan in self.perspective_plans.items()
            },
            "unified_plan": asdict(self.unified_plan) if self.unified_plan else None,
            "conflicts": self.conflicts,
            "execution_summary": {
                "total_perspectives": len(self.perspectives),
                "successful_perspectives": len(self.perspective_plans),
                "conflicts_detected": len(self.conflicts),
                "plan_generated": self.unified_plan is not None
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def save_results(self, output_path: Path):
        """Save planning results to file."""
        results = self._generate_planning_results()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Planning results saved to {output_path}")


# Utility functions for common planning scenarios

def create_software_development_planner(project_description: str, requirements: Dict[str, Any]) -> ParallelPlanner:
    """Create a planner configured for software development projects."""
    
    config = {
        "project_type": "software_development",
        "perspectives": [
            {
                "perspective_id": "technical",
                "name": "Technical Architecture",
                "agent_type": "code-architect",
                "focus_areas": ["architecture", "technology_stack", "implementation"],
                "priority": 1,
                "constraints": {}
            },
            {
                "perspective_id": "security",
                "name": "Security & Compliance", 
                "agent_type": "code-reviewer",
                "focus_areas": ["security_controls", "compliance", "threat_modeling"],
                "priority": 1,
                "constraints": {}
            },
            {
                "perspective_id": "performance",
                "name": "Performance & Scalability",
                "agent_type": "general-purpose", 
                "focus_areas": ["scalability", "performance", "optimization"],
                "priority": 2,
                "constraints": {}
            },
            {
                "perspective_id": "user_experience",
                "name": "User Experience",
                "agent_type": "general-purpose",
                "focus_areas": ["usability", "accessibility", "user_interface"],
                "priority": 2,
                "constraints": {}
            },
            {
                "perspective_id": "operational",
                "name": "Operations & DevOps",
                "agent_type": "code-architect",
                "focus_areas": ["deployment", "monitoring", "maintenance"],
                "priority": 3,
                "constraints": {}
            }
        ]
    }
    
    planner = ParallelPlanner()
    planner.config = config
    planner._initialize_perspectives()
    
    context = PlanningContext(
        project_type="software_development",
        problem_description=project_description,
        requirements=requirements,
        constraints={},
        stakeholders=["development_team", "product_owner", "users"]
    )
    planner.set_planning_context(context)
    
    return planner


def create_trading_system_planner(system_description: str, requirements: Dict[str, Any]) -> ParallelPlanner:
    """Create a planner configured for trading system projects."""
    
    config = {
        "project_type": "trading_system",
        "perspectives": [
            {
                "perspective_id": "trading_logic",
                "name": "Trading Logic & Strategy",
                "agent_type": "general-purpose",
                "focus_areas": ["trading_algorithms", "strategy_implementation", "signal_processing"],
                "priority": 1,
                "constraints": {}
            },
            {
                "perspective_id": "risk_management",
                "name": "Risk Management",
                "agent_type": "code-reviewer",
                "focus_areas": ["risk_controls", "position_limits", "var_calculation"],
                "priority": 1,
                "constraints": {}
            },
            {
                "perspective_id": "performance",
                "name": "Performance & Latency",
                "agent_type": "code-architect",
                "focus_areas": ["low_latency", "high_throughput", "real_time_processing"],
                "priority": 1,
                "constraints": {}
            },
            {
                "perspective_id": "data_management",
                "name": "Market Data Management",
                "agent_type": "code-architect",
                "focus_areas": ["data_ingestion", "data_quality", "historical_data"],
                "priority": 2,
                "constraints": {}
            },
            {
                "perspective_id": "compliance",
                "name": "Regulatory Compliance",
                "agent_type": "code-reviewer",
                "focus_areas": ["regulatory_requirements", "audit_trails", "reporting"],
                "priority": 2,
                "constraints": {}
            }
        ]
    }
    
    planner = ParallelPlanner()
    planner.config = config
    planner._initialize_perspectives()
    
    context = PlanningContext(
        project_type="trading_system",
        problem_description=system_description,
        requirements=requirements,
        constraints={},
        stakeholders=["traders", "risk_managers", "compliance_officers", "it_operations"]
    )
    planner.set_planning_context(context)
    
    return planner