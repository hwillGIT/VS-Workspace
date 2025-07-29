"""
Manager Agent implementation for task orchestration and coordination.

The Manager Agent is responsible for:
- Breaking down complex tasks into manageable subtasks
- Coordinating execution across multiple agent types
- Managing project state and progress tracking
- Making high-level architectural decisions
- Ensuring quality and consistency across the project
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import dspy
from pydantic import BaseModel, Field

from .base_agent import BaseAgent, AgentMessage


class TaskStatus(Enum):
    """Status of a task or subtask."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Subtask(BaseModel):
    """Represents a subtask in the project decomposition."""
    id: str = Field(description="Unique identifier for the subtask")
    title: str = Field(description="Human-readable title")
    description: str = Field(description="Detailed description of what needs to be done")
    agent_type: str = Field(description="Type of agent best suited for this task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks this depends on")
    estimated_effort: int = Field(default=1, description="Estimated effort in hours")
    assigned_to: Optional[str] = Field(default=None, description="Agent ID if assigned")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task result when completed")
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(default=None)


class ProjectPlan(BaseModel):
    """Represents the overall project plan and state."""
    project_id: str = Field(description="Unique project identifier")
    title: str = Field(description="Project title")
    description: str = Field(description="Project description and requirements")
    subtasks: List[Subtask] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completion_percentage: float = Field(default=0.0)


class TaskDecomposition(dspy.Signature):
    """DSPy signature for breaking down complex tasks into subtasks."""
    
    project_description = dspy.InputField(desc="The project or task to be broken down")
    requirements = dspy.InputField(desc="Specific requirements and constraints")
    context = dspy.InputField(desc="Additional context about the project")
    
    subtasks = dspy.OutputField(desc="JSON list of subtasks with id, title, description, agent_type, priority, dependencies, estimated_effort")
    approach = dspy.OutputField(desc="High-level approach and architectural decisions")
    risks = dspy.OutputField(desc="Identified risks and mitigation strategies")


class TaskCoordination(dspy.Signature):
    """DSPy signature for coordinating task execution and agent assignment."""
    
    current_plan = dspy.InputField(desc="Current project plan state")
    available_agents = dspy.InputField(desc="List of available agents and their capabilities")
    completed_tasks = dspy.InputField(desc="Recently completed tasks and their results")
    
    next_actions = dspy.OutputField(desc="Next tasks to execute with agent assignments")
    plan_updates = dspy.OutputField(desc="Any updates needed to the project plan")
    coordination_notes = dspy.OutputField(desc="Notes about coordination decisions")


class ManagerAgent(BaseAgent):
    """
    Manager Agent responsible for project orchestration and coordination.
    
    This agent acts as the central coordinator for complex development tasks,
    breaking them down into manageable subtasks and coordinating execution
    across specialized agents.
    """
    
    def __init__(self, agent_id: str = "manager", **kwargs):
        super().__init__(agent_id, **kwargs)
        
        # DSPy modules for core functionality
        if self.dspy_enabled:
            self.task_decomposer = dspy.TypedChainOfThought(TaskDecomposition)
            self.task_coordinator = dspy.TypedChainOfThought(TaskCoordination)
        
        # Project management state
        self.current_project: Optional[ProjectPlan] = None
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
        # Coordination settings
        self.max_concurrent_tasks = self.config.get("max_concurrent_tasks", 3)
        self.task_timeout = self.config.get("task_timeout", 300)  # 5 minutes
        
        self.logger.info("Manager Agent initialized and ready for coordination")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Manager Agent."""
        return """You are the Manager Agent in a Self-Reflecting Claude Code Agent system.

Your primary responsibilities are:
1. Breaking down complex development tasks into manageable subtasks
2. Coordinating execution across multiple specialized agents (Coder, Reviewer, Researcher)
3. Managing project state and tracking progress
4. Making high-level architectural and design decisions
5. Ensuring quality and consistency across the entire project

You should think strategically about task decomposition, consider dependencies between tasks,
and make efficient use of available agents. Always prioritize code quality, maintainability,
and following best practices.

When decomposing tasks, consider:
- Logical separation of concerns
- Dependencies between components
- Appropriate agent specialization (coding, reviewing, research)
- Risk factors and mitigation strategies
- Resource constraints and timelines

Communicate clearly with other agents and provide sufficient context for their work."""
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a high-level task by decomposing it and coordinating execution.
        
        Args:
            task: Task specification containing description, requirements, context
            
        Returns:
            Result dictionary with project plan, execution status, and outputs
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing task: {task.get('title', 'Unnamed Task')}")
            
            # Update state
            self.state.current_task = task.get('title', 'Task Execution')
            
            # Create or update project plan
            project_plan = await self._create_project_plan(task)
            self.current_project = project_plan
            
            # Store in memory
            await self.update_memory(
                f"Started project: {project_plan.title}",
                {"task_type": "project_start", "project_id": project_plan.project_id}
            )
            
            # Execute the project plan
            execution_result = await self._execute_project_plan(project_plan)
            
            # Finalize results
            response_time = (datetime.now() - start_time).total_seconds()
            success = execution_result.get("status") == "completed"
            
            self.update_metrics(response_time, success)
            
            result = {
                "status": execution_result.get("status", "completed"),
                "project_plan": project_plan.model_dump(),
                "execution_summary": execution_result,
                "completion_percentage": project_plan.completion_percentage,
                "total_subtasks": len(project_plan.subtasks),
                "completed_subtasks": len([t for t in project_plan.subtasks if t.status == TaskStatus.COMPLETED]),
                "response_time": response_time
            }
            
            await self.update_memory(
                f"Completed project: {project_plan.title} with {project_plan.completion_percentage:.1f}% completion",
                {"task_type": "project_completion", "result": result}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing task: {str(e)}")
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(response_time, False)
            
            return {
                "status": "failed",
                "error": str(e),
                "response_time": response_time
            }
    
    async def _create_project_plan(self, task: Dict[str, Any]) -> ProjectPlan:
        """Create a detailed project plan from a high-level task."""
        
        project_description = task.get("description", "")
        requirements = task.get("requirements", {})
        context = task.get("context", {})
        
        if self.dspy_enabled:
            # Use DSPy for intelligent task decomposition
            result = self.task_decomposer(
                project_description=project_description,
                requirements=json.dumps(requirements, indent=2),
                context=json.dumps(context, indent=2)
            )
            
            # Parse subtasks from DSPy output
            try:
                subtasks_data = json.loads(result.subtasks)
                subtasks = [Subtask(**subtask_data) for subtask_data in subtasks_data]
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"Failed to parse DSPy subtasks, using fallback: {e}")
                subtasks = self._create_fallback_subtasks(task)
                
        else:
            # Fallback decomposition without DSPy
            subtasks = self._create_fallback_subtasks(task)
        
        # Create project plan
        project_plan = ProjectPlan(
            project_id=f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=task.get("title", "Development Project"),
            description=project_description,
            subtasks=subtasks
        )
        
        self.logger.info(f"Created project plan with {len(subtasks)} subtasks")
        return project_plan
    
    def _create_fallback_subtasks(self, task: Dict[str, Any]) -> List[Subtask]:
        """Create a basic task decomposition when DSPy is not available."""
        
        base_subtasks = [
            {
                "id": "research_requirements",
                "title": "Research Requirements and Approach", 
                "description": "Research the requirements and identify the best approach for implementation",
                "agent_type": "researcher",
                "priority": TaskPriority.HIGH,
                "estimated_effort": 2
            },
            {
                "id": "design_architecture",
                "title": "Design System Architecture",
                "description": "Design the overall system architecture and component structure", 
                "agent_type": "coder",
                "priority": TaskPriority.HIGH,
                "dependencies": ["research_requirements"],
                "estimated_effort": 3
            },
            {
                "id": "implement_core",
                "title": "Implement Core Functionality",
                "description": "Implement the main functionality based on the designed architecture",
                "agent_type": "coder", 
                "priority": TaskPriority.HIGH,
                "dependencies": ["design_architecture"],
                "estimated_effort": 5
            },
            {
                "id": "review_implementation",
                "title": "Review Implementation Quality",
                "description": "Review the implemented code for quality, security, and best practices",
                "agent_type": "reviewer",
                "priority": TaskPriority.MEDIUM,
                "dependencies": ["implement_core"],
                "estimated_effort": 2
            },
            {
                "id": "testing_validation",
                "title": "Testing and Validation",
                "description": "Create and run tests to validate the implementation",
                "agent_type": "coder",
                "priority": TaskPriority.MEDIUM, 
                "dependencies": ["review_implementation"],
                "estimated_effort": 3
            }
        ]
        
        return [Subtask(**subtask_data) for subtask_data in base_subtasks]
    
    async def _execute_project_plan(self, project_plan: ProjectPlan) -> Dict[str, Any]:
        """Execute the project plan by coordinating agent tasks."""
        
        execution_start = datetime.now()
        completed_tasks = []
        failed_tasks = []
        
        while True:
            # Get next executable tasks
            ready_tasks = self._get_ready_tasks(project_plan)
            
            if not ready_tasks:
                # Check if we're done or stuck
                pending_tasks = [t for t in project_plan.subtasks if t.status == TaskStatus.PENDING]
                in_progress_tasks = [t for t in project_plan.subtasks if t.status == TaskStatus.IN_PROGRESS]
                
                if not pending_tasks and not in_progress_tasks:
                    # All tasks completed
                    break
                elif not in_progress_tasks:
                    # Stuck - no tasks can proceed
                    self.logger.warning("Project execution stuck - no tasks can proceed")
                    break
                else:
                    # Wait for in-progress tasks
                    await asyncio.sleep(1)
                    continue
            
            # Execute ready tasks (up to max concurrent)
            tasks_to_execute = ready_tasks[:self.max_concurrent_tasks]
            
            # Create execution coroutines
            execution_tasks = []
            for subtask in tasks_to_execute:
                subtask.status = TaskStatus.IN_PROGRESS
                execution_tasks.append(self._execute_subtask(subtask))
            
            # Wait for task completion
            results = await asyncio.gather(*execution_tasks, return_exceptions=True)
            
            # Process results
            for subtask, result in zip(tasks_to_execute, results):
                if isinstance(result, Exception):
                    subtask.status = TaskStatus.FAILED
                    failed_tasks.append(subtask.id)
                    self.logger.error(f"Subtask {subtask.id} failed: {result}")
                else:
                    subtask.status = TaskStatus.COMPLETED
                    subtask.result = result
                    subtask.completed_at = datetime.now()
                    completed_tasks.append(subtask.id)
                    self.logger.info(f"Subtask {subtask.id} completed successfully")
            
            # Update project completion percentage
            total_tasks = len(project_plan.subtasks)
            completed_count = len([t for t in project_plan.subtasks if t.status == TaskStatus.COMPLETED])
            project_plan.completion_percentage = (completed_count / total_tasks) * 100
            project_plan.updated_at = datetime.now()
        
        execution_time = (datetime.now() - execution_start).total_seconds()
        
        return {
            "status": "completed" if not failed_tasks else "partial",
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "execution_time": execution_time,
            "completion_percentage": project_plan.completion_percentage
        }
    
    def _get_ready_tasks(self, project_plan: ProjectPlan) -> List[Subtask]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        
        ready_tasks = []
        completed_task_ids = {t.id for t in project_plan.subtasks if t.status == TaskStatus.COMPLETED}
        
        for subtask in project_plan.subtasks:
            if subtask.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                if all(dep_id in completed_task_ids for dep_id in subtask.dependencies):
                    ready_tasks.append(subtask)
        
        # Sort by priority
        priority_order = {TaskPriority.CRITICAL: 0, TaskPriority.HIGH: 1, TaskPriority.MEDIUM: 2, TaskPriority.LOW: 3}
        ready_tasks.sort(key=lambda t: priority_order[t.priority])
        
        return ready_tasks
    
    async def _execute_subtask(self, subtask: Subtask) -> Dict[str, Any]:
        """Execute a single subtask by delegating to the appropriate agent."""
        
        # Create task specification for the target agent
        task_spec = {
            "id": subtask.id,
            "title": subtask.title,
            "description": subtask.description,
            "context": self.state.context.copy(),
            "project_context": self.current_project.model_dump() if self.current_project else {}
        }
        
        # For now, simulate agent execution
        # In a real implementation, this would delegate to actual agents
        await asyncio.sleep(1)  # Simulate processing time
        
        result = {
            "subtask_id": subtask.id,
            "status": "completed",
            "output": f"Completed {subtask.title}",
            "agent_type": subtask.agent_type,
            "execution_time": 1.0
        }
        
        return result
    
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]) -> None:
        """Register an agent with the manager."""
        self.agent_registry[agent_id] = {
            "type": agent_type,
            "capabilities": capabilities,
            "status": "available",
            "last_seen": datetime.now()
        }
        self.logger.info(f"Registered {agent_type} agent: {agent_id}")
    
    def get_project_status(self) -> Optional[Dict[str, Any]]:
        """Get current project status summary."""
        if not self.current_project:
            return None
            
        return {
            "project_id": self.current_project.project_id,
            "title": self.current_project.title,
            "completion_percentage": self.current_project.completion_percentage,
            "total_subtasks": len(self.current_project.subtasks),
            "completed_subtasks": len([t for t in self.current_project.subtasks if t.status == TaskStatus.COMPLETED]),
            "in_progress_subtasks": len([t for t in self.current_project.subtasks if t.status == TaskStatus.IN_PROGRESS]),
            "failed_subtasks": len([t for t in self.current_project.subtasks if t.status == TaskStatus.FAILED]),
            "created_at": self.current_project.created_at.isoformat(),
            "updated_at": self.current_project.updated_at.isoformat()
        }