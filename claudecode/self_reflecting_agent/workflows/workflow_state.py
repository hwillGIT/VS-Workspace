"""
State management for LangGraph workflows.

This module defines the state structures used throughout the workflow execution,
including task state, agent state, and overall workflow state management.
"""

from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class TaskStatus(Enum):
    """Status of a task in the workflow."""
    PENDING = "pending"
    PLANNING = "planning"
    RESEARCHING = "researching"
    IMPLEMENTING = "implementing"
    REVIEWING = "reviewing"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class AgentType(Enum):
    """Types of agents in the system."""
    MANAGER = "manager"
    CODER = "coder"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"


class WorkflowPhase(Enum):
    """Phases of the development workflow."""
    INITIALIZATION = "initialization"
    RESEARCH = "research"
    PLANNING = "planning"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    FINALIZATION = "finalization"


@dataclass
class TaskState:
    """Represents the state of a single task."""
    task_id: str
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def update_status(self, new_status: TaskStatus, error_message: Optional[str] = None):
        """Update task status with timestamp."""
        self.status = new_status
        self.updated_at = datetime.now()
        
        if new_status == TaskStatus.COMPLETED:
            self.completed_at = datetime.now()
        elif new_status == TaskStatus.FAILED:
            self.error_message = error_message
    
    def can_retry(self) -> bool:
        """Check if the task can be retried."""
        return self.retry_count < self.max_retries and self.status == TaskStatus.FAILED
    
    def retry(self) -> None:
        """Mark task for retry."""
        if self.can_retry():
            self.retry_count += 1
            self.status = TaskStatus.PENDING
            self.updated_at = datetime.now()
            self.error_message = None


@dataclass
class AgentState:
    """Represents the state of an agent."""
    agent_id: str
    agent_type: AgentType
    status: str = "idle"  # idle, busy, error, offline
    current_task: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_activity: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    success_count: int = 0
    
    def assign_task(self, task_id: str):
        """Assign a task to this agent."""
        self.current_task = task_id
        self.status = "busy"
        self.last_activity = datetime.now()
    
    def complete_task(self, success: bool = True):
        """Mark task as completed for this agent."""
        self.current_task = None
        self.status = "idle"
        self.last_activity = datetime.now()
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_success_rate(self) -> float:
        """Calculate success rate for this agent."""
        total_tasks = self.success_count + self.error_count
        return self.success_count / total_tasks if total_tasks > 0 else 0.0


class WorkflowState(BaseModel):
    """
    Main workflow state that tracks the entire development process.
    
    This class maintains the state of all tasks, agents, and workflow execution
    throughout the development lifecycle.
    """
    
    # Workflow identification
    workflow_id: str = Field(description="Unique workflow identifier")
    project_title: str = Field(description="Title of the project being developed")
    project_description: str = Field(description="Description of the project requirements")
    
    # Current workflow state
    current_phase: WorkflowPhase = Field(default=WorkflowPhase.INITIALIZATION)
    status: str = Field(default="initializing", description="Overall workflow status")
    
    # Task management
    tasks: Dict[str, TaskState] = Field(default_factory=dict, description="All tasks in the workflow")
    task_execution_order: List[str] = Field(default_factory=list, description="Planned task execution order")
    
    # Agent management  
    agents: Dict[str, AgentState] = Field(default_factory=dict, description="All agents in the workflow")
    agent_assignments: Dict[str, str] = Field(default_factory=dict, description="Current task assignments (task_id -> agent_id)")
    
    # Workflow results and artifacts
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Workflow outputs and artifacts")
    deliverables: List[str] = Field(default_factory=list, description="List of deliverable file paths")
    
    # Context and memory
    context: Dict[str, Any] = Field(default_factory=dict, description="Shared workflow context")
    memory_keys: List[str] = Field(default_factory=list, description="Memory keys for retrieval")
    
    # Execution tracking
    started_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = Field(default=None)
    
    # Quality and metrics
    quality_gates: Dict[str, bool] = Field(default_factory=dict, description="Quality gate results")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Workflow metrics")
    
    # Error handling
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Recorded errors")
    warnings: List[Dict[str, Any]] = Field(default_factory=list, description="Recorded warnings")
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_task(
        self,
        task_id: str,
        title: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskState:
        """Add a new task to the workflow."""
        
        task = TaskState(
            task_id=task_id,
            title=title,
            description=description,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        self.updated_at = datetime.now()
        
        return task
    
    def add_agent(
        self,
        agent_id: str,
        agent_type: AgentType,
        capabilities: Optional[List[str]] = None
    ) -> AgentState:
        """Add an agent to the workflow."""
        
        agent = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities or []
        )
        
        self.agents[agent_id] = agent
        self.updated_at = datetime.now()
        
        return agent
    
    def assign_task_to_agent(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent."""
        
        if task_id not in self.tasks or agent_id not in self.agents:
            return False
        
        task = self.tasks[task_id]
        agent = self.agents[agent_id]
        
        # Check if agent is available
        if agent.status != "idle":
            return False
        
        # Check if task dependencies are met
        if not self._are_dependencies_met(task_id):
            return False
        
        # Make the assignment
        self.agent_assignments[task_id] = agent_id
        task.assigned_agent = agent_id
        task.update_status(TaskStatus.PENDING)
        agent.assign_task(task_id)
        
        self.updated_at = datetime.now()
        return True
    
    def complete_task(self, task_id: str, success: bool = True, outputs: Optional[Dict[str, Any]] = None):
        """Mark a task as completed."""
        
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        # Update task state
        if success:
            task.update_status(TaskStatus.COMPLETED)
            if outputs:
                task.outputs.update(outputs)
        else:
            task.update_status(TaskStatus.FAILED)
        
        # Update agent state
        if task.assigned_agent and task.assigned_agent in self.agents:
            agent = self.agents[task.assigned_agent]
            agent.complete_task(success)
        
        # Remove assignment
        if task_id in self.agent_assignments:
            del self.agent_assignments[task_id]
        
        self.updated_at = datetime.now()
        return True
    
    def get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to be executed (dependencies met)."""
        
        ready_tasks = []
        
        for task_id, task in self.tasks.items():
            if (task.status == TaskStatus.PENDING and 
                task.assigned_agent is None and
                self._are_dependencies_met(task_id)):
                ready_tasks.append(task_id)
        
        return ready_tasks
    
    def get_available_agents(self) -> List[str]:
        """Get agents that are currently available."""
        
        return [agent_id for agent_id, agent in self.agents.items() 
                if agent.status == "idle"]
    
    def _are_dependencies_met(self, task_id: str) -> bool:
        """Check if all dependencies for a task are completed."""
        
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            
            dep_task = self.tasks[dep_id]
            if dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def update_phase(self, new_phase: WorkflowPhase):
        """Update the current workflow phase."""
        self.current_phase = new_phase
        self.updated_at = datetime.now()
    
    def add_output(self, key: str, value: Any):
        """Add an output to the workflow."""
        self.outputs[key] = value
        self.updated_at = datetime.now()
    
    def add_deliverable(self, file_path: str):
        """Add a deliverable file to the workflow."""
        if file_path not in self.deliverables:
            self.deliverables.append(file_path)
            self.updated_at = datetime.now()
    
    def update_context(self, key: str, value: Any):
        """Update the workflow context."""
        self.context[key] = value
        self.updated_at = datetime.now()
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the workflow context."""
        return self.context.get(key, default)
    
    def add_error(self, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Add an error to the workflow."""
        error_entry = {
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        self.errors.append(error_entry)
        self.updated_at = datetime.now()
    
    def add_warning(self, warning_message: str, context: Optional[Dict[str, Any]] = None):
        """Add a warning to the workflow."""
        warning_entry = {
            "message": warning_message,
            "timestamp": datetime.now().isoformat(),
            "context": context or {}
        }
        self.warnings.append(warning_entry)
        self.updated_at = datetime.now()
    
    def set_quality_gate(self, gate_name: str, passed: bool):
        """Set the result of a quality gate."""
        self.quality_gates[gate_name] = passed
        self.updated_at = datetime.now()
    
    def update_metric(self, metric_name: str, value: float):
        """Update a workflow metric."""
        self.metrics[metric_name] = value
        self.updated_at = datetime.now()
    
    def get_completion_percentage(self) -> float:
        """Calculate workflow completion percentage."""
        if not self.tasks:
            return 0.0
        
        completed_tasks = sum(1 for task in self.tasks.values() 
                            if task.status == TaskStatus.COMPLETED)
        return (completed_tasks / len(self.tasks)) * 100
    
    def get_task_summary(self) -> Dict[str, int]:
        """Get a summary of task statuses."""
        summary = {}
        
        for status in TaskStatus:
            summary[status.value] = sum(1 for task in self.tasks.values() 
                                      if task.status == status)
        
        return summary
    
    def get_agent_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of agent states."""
        summary = {}
        
        for agent_id, agent in self.agents.items():
            summary[agent_id] = {
                "type": agent.agent_type.value,
                "status": agent.status,
                "current_task": agent.current_task,
                "success_rate": agent.get_success_rate(),
                "total_tasks": agent.success_count + agent.error_count
            }
        
        return summary
    
    def is_workflow_complete(self) -> bool:
        """Check if the workflow is complete."""
        if not self.tasks:
            return False
        
        return all(task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] 
                  for task in self.tasks.values())
    
    def has_critical_errors(self) -> bool:
        """Check if the workflow has critical errors."""
        # Consider a workflow to have critical errors if multiple tasks failed
        failed_tasks = sum(1 for task in self.tasks.values() 
                          if task.status == TaskStatus.FAILED)
        return failed_tasks > len(self.tasks) * 0.3  # More than 30% failed
    
    def can_proceed(self) -> bool:
        """Check if the workflow can proceed to the next phase."""
        # Basic logic - can be extended
        return (not self.has_critical_errors() and 
                len(self.get_ready_tasks()) > 0 or 
                len([a for a in self.agents.values() if a.status == "busy"]) > 0)
    
    def finalize(self, success: bool = True):
        """Finalize the workflow."""
        self.completed_at = datetime.now()
        self.status = "completed" if success else "failed"
        self.current_phase = WorkflowPhase.FINALIZATION
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow state to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "project_title": self.project_title,
            "project_description": self.project_description,
            "current_phase": self.current_phase.value,
            "status": self.status,
            "completion_percentage": self.get_completion_percentage(),
            "task_summary": self.get_task_summary(),
            "agent_summary": self.get_agent_summary(),
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "deliverables": self.deliverables,
            "quality_gates": self.quality_gates,
            "metrics": self.metrics,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings)
        }