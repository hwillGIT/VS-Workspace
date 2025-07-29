"""
Workflow nodes for LangGraph execution.

This module defines the individual nodes that make up the development workflow,
each representing a specific step or decision point in the development process.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from ..agents import ManagerAgent, CoderAgent, ReviewerAgent, ResearcherAgent
from .workflow_state import WorkflowState, WorkflowPhase, TaskStatus, AgentType


class WorkflowNodes:
    """
    Collection of workflow nodes for the development process.
    
    Each node represents a specific step in the development workflow and can
    interact with agents to perform tasks, make decisions, or update state.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize agents
        self.agents = {
            "manager": ManagerAgent("manager", config=config.get("agents", {}).get("manager", {})),
            "coder": CoderAgent("coder", config=config.get("agents", {}).get("coder", {})),
            "reviewer": ReviewerAgent("reviewer", config=config.get("agents", {}).get("reviewer", {})),
            "researcher": ResearcherAgent("researcher", config=config.get("agents", {}).get("researcher", {}))
        }
        
        self.logger.info("Workflow nodes initialized with all agents")
    
    async def initialize_workflow(self, state: WorkflowState) -> WorkflowState:
        """
        Initialize the workflow with project requirements.
        
        This node sets up the initial workflow state, registers agents,
        and prepares for the development process.
        """
        self.logger.info(f"Initializing workflow: {state.project_title}")
        
        try:
            # Update workflow phase
            state.update_phase(WorkflowPhase.INITIALIZATION)
            state.status = "initializing"
            
            # Register all agents
            for agent_id, agent in self.agents.items():
                agent_type = AgentType(agent_id)
                capabilities = self._get_agent_capabilities(agent_id)
                state.add_agent(agent_id, agent_type, capabilities)
            
            # Add initial context
            state.update_context("project_requirements", {
                "title": state.project_title,
                "description": state.project_description
            })
            
            # Set initial quality gates
            state.set_quality_gate("initialization", True)
            
            self.logger.info("Workflow initialization completed successfully")
            return state
            
        except Exception as e:
            error_msg = f"Failed to initialize workflow: {str(e)}"
            self.logger.error(error_msg)
            state.add_error(error_msg)
            state.status = "failed"
            return state
    
    async def research_phase(self, state: WorkflowState) -> WorkflowState:
        """
        Research phase node - gather information and analyze requirements.
        
        This node uses the researcher agent to gather information about
        the project requirements, technologies, and existing solutions.
        """
        self.logger.info("Starting research phase")
        
        try:
            state.update_phase(WorkflowPhase.RESEARCH)
            state.status = "researching"
            
            # Create research tasks
            research_tasks = [
                {
                    "task_id": "requirements_analysis",
                    "title": "Requirements Analysis",
                    "description": "Analyze and clarify project requirements",
                    "type": "requirements_gathering"
                },
                {
                    "task_id": "solution_research", 
                    "title": "Solution Research",
                    "description": "Research potential solutions and approaches",
                    "type": "solution_research"
                },
                {
                    "task_id": "technology_analysis",
                    "title": "Technology Analysis", 
                    "description": "Analyze technologies and frameworks",
                    "type": "technology_analysis"
                }
            ]
            
            # Add tasks to workflow
            for task_info in research_tasks:
                state.add_task(
                    task_id=task_info["task_id"],
                    title=task_info["title"],
                    description=task_info["description"],
                    metadata={"type": task_info["type"]}
                )
            
            # Execute research tasks
            research_results = await self._execute_research_tasks(state, research_tasks)
            
            # Store research results in context
            state.update_context("research_results", research_results)
            
            # Set quality gate
            research_quality = self._assess_research_quality(research_results)
            state.set_quality_gate("research_complete", research_quality)
            
            self.logger.info("Research phase completed")
            return state
            
        except Exception as e:
            error_msg = f"Research phase failed: {str(e)}"
            self.logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    async def planning_phase(self, state: WorkflowState) -> WorkflowState:
        """
        Planning phase node - create detailed implementation plan.
        
        This node uses the manager agent to create a comprehensive
        implementation plan based on research findings.
        """
        self.logger.info("Starting planning phase")
        
        try:
            state.update_phase(WorkflowPhase.PLANNING)
            state.status = "planning"
            
            # Get research results
            research_results = state.get_context("research_results", {})
            
            # Create planning task
            planning_task = {
                "title": "Create Implementation Plan",
                "description": state.project_description,
                "requirements": research_results.get("requirements_analysis", {}),
                "context": {
                    "research_findings": research_results,
                    "project_constraints": state.get_context("constraints", {})
                }
            }
            
            # Execute planning with manager agent
            manager = self.agents["manager"]
            planning_result = await manager.process_task(planning_task)
            
            if planning_result.get("status") == "completed":
                # Extract implementation plan
                project_plan = planning_result.get("project_plan", {})
                subtasks = project_plan.get("subtasks", [])
                
                # Add implementation tasks to workflow
                for subtask in subtasks:
                    state.add_task(
                        task_id=subtask.get("id", ""),
                        title=subtask.get("title", ""),
                        description=subtask.get("description", ""),
                        dependencies=subtask.get("dependencies", []),
                        metadata={
                            "agent_type": subtask.get("agent_type", "coder"),
                            "priority": subtask.get("priority", "medium"),
                            "estimated_effort": subtask.get("estimated_effort", 1)
                        }
                    )
                
                # Store plan in context
                state.update_context("implementation_plan", project_plan)
                state.set_quality_gate("planning_complete", True)
                
                self.logger.info(f"Planning completed with {len(subtasks)} tasks")
            else:
                error_msg = f"Planning failed: {planning_result.get('error', 'Unknown error')}"
                state.add_error(error_msg)
                state.set_quality_gate("planning_complete", False)
            
            return state
            
        except Exception as e:
            error_msg = f"Planning phase failed: {str(e)}"
            self.logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    async def implementation_phase(self, state: WorkflowState) -> WorkflowState:
        """
        Implementation phase node - execute development tasks.
        
        This node coordinates the execution of implementation tasks
        across multiple agents based on the created plan.
        """
        self.logger.info("Starting implementation phase")
        
        try:
            state.update_phase(WorkflowPhase.IMPLEMENTATION)
            state.status = "implementing"
            
            # Execute tasks in dependency order
            implementation_results = await self._execute_implementation_tasks(state)
            
            # Store implementation results
            state.update_context("implementation_results", implementation_results)
            
            # Assess implementation quality
            implementation_success = implementation_results.get("success_rate", 0.0) > 0.8
            state.set_quality_gate("implementation_complete", implementation_success)
            
            self.logger.info("Implementation phase completed")
            return state
            
        except Exception as e:
            error_msg = f"Implementation phase failed: {str(e)}"
            self.logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    async def review_phase(self, state: WorkflowState) -> WorkflowState:
        """
        Review phase node - conduct code review and quality assessment.
        
        This node uses the reviewer agent to assess code quality,
        security, and adherence to best practices.
        """
        self.logger.info("Starting review phase")
        
        try:
            state.update_phase(WorkflowPhase.REVIEW)
            state.status = "reviewing"
            
            # Get implementation results
            implementation_results = state.get_context("implementation_results", {})
            deliverables = state.deliverables
            
            # Create review tasks
            review_tasks = []
            
            if deliverables:
                for deliverable in deliverables[:5]:  # Limit reviews
                    review_tasks.append({
                        "task_id": f"review_{deliverable.replace('/', '_').replace('.', '_')}",
                        "title": f"Review {deliverable}",
                        "description": f"Comprehensive review of {deliverable}",
                        "file_path": deliverable,
                        "type": "code_review"
                    })
            
            # Execute review tasks
            review_results = await self._execute_review_tasks(state, review_tasks)
            
            # Store review results
            state.update_context("review_results", review_results)
            
            # Assess overall quality
            overall_quality_score = review_results.get("average_quality_score", 0.0)
            quality_gate_passed = overall_quality_score >= 7.0
            state.set_quality_gate("review_complete", quality_gate_passed)
            state.update_metric("quality_score", overall_quality_score)
            
            if not quality_gate_passed:
                state.add_warning("Code quality below threshold, consider improvements")
            
            self.logger.info(f"Review phase completed with quality score: {overall_quality_score}")
            return state
            
        except Exception as e:
            error_msg = f"Review phase failed: {str(e)}"
            self.logger.error(error_msg)
            state.add_error(error_msg)
            return state
    
    async def finalization_phase(self, state: WorkflowState) -> WorkflowState:
        """
        Finalization phase node - complete the workflow and generate summary.
        
        This node finalizes the workflow, generates documentation,
        and prepares final deliverables.
        """
        self.logger.info("Starting finalization phase")
        
        try:
            state.update_phase(WorkflowPhase.FINALIZATION)
            state.status = "finalizing"
            
            # Generate final summary
            summary = await self._generate_workflow_summary(state)
            state.update_context("final_summary", summary)
            
            # Check if workflow was successful
            success = self._assess_workflow_success(state)
            
            # Finalize workflow
            state.finalize(success)
            
            # Update final metrics
            state.update_metric("completion_percentage", state.get_completion_percentage())
            state.update_metric("total_tasks", len(state.tasks))
            state.update_metric("total_errors", len(state.errors))
            
            self.logger.info(f"Workflow finalization completed - Success: {success}")
            return state
            
        except Exception as e:
            error_msg = f"Finalization phase failed: {str(e)}"
            self.logger.error(error_msg)
            state.add_error(error_msg)
            state.finalize(False)
            return state
    
    async def _execute_research_tasks(
        self, 
        state: WorkflowState, 
        research_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute research tasks using the researcher agent."""
        
        researcher = self.agents["researcher"]
        results = {}
        
        for task_info in research_tasks:
            try:
                task_id = task_info["task_id"]
                
                # Assign task to researcher
                if state.assign_task_to_agent(task_id, "researcher"):
                    
                    # Execute task
                    task_result = await researcher.process_task({
                        "type": task_info["type"],
                        "title": task_info["title"],
                        "description": task_info["description"],
                        "context": state.context
                    })
                    
                    # Complete task
                    success = task_result.get("status") == "completed"
                    state.complete_task(task_id, success, task_result)
                    
                    if success:
                        results[task_id] = task_result
                    else:
                        state.add_error(f"Research task {task_id} failed: {task_result.get('error', 'Unknown error')}")
                
            except Exception as e:
                state.add_error(f"Error executing research task {task_id}: {str(e)}")
        
        return results
    
    async def _execute_implementation_tasks(self, state: WorkflowState) -> Dict[str, Any]:
        """Execute implementation tasks across agents."""
        
        results = {
            "completed_tasks": [],
            "failed_tasks": [],
            "success_rate": 0.0
        }
        
        # Get tasks that can be executed
        max_iterations = 20
        iteration = 0
        
        while iteration < max_iterations and not state.is_workflow_complete():
            iteration += 1
            
            # Get ready tasks
            ready_tasks = state.get_ready_tasks()
            available_agents = state.get_available_agents()
            
            if not ready_tasks or not available_agents:
                # Wait for running tasks or break if nothing is happening
                await asyncio.sleep(1)
                continue
            
            # Assign tasks to agents
            assignments = []
            for task_id in ready_tasks[:len(available_agents)]:  # Limit to available agents
                task = state.tasks[task_id]
                preferred_agent = task.metadata.get("agent_type", "coder")
                
                # Find suitable agent
                suitable_agent = None
                if preferred_agent in available_agents:
                    suitable_agent = preferred_agent
                elif available_agents:
                    suitable_agent = available_agents[0]  # Fallback
                
                if suitable_agent and state.assign_task_to_agent(task_id, suitable_agent):
                    assignments.append((task_id, suitable_agent))
                    available_agents.remove(suitable_agent)
            
            # Execute assigned tasks
            execution_coroutines = []
            for task_id, agent_id in assignments:
                execution_coroutines.append(self._execute_single_task(state, task_id, agent_id))
            
            if execution_coroutines:
                await asyncio.gather(*execution_coroutines, return_exceptions=True)
        
        # Calculate results
        completed_tasks = [tid for tid, task in state.tasks.items() if task.status == TaskStatus.COMPLETED]
        failed_tasks = [tid for tid, task in state.tasks.items() if task.status == TaskStatus.FAILED]
        
        results["completed_tasks"] = completed_tasks
        results["failed_tasks"] = failed_tasks
        
        total_tasks = len(state.tasks)
        if total_tasks > 0:
            results["success_rate"] = len(completed_tasks) / total_tasks
        
        return results
    
    async def _execute_single_task(self, state: WorkflowState, task_id: str, agent_id: str):
        """Execute a single task with the assigned agent."""
        
        try:
            task = state.tasks[task_id]
            agent = self.agents[agent_id]
            
            # Prepare task context
            task_context = {
                "id": task_id,
                "title": task.title,
                "description": task.description,
                "metadata": task.metadata,
                "workflow_context": state.context
            }
            
            # Execute task
            result = await agent.process_task(task_context)
            
            # Handle result
            success = result.get("status") == "completed"
            
            if success:
                # Add deliverables if any
                if "deliverables" in result:
                    for deliverable in result["deliverables"]:
                        state.add_deliverable(deliverable)
                
                # Add outputs
                if "outputs" in result:
                    task.outputs.update(result["outputs"])
            
            # Complete task
            state.complete_task(task_id, success, result)
            
        except Exception as e:
            error_msg = f"Task execution failed for {task_id}: {str(e)}"
            self.logger.error(error_msg)
            state.add_error(error_msg)
            state.complete_task(task_id, False)
    
    async def _execute_review_tasks(
        self, 
        state: WorkflowState, 
        review_tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute review tasks using the reviewer agent."""
        
        reviewer = self.agents["reviewer"]
        results = {
            "reviews": [],
            "average_quality_score": 0.0,
            "total_findings": 0
        }
        
        quality_scores = []
        
        for task_info in review_tasks:
            try:
                # Execute review
                review_result = await reviewer.process_task(task_info)
                
                if review_result.get("status") == "completed":
                    results["reviews"].append(review_result)
                    
                    # Track quality metrics
                    quality_score = review_result.get("overall_score", 0.0)
                    quality_scores.append(quality_score)
                    
                    findings_count = review_result.get("total_findings", 0)
                    results["total_findings"] += findings_count
                
            except Exception as e:
                state.add_error(f"Review task failed: {str(e)}")
        
        # Calculate average quality score
        if quality_scores:
            results["average_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        return results
    
    async def _generate_workflow_summary(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate a comprehensive workflow summary."""
        
        summary = {
            "workflow_id": state.workflow_id,
            "project_title": state.project_title,
            "completion_status": state.status,
            "duration": (state.updated_at - state.started_at).total_seconds(),
            "task_summary": state.get_task_summary(),
            "agent_summary": state.get_agent_summary(),
            "quality_gates": state.quality_gates,
            "metrics": state.metrics,
            "deliverables_count": len(state.deliverables),
            "error_count": len(state.errors),
            "warning_count": len(state.warnings),
            "success_indicators": {
                "tasks_completed": state.get_completion_percentage(),
                "quality_gates_passed": sum(1 for passed in state.quality_gates.values() if passed),
                "critical_errors": state.has_critical_errors()
            }
        }
        
        return summary
    
    def _assess_workflow_success(self, state: WorkflowState) -> bool:
        """Assess if the workflow was successful."""
        
        # Check completion percentage
        completion = state.get_completion_percentage()
        if completion < 80.0:
            return False
        
        # Check critical quality gates
        critical_gates = ["initialization", "planning_complete", "implementation_complete"]
        for gate in critical_gates:
            if not state.quality_gates.get(gate, False):
                return False
        
        # Check for critical errors
        if state.has_critical_errors():
            return False
        
        return True
    
    def _assess_research_quality(self, research_results: Dict[str, Any]) -> bool:
        """Assess the quality of research results."""
        
        required_research = ["requirements_analysis", "solution_research"]
        
        for required in required_research:
            if required not in research_results:
                return False
            
            result = research_results[required]
            if result.get("status") != "completed":
                return False
        
        return True
    
    def _get_agent_capabilities(self, agent_id: str) -> List[str]:
        """Get capabilities for an agent type."""
        
        capabilities_map = {
            "manager": ["task_coordination", "project_planning", "decision_making"],
            "coder": ["code_implementation", "testing", "debugging", "refactoring"],
            "reviewer": ["code_review", "security_analysis", "quality_assessment"],
            "researcher": ["information_gathering", "technology_analysis", "requirements_analysis"]
        }
        
        return capabilities_map.get(agent_id, [])