"""
Main development workflow using LangGraph for orchestration.

This module implements the primary development workflow that coordinates
multiple agents through a structured, stateful process using LangGraph.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

try:
    from langgraph import Graph, StateGraph
    from langgraph.graph import Node, Edge
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    # Fallback simple implementation
    class Graph:
        def __init__(self):
            self.nodes = {}
            self.edges = []
        
        def add_node(self, name: str, func: Callable):
            self.nodes[name] = func
        
        def add_edge(self, from_node: str, to_node: str):
            self.edges.append((from_node, to_node))
        
        def set_entry_point(self, node: str):
            self.entry_point = node
        
        def set_finish_point(self, node: str):
            self.finish_point = node

from .workflow_state import WorkflowState, WorkflowPhase
from .workflow_nodes import WorkflowNodes


class DevelopmentWorkflow:
    """
    Main development workflow that orchestrates the entire development process.
    
    This workflow uses LangGraph to coordinate multiple agents through a
    structured development process including research, planning, implementation,
    review, and finalization phases.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize workflow nodes
        self.nodes = WorkflowNodes(config)
        
        # Initialize workflow graph
        self.graph = self._build_workflow_graph()
        
        # Workflow state
        self.current_state: Optional[WorkflowState] = None
        
        # Execution settings
        self.max_retries = config.get("max_retries", 3)
        self.timeout_seconds = config.get("timeout_seconds", 3600)  # 1 hour
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
        self.phase_callbacks: List[Callable] = []
        
        self.logger.info("Development workflow initialized")
    
    def _build_workflow_graph(self) -> Graph:
        """Build the LangGraph workflow graph."""
        
        if LANGGRAPH_AVAILABLE:
            # Use actual LangGraph implementation
            graph = StateGraph(WorkflowState)
            
            # Add nodes
            graph.add_node("initialize", self.nodes.initialize_workflow)
            graph.add_node("research", self.nodes.research_phase)
            graph.add_node("planning", self.nodes.planning_phase)
            graph.add_node("implementation", self.nodes.implementation_phase)
            graph.add_node("review", self.nodes.review_phase)
            graph.add_node("finalization", self.nodes.finalization_phase)
            
            # Add edges (workflow flow)
            graph.add_edge("initialize", "research")
            graph.add_edge("research", "planning")
            graph.add_edge("planning", "implementation")
            graph.add_edge("implementation", "review")
            graph.add_edge("review", "finalization")
            
            # Set entry and exit points
            graph.set_entry_point("initialize")
            graph.set_finish_point("finalization")
            
            return graph.compile()
        
        else:
            # Fallback simple graph implementation
            graph = Graph()
            
            # Add nodes
            graph.add_node("initialize", self.nodes.initialize_workflow)
            graph.add_node("research", self.nodes.research_phase)
            graph.add_node("planning", self.nodes.planning_phase)
            graph.add_node("implementation", self.nodes.implementation_phase)
            graph.add_node("review", self.nodes.review_phase)
            graph.add_node("finalization", self.nodes.finalization_phase)
            
            # Add edges
            graph.add_edge("initialize", "research")
            graph.add_edge("research", "planning")
            graph.add_edge("planning", "implementation")
            graph.add_edge("implementation", "review")
            graph.add_edge("review", "finalization")
            
            # Set entry and exit points
            graph.set_entry_point("initialize")
            graph.set_finish_point("finalization")
            
            return graph
    
    async def execute_workflow(
        self, 
        project_title: str,
        project_description: str,
        requirements: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete development workflow.
        
        Args:
            project_title: Title of the project to develop
            project_description: Detailed description of project requirements
            requirements: Additional structured requirements
            constraints: Project constraints and limitations
            
        Returns:
            Dictionary containing workflow results and final state
        """
        
        start_time = datetime.now()
        workflow_id = f"workflow_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting development workflow: {project_title}")
        
        try:
            # Initialize workflow state
            self.current_state = WorkflowState(
                workflow_id=workflow_id,
                project_title=project_title,
                project_description=project_description
            )
            
            # Add requirements and constraints to context
            if requirements:
                self.current_state.update_context("requirements", requirements)
            if constraints:
                self.current_state.update_context("constraints", constraints)
            
            # Execute workflow
            if LANGGRAPH_AVAILABLE:
                # Use LangGraph execution
                final_state = await self._execute_langgraph_workflow()
            else:
                # Use fallback execution
                final_state = await self._execute_fallback_workflow()
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            final_state.update_metric("execution_time", execution_time)
            
            # Generate final result
            result = {
                "status": "completed" if final_state.status == "completed" else "failed",
                "workflow_id": workflow_id,
                "execution_time": execution_time,
                "project_summary": {
                    "title": final_state.project_title,
                    "description": final_state.project_description,
                    "completion_percentage": final_state.get_completion_percentage(),
                    "deliverables": final_state.deliverables,
                    "quality_gates": final_state.quality_gates
                },
                "task_summary": final_state.get_task_summary(),
                "agent_summary": final_state.get_agent_summary(),
                "metrics": final_state.metrics,
                "errors": final_state.errors,
                "warnings": final_state.warnings,
                "final_summary": final_state.get_context("final_summary", {})
            }
            
            self.logger.info(f"Workflow completed - Status: {result['status']}, " +
                           f"Completion: {result['project_summary']['completion_percentage']:.1f}%")
            
            return result
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            self.logger.error(error_msg)
            
            if self.current_state:
                self.current_state.add_error(error_msg)
                self.current_state.finalize(False)
            
            return {
                "status": "failed",
                "error": error_msg,
                "workflow_id": workflow_id,
                "execution_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def _execute_langgraph_workflow(self) -> WorkflowState:
        """Execute workflow using LangGraph."""
        
        try:
            # Execute the graph with the current state
            result = await self.graph.ainvoke(self.current_state)
            return result
            
        except Exception as e:
            self.logger.error(f"LangGraph execution failed: {e}")
            if self.current_state:
                self.current_state.add_error(f"LangGraph execution failed: {str(e)}")
                self.current_state.finalize(False)
            return self.current_state
    
    async def _execute_fallback_workflow(self) -> WorkflowState:
        """Execute workflow using fallback implementation."""
        
        # Define workflow steps in order
        workflow_steps = [
            ("initialize", self.nodes.initialize_workflow),
            ("research", self.nodes.research_phase),
            ("planning", self.nodes.planning_phase), 
            ("implementation", self.nodes.implementation_phase),
            ("review", self.nodes.review_phase),
            ("finalization", self.nodes.finalization_phase)
        ]
        
        state = self.current_state
        
        for step_name, step_function in workflow_steps:
            try:
                self.logger.info(f"Executing workflow step: {step_name}")
                
                # Call progress callbacks
                await self._call_progress_callbacks(step_name, state)
                
                # Execute step
                state = await step_function(state)
                
                # Check for critical errors
                if state.has_critical_errors():
                    self.logger.warning(f"Critical errors detected after {step_name}, continuing...")
                
                # Call phase callbacks
                await self._call_phase_callbacks(step_name, state)
                
            except Exception as e:
                error_msg = f"Workflow step {step_name} failed: {str(e)}"
                self.logger.error(error_msg)
                state.add_error(error_msg)
                
                # Continue execution unless it's a critical step
                if step_name in ["initialize", "planning"]:
                    state.finalize(False)
                    break
        
        return state
    
    async def _call_progress_callbacks(self, step_name: str, state: WorkflowState):
        """Call registered progress callbacks."""
        
        for callback in self.progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(step_name, state)
                else:
                    callback(step_name, state)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    async def _call_phase_callbacks(self, phase_name: str, state: WorkflowState):
        """Call registered phase callbacks."""
        
        for callback in self.phase_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(phase_name, state)
                else:
                    callback(phase_name, state)  
            except Exception as e:
                self.logger.warning(f"Phase callback failed: {e}")
    
    def add_progress_callback(self, callback: Callable):
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)
    
    def add_phase_callback(self, callback: Callable):
        """Add a phase callback function."""
        self.phase_callbacks.append(callback)
    
    def get_current_state(self) -> Optional[WorkflowState]:
        """Get the current workflow state."""
        return self.current_state
    
    async def pause_workflow(self) -> bool:
        """Pause the workflow execution (if supported)."""
        # This would implement workflow pausing logic
        # For now, just log the request
        self.logger.info("Workflow pause requested")
        return True
    
    async def resume_workflow(self) -> bool:
        """Resume paused workflow execution."""
        # This would implement workflow resumption logic
        self.logger.info("Workflow resume requested")
        return True
    
    async def cancel_workflow(self) -> bool:
        """Cancel the workflow execution."""
        if self.current_state:
            self.current_state.add_error("Workflow cancelled by user")
            self.current_state.finalize(False)
            self.logger.info("Workflow cancelled")
            return True
        return False
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """Get comprehensive workflow metrics."""
        
        if not self.current_state:
            return {}
        
        return {
            "workflow_id": self.current_state.workflow_id,
            "current_phase": self.current_state.current_phase.value,
            "status": self.current_state.status,
            "completion_percentage": self.current_state.get_completion_percentage(),
            "task_metrics": {
                "total_tasks": len(self.current_state.tasks),
                "completed_tasks": len([t for t in self.current_state.tasks.values() 
                                      if t.status.value == "completed"]),
                "failed_tasks": len([t for t in self.current_state.tasks.values() 
                                   if t.status.value == "failed"]),
                "in_progress_tasks": len([t for t in self.current_state.tasks.values() 
                                        if t.status.value in ["implementing", "reviewing"]])
            },
            "agent_metrics": {
                "total_agents": len(self.current_state.agents),
                "active_agents": len([a for a in self.current_state.agents.values() 
                                    if a.status == "busy"]),
                "agent_success_rates": {
                    agent_id: agent.get_success_rate() 
                    for agent_id, agent in self.current_state.agents.items()
                }
            },
            "quality_metrics": {
                "quality_gates_passed": sum(1 for passed in self.current_state.quality_gates.values() if passed),
                "quality_gates_total": len(self.current_state.quality_gates),
                "overall_quality_score": self.current_state.metrics.get("quality_score", 0.0)
            },
            "execution_metrics": {
                "started_at": self.current_state.started_at.isoformat(),
                "updated_at": self.current_state.updated_at.isoformat(),
                "execution_time": self.current_state.metrics.get("execution_time", 0.0),
                "error_count": len(self.current_state.errors),
                "warning_count": len(self.current_state.warnings)
            }
        }
    
    async def create_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Create a checkpoint of the current workflow state."""
        
        if not self.current_state:
            return None
        
        checkpoint = {
            "workflow_id": self.current_state.workflow_id,
            "checkpoint_time": datetime.now().isoformat(),
            "state": self.current_state.to_dict(),
            "config": self.config
        }
        
        self.logger.info(f"Created workflow checkpoint for {self.current_state.workflow_id}")
        return checkpoint
    
    async def restore_from_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """Restore workflow from a checkpoint."""
        
        try:
            # This would implement checkpoint restoration logic
            # For now, just log the attempt
            workflow_id = checkpoint.get("workflow_id", "unknown")
            self.logger.info(f"Checkpoint restoration requested for {workflow_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore from checkpoint: {e}")
            return False
    
    def validate_workflow_config(self) -> List[str]:
        """Validate the workflow configuration and return any issues."""
        
        issues = []
        
        # Check required configuration
        required_configs = ["agents", "workflows"]
        for config_key in required_configs:
            if config_key not in self.config:
                issues.append(f"Missing required configuration: {config_key}")
        
        # Check agent configurations
        if "agents" in self.config:
            required_agents = ["manager", "coder", "reviewer", "researcher"]
            for agent_type in required_agents:
                if agent_type not in self.config["agents"]:
                    issues.append(f"Missing agent configuration: {agent_type}")
        
        # Check workflow settings
        if "workflows" in self.config:
            workflow_config = self.config["workflows"]
            if "development" not in workflow_config:
                issues.append("Missing development workflow configuration")
        
        return issues