"""
Main entry point for the Self-Reflecting Claude Code Agent system.

This module provides the primary interface for initializing and using
the complete agent system with all its components.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import yaml

from .agents import ManagerAgent, CoderAgent, ReviewerAgent, ResearcherAgent
from .workflows import DevelopmentWorkflow
from .dspy_integration import DSPyManager
from .rag import HybridRAG
from .memory import AgentMemory
from .context import ContextManager
from .evaluation import AgentEvaluator
from .domains import DomainManager


class SelfReflectingAgent:
    """
    Main Self-Reflecting Claude Code Agent system.
    
    This class orchestrates all components of the agent system including:
    - Multi-agent coordination (Manager, Coder, Reviewer, Researcher)
    - LangGraph workflows for structured development processes
    - DSPy integration for optimizable prompting
    - Hybrid RAG for information retrieval
    - Persistent memory with mem0
    - Context engineering for efficient token usage
    - Self-improvement through evaluation and optimization
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        project_path: Optional[str] = None,
        enable_memory: bool = True,
        enable_self_improvement: bool = True
    ):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.project_path = Path(project_path) if project_path else Path.cwd()
        
        # Feature flags
        self.enable_memory = enable_memory
        self.enable_self_improvement = enable_self_improvement
        
        # Core components
        self.dspy_manager: Optional[DSPyManager] = None
        self.rag_system: Optional[HybridRAG] = None
        self.memory_system: Optional[AgentMemory] = None
        self.context_manager: Optional[ContextManager] = None
        self.evaluator: Optional[AgentEvaluator] = None
        self.domain_manager: Optional[DomainManager] = None
        
        # Agents
        self.agents: Dict[str, Any] = {}
        
        # Workflow
        self.workflow: Optional[DevelopmentWorkflow] = None
        
        # System state
        self.initialized = False
        
        self.logger.info("Self-Reflecting Agent system created")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration."""
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        return yaml.safe_load(f)
                    else:
                        import json
                        return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # Return default configuration
        return {
            "agents": {
                "manager": {"model": "gpt-4o", "temperature": 0.1},
                "coder": {"model": "gpt-4o", "temperature": 0.2},
                "reviewer": {"model": "gpt-4o", "temperature": 0.1},
                "researcher": {"model": "gpt-4o", "temperature": 0.3}
            },
            "workflows": {
                "development": {
                    "max_iterations": 10,
                    "enable_parallel_execution": True,
                    "timeout_seconds": 300
                }
            },
            "dspy": {
                "enabled": True,
                "model": {"name": "gpt-4o", "params": {"max_tokens": 4000}}
            },
            "rag": {
                "enabled": True,
                "vector_store": {"provider": "faiss"},
                "bm25_weight": 0.3,
                "vector_weight": 0.7
            },
            "memory": {
                "enabled": True,
                "provider": "mem0"
            },
            "evaluation": {
                "enabled": True,
                "llm_as_judge": {"enabled": True}
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize all system components."""
        
        if self.initialized:
            return True
        
        try:
            self.logger.info("Initializing Self-Reflecting Agent system...")
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize domain manager
            await self._initialize_domain_manager()
            
            # Initialize agents
            await self._initialize_agents()
            
            # Initialize workflow
            await self._initialize_workflow()
            
            # Setup integrations
            await self._setup_integrations()
            
            self.initialized = True
            self.logger.info("Self-Reflecting Agent system initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False
    
    async def _initialize_core_components(self) -> None:
        """Initialize core system components."""
        
        # Initialize DSPy manager
        if self.config.get("dspy", {}).get("enabled", True):
            self.dspy_manager = DSPyManager(self.config.get("dspy", {}))
        
        # Initialize RAG system
        if self.config.get("rag", {}).get("enabled", True):
            self.rag_system = HybridRAG(self.config.get("rag", {}))
            await self.rag_system.initialize()
        
        # Initialize memory system
        if self.enable_memory and self.config.get("memory", {}).get("enabled", True):
            try:
                from .memory import AgentMemory
                self.memory_system = AgentMemory(self.config.get("memory", {}))
                await self.memory_system.initialize()
            except ImportError:
                self.logger.warning("Memory system not available, continuing without persistent memory")
        
        # Initialize context manager
        try:
            from .context import ContextManager
            self.context_manager = ContextManager(self.config.get("context", {}))
        except ImportError:
            self.logger.warning("Context manager not available, using basic context handling")
        
        # Initialize evaluator
        if self.enable_self_improvement and self.config.get("evaluation", {}).get("enabled", True):
            try:
                from .evaluation import AgentEvaluator
                self.evaluator = AgentEvaluator(self.config.get("evaluation", {}))
                await self.evaluator.initialize()
            except ImportError:
                self.logger.warning("Evaluator not available, continuing without self-improvement")
    
    async def _initialize_domain_manager(self) -> None:
        """Initialize domain manager for specialized agents."""
        
        try:
            self.domain_manager = DomainManager(self.config)
            await self.domain_manager.initialize()
            self.logger.info("Domain manager initialized")
        except Exception as e:
            self.logger.warning(f"Domain manager initialization failed: {e}")
    
    async def _initialize_agents(self) -> None:
        """Initialize all agent instances."""
        
        # Get DSPy language model if available
        dspy_lm = None
        if self.dspy_manager and self.dspy_manager.is_available():
            dspy_lm = self.dspy_manager.language_model
        
        # Initialize agents with shared components
        self.agents = {
            "manager": ManagerAgent(
                agent_id="manager",
                config=self.config["agents"]["manager"],
                memory=self.memory_system,
                context_manager=self.context_manager,
                evaluator=self.evaluator,
                dspy_lm=dspy_lm
            ),
            "coder": CoderAgent(
                agent_id="coder",
                config=self.config["agents"]["coder"],
                memory=self.memory_system,
                context_manager=self.context_manager,
                evaluator=self.evaluator,
                dspy_lm=dspy_lm
            ),
            "reviewer": ReviewerAgent(
                agent_id="reviewer",
                config=self.config["agents"]["reviewer"],
                memory=self.memory_system,
                context_manager=self.context_manager,
                evaluator=self.evaluator,
                dspy_lm=dspy_lm
            ),
            "researcher": ResearcherAgent(
                agent_id="researcher",
                config=self.config["agents"]["researcher"],
                memory=self.memory_system,
                context_manager=self.context_manager,
                evaluator=self.evaluator,
                dspy_lm=dspy_lm
            )
        }
        
        self.logger.info("All agents initialized")
    
    async def _initialize_workflow(self) -> None:
        """Initialize the development workflow."""
        
        self.workflow = DevelopmentWorkflow(self.config)
        
        # Add progress callbacks if needed
        if self.logger.isEnabledFor(logging.INFO):
            self.workflow.add_progress_callback(self._log_workflow_progress)
    
    async def _setup_integrations(self) -> None:
        """Setup integrations between components."""
        
        # Add project codebase to RAG system if available
        if self.rag_system and self.project_path.exists():
            try:
                await self.rag_system.add_documents_from_directory(
                    self.project_path,
                    file_patterns=["*.py", "*.md", "*.txt", "*.json", "*.yaml"],
                    recursive=True
                )
                self.logger.info(f"Added project codebase to RAG system: {self.project_path}")
            except Exception as e:
                self.logger.warning(f"Failed to add codebase to RAG: {e}")
    
    async def execute_task(
        self,
        task_description: str,
        requirements: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a development task using the agent system.
        
        Args:
            task_description: Description of the task to execute
            requirements: Optional structured requirements
            constraints: Optional constraints and limitations
            
        Returns:
            Dictionary containing execution results
        """
        
        if not self.initialized:
            await self.initialize()
        
        if not self.workflow:
            return {
                "status": "failed",
                "error": "Workflow not initialized",
                "task_description": task_description
            }
        
        try:
            self.logger.info(f"Executing task: {task_description}")
            
            # Execute the task through the workflow
            result = await self.workflow.execute_workflow(
                project_title=f"Task: {task_description[:50]}...",
                project_description=task_description,
                requirements=requirements,
                constraints=constraints
            )
            
            # Trigger self-improvement if enabled
            if self.enable_self_improvement and result.get("status") == "completed":
                await self._trigger_self_improvement(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "task_description": task_description
            }
    
    async def _trigger_self_improvement(self, execution_result: Dict[str, Any]) -> None:
        """Trigger self-improvement based on execution results."""
        
        try:
            # Collect metrics from agents
            agent_metrics = {}
            for agent_id, agent in self.agents.items():
                agent_metrics[agent_id] = agent.get_metrics()
            
            # Trigger DSPy optimization if available
            if self.dspy_manager and self.dspy_manager.is_available():
                await self.dspy_manager.auto_optimize_signatures(
                    min_executions=5,
                    performance_threshold=0.8
                )
            
            # Agent self-improvement
            for agent in self.agents.values():
                await agent.self_improve()
            
            self.logger.info("Self-improvement cycle completed")
            
        except Exception as e:
            self.logger.warning(f"Self-improvement failed: {e}")
    
    async def execute_domain_workflow(
        self,
        domain_name: str,
        workflow_name: str,
        task_description: str,
        task_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a domain-specific workflow.
        
        Args:
            domain_name: Name of the domain (e.g., 'software_development')
            workflow_name: Name of the workflow (e.g., 'architecture_review')  
            task_description: Description of the task
            task_context: Additional context for the task
            
        Returns:
            Workflow execution results
        """
        
        if not self.domain_manager:
            return {"error": "Domain manager not initialized"}
        
        try:
            self.logger.info(f"Executing domain workflow: {domain_name}.{workflow_name}")
            
            # Prepare task context
            context = task_context or {}
            context["task"] = task_description
            context["domain"] = domain_name
            context["workflow"] = workflow_name
            
            # Execute workflow
            result = await self.domain_manager.execute_domain_workflow(
                domain_name, workflow_name, context
            )
            
            self.logger.info(f"Domain workflow completed: {domain_name}.{workflow_name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Domain workflow execution failed: {e}")
            return {"error": str(e)}
    
    def get_domain_agent(self, domain_name: str, agent_name: str):
        """Get a specific domain agent."""
        
        if not self.domain_manager:
            return None
        
        return self.domain_manager.get_domain_agent(domain_name, agent_name)
    
    def list_available_domains(self) -> List[str]:
        """List all available domains."""
        
        if not self.domain_manager:
            return []
        
        return self.domain_manager.list_available_domains()
    
    def list_domain_agents(self, domain_name: str) -> List[str]:
        """List all agents in a specific domain."""
        
        if not self.domain_manager:
            return []
        
        return self.domain_manager.list_domain_agents(domain_name)
    
    def get_domain_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded domains."""
        
        if not self.domain_manager:
            return {"error": "Domain manager not available"}
        
        return self.domain_manager.get_domain_statistics()
    
    async def _log_workflow_progress(self, step_name: str, state: Any) -> None:
        """Log workflow progress."""
        completion = getattr(state, 'get_completion_percentage', lambda: 0)()
        self.logger.info(f"Workflow step: {step_name} (Completion: {completion:.1f}%)")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        status = {
            "initialized": self.initialized,
            "project_path": str(self.project_path),
            "enable_memory": self.enable_memory,
            "enable_self_improvement": self.enable_self_improvement,
            "components": {
                "dspy_manager": self.dspy_manager.get_status() if self.dspy_manager else None,
                "rag_system": self.rag_system.get_search_metrics() if self.rag_system else None,
                "memory_system": "initialized" if self.memory_system else "not_available",
                "context_manager": "initialized" if self.context_manager else "not_available",
                "evaluator": "initialized" if self.evaluator else "not_available"
            },
            "agents": {
                agent_id: agent.get_metrics()
                for agent_id, agent in self.agents.items()
            }
        }
        
        if self.workflow:
            status["workflow"] = self.workflow.get_workflow_metrics()
        
        return status
    
    async def add_knowledge(
        self,
        content: str,
        source: str = "user_input",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add knowledge to the system."""
        
        success = True
        
        # Add to RAG system
        if self.rag_system:
            document = {
                "content": content,
                "metadata": metadata or {}
            }
            success &= await self.rag_system.add_documents([document], source)
        
        # Add to memory system
        if self.memory_system:
            success &= await self.memory_system.add_memory(
                content=content,
                metadata={
                    "source": source,
                    **(metadata or {})
                }
            )
        
        return success
    
    async def search_knowledge(
        self,
        query: str,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for relevant knowledge."""
        
        if self.rag_system:
            return await self.rag_system.search(
                query=query,
                search_type="hybrid",
                max_results=max_results
            )
        
        return []
    
    async def export_system_state(self, export_path: str) -> bool:
        """Export complete system state."""
        
        try:
            export_data = {
                "export_timestamp": asyncio.get_event_loop().time(),
                "system_status": self.get_system_status(),
                "config": self.config
            }
            
            # Add component-specific exports
            if self.dspy_manager:
                dspy_export_path = f"{export_path}_dspy.json"
                self.dspy_manager.export_configuration(Path(dspy_export_path))
                export_data["dspy_export"] = dspy_export_path
            
            # Export main state
            import json
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"System state exported to: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export system state: {e}")
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the agent system gracefully."""
        
        try:
            self.logger.info("Shutting down Self-Reflecting Agent system...")
            
            # Shutdown components
            if self.rag_system:
                await self.rag_system.shutdown()
            
            if self.memory_system:
                await self.memory_system.shutdown()
            
            # Final self-improvement if enabled
            if self.enable_self_improvement and self.initialized:
                await self._trigger_self_improvement({})
            
            self.initialized = False
            self.logger.info("System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Convenience function for simple usage
async def create_agent(
    project_path: Optional[str] = None,
    config_path: Optional[str] = None,
    **kwargs
) -> SelfReflectingAgent:
    """
    Create and initialize a Self-Reflecting Agent.
    
    Args:
        project_path: Path to the project directory
        config_path: Path to configuration file
        **kwargs: Additional configuration options
        
    Returns:
        Initialized SelfReflectingAgent instance
    """
    
    agent = SelfReflectingAgent(
        project_path=project_path,
        config_path=config_path,
        **kwargs
    )
    
    await agent.initialize()
    return agent