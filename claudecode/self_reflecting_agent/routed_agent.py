"""
Routed Self-Reflecting Agent

Main agent class that integrates intelligent model routing with the existing
self-reflecting agent system, providing seamless model switching with context preservation.
"""

import asyncio
import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .routing.agent_integration import RouterIntegratedAgent
from .routing.model_router import TaskType
from .routing.rag_semantic_integration import SemanticSearchEngine
from .main import SelfReflectingAgent


class RoutedSelfReflectingAgent(SelfReflectingAgent):
    """
    Enhanced self-reflecting agent with intelligent model routing capabilities.
    
    This agent extends the base SelfReflectingAgent with:
    - Intelligent model routing based on task type and context
    - Context preservation across model switches
    - RAG and semantic search integration
    - Performance tracking and optimization
    - Fallback handling for model failures
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        project_path: Optional[str] = None,
        enable_memory: bool = True,
        enable_self_improvement: bool = True,
        router_config_path: Optional[Path] = None,
        enable_rag: bool = True,
        enable_semantic_search: bool = True
    ):
        """
        Initialize the routed agent.
        
        Args:
            config_path: Path to agent configuration
            project_path: Project directory path
            enable_memory: Enable persistent memory
            enable_self_improvement: Enable self-improvement features
            router_config_path: Path to router configuration
            enable_rag: Enable RAG integration
            enable_semantic_search: Enable semantic search
        """
        
        # Initialize base agent
        super().__init__(
            config_path=config_path,
            project_path=project_path,
            enable_memory=enable_memory,
            enable_self_improvement=enable_self_improvement
        )
        
        # Initialize router-integrated agent
        self.router_agent = RouterIntegratedAgent(
            router_config_path=router_config_path,
            rag_config=self.config.get("rag", {}) if enable_rag else None,
            memory_config=self.config.get("memory", {}) if enable_memory else None,
            agent_id=f"routed_agent_{id(self)}"
        )
        
        # Initialize semantic search if enabled
        self.semantic_search = None
        if enable_semantic_search:
            semantic_config = self.config.get("semantic_search", {})
            self.semantic_search = SemanticSearchEngine(semantic_config)
        
        # Session management
        self.default_session_id = f"session_{id(self)}"
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Enhanced capabilities flags
        self.enable_rag = enable_rag
        self.enable_semantic_search = enable_semantic_search
        
        self.logger.info("RoutedSelfReflectingAgent initialized")
    
    async def initialize(self):
        """Initialize all agent systems."""
        
        # Initialize base agent
        await super().initialize()
        
        # Initialize router agent
        await self.router_agent.initialize()
        
        # Initialize semantic search
        if self.semantic_search:
            await self.semantic_search.initialize()
            
            # Index project files if project path is provided
            if self.project_path:
                await self._index_project_files()
        
        self.logger.info("All systems initialized successfully")
    
    async def _index_project_files(self):
        """Index project files for semantic search."""
        if not self.semantic_search or not self.project_path:
            return
        
        project_dir = Path(self.project_path)
        
        try:
            # Index code files
            await self.semantic_search.add_code_repository(project_dir)
            
            # Index documentation
            docs_dirs = ['docs', 'documentation', 'README.md', 'CLAUDE.md']
            for docs_name in docs_dirs:
                docs_path = project_dir / docs_name
                if docs_path.exists():
                    if docs_path.is_dir():
                        await self.semantic_search.add_documentation(docs_path)
                    elif docs_path.is_file():
                        # Single documentation file
                        content = docs_path.read_text(encoding='utf-8')
                        await self.semantic_search.add_documents([{
                            "content": content,
                            "metadata": {
                                "file_path": str(docs_path),
                                "content_type": "documentation"
                            },
                            "source": f"file://{docs_path}",
                            "id": str(docs_path)
                        }])
            
            self.logger.info(f"Indexed project files from {project_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to index project files: {e}")
    
    # Enhanced task execution methods
    
    async def execute_task(
        self,
        task_description: str,
        requirements: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a task with intelligent routing.
        
        Args:
            task_description: Description of the task
            requirements: Task requirements
            constraints: Task constraints
            session_id: Session identifier (creates new if None)
            task_type: Specific task type for routing
            **kwargs: Additional parameters
            
        Returns:
            Task execution result with routing metadata
        """
        
        # Use default session if not provided
        if session_id is None:
            session_id = self.default_session_id
        
        # Auto-detect task type if not provided
        if task_type is None:
            task_type = self._detect_task_type(task_description)
        
        # Prepare system context
        system_context = self._prepare_system_context(requirements, constraints)
        
        # Determine complexity
        complexity = self._determine_complexity(task_description, requirements)
        
        # Execute using router agent
        result = await self.router_agent.process_request(
            session_id=session_id,
            user_message=task_description,
            task_type=task_type,
            complexity=complexity,
            system_context=system_context,
            use_rag=self.enable_rag,
            use_memory=self.enable_memory,
            **kwargs
        )
        
        # Track execution in session
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "task_count": 0,
                "models_used": set(),
                "total_tokens": 0
            }
        
        session = self.active_sessions[session_id]
        session["task_count"] += 1
        if result.get("model_used"):
            session["models_used"].add(result["model_used"])
        if result.get("tokens_used"):
            session["total_tokens"] += result["tokens_used"]
        
        return result
    
    async def execute_domain_workflow(
        self,
        domain_name: str,
        workflow_name: str,
        task_description: str,
        task_context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a domain-specific workflow with routing.
        
        Args:
            domain_name: Domain name (e.g., 'software_development')
            workflow_name: Workflow name (e.g., 'architecture_review')
            task_description: Task description
            task_context: Additional task context
            session_id: Session identifier
            **kwargs: Additional parameters
            
        Returns:
            Workflow execution result
        """
        
        # Use default session if not provided
        if session_id is None:
            session_id = self.default_session_id
        
        # Map domain workflow to task type
        task_type = self._map_workflow_to_task_type(domain_name, workflow_name)
        
        # Prepare enhanced task description
        enhanced_description = f"Execute {domain_name} workflow '{workflow_name}': {task_description}"
        
        if task_context:
            context_info = ", ".join([f"{k}: {v}" for k, v in task_context.items()])
            enhanced_description += f"\n\nContext: {context_info}"
        
        # Execute as regular task with appropriate routing
        return await self.execute_task(
            task_description=enhanced_description,
            session_id=session_id,
            task_type=task_type,
            domain_workflow={"domain": domain_name, "workflow": workflow_name},
            **kwargs
        )
    
    # Convenience methods with routing
    
    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Chat with intelligent routing."""
        return await self.router_agent.chat(
            session_id=session_id or self.default_session_id,
            message=message,
            system_prompt=system_prompt,
            **kwargs
        )
    
    async def generate_code(
        self,
        request: str,
        language: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate code with optimal model selection."""
        return await self.router_agent.generate_code(
            session_id=session_id or self.default_session_id,
            request=request,
            language=language,
            **kwargs
        )
    
    async def debug_code(
        self,
        code_or_error: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Debug code with models optimized for debugging."""
        return await self.router_agent.debug_code(
            session_id=session_id or self.default_session_id,
            code_or_error=code_or_error,
            **kwargs
        )
    
    async def review_code(
        self,
        code: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Review code with optimal model selection."""
        return await self.router_agent.review_code(
            session_id=session_id or self.default_session_id,
            code=code,
            **kwargs
        )
    
    async def plan_architecture(
        self,
        requirements: str,
        session_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Plan architecture with models optimized for system design."""
        return await self.router_agent.plan_architecture(
            session_id=session_id or self.default_session_id,
            requirements=requirements,
            **kwargs
        )
    
    # Semantic search methods
    
    async def semantic_search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.7,
        search_type: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search on indexed content.
        
        Args:
            query: Search query
            k: Number of results
            min_score: Minimum similarity score
            search_type: Search type ('semantic', 'keyword', 'hybrid')
            
        Returns:
            List of search results
        """
        
        if not self.semantic_search:
            self.logger.warning("Semantic search not enabled")
            return []
        
        try:
            results = await self.semantic_search.search(
                query=query,
                k=k,
                min_score=min_score,
                search_type=search_type
            )
            
            return [result.to_dict() for result in results]
            
        except Exception as e:
            self.logger.error(f"Semantic search failed: {e}")
            return []
    
    async def search_and_answer(
        self,
        question: str,
        session_id: Optional[str] = None,
        max_context_results: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Search for relevant context and answer question.
        
        Args:
            question: Question to answer
            session_id: Session identifier
            max_context_results: Maximum search results to include
            **kwargs: Additional parameters
            
        Returns:
            Answer with search context
        """
        
        # Perform semantic search for context
        search_results = await self.semantic_search(
            query=question,
            k=max_context_results,
            search_type="hybrid"
        )
        
        # Prepare enhanced system context
        context_parts = []
        if search_results:
            context_parts.append("Relevant context from project:")
            for i, result in enumerate(search_results, 1):
                context_parts.append(f"{i}. {result['content'][:300]}...")
        
        enhanced_system_context = "\n".join(context_parts) if context_parts else None
        
        # Answer with context
        result = await self.chat(
            message=question,
            session_id=session_id,
            system_prompt=enhanced_system_context,
            **kwargs
        )
        
        # Add search metadata
        result["search_results"] = search_results
        result["context_enhanced"] = bool(search_results)
        
        return result
    
    # Helper methods
    
    def _detect_task_type(self, task_description: str) -> TaskType:
        """Auto-detect task type from description."""
        
        description_lower = task_description.lower()
        
        # Code-related keywords
        if any(keyword in description_lower for keyword in 
               ['implement', 'code', 'function', 'class', 'method', 'algorithm']):
            return TaskType.CODE_GENERATION
        
        # Debugging keywords
        if any(keyword in description_lower for keyword in 
               ['debug', 'fix', 'error', 'issue', 'problem', 'bug']):
            return TaskType.DEBUGGING
        
        # Review keywords
        if any(keyword in description_lower for keyword in 
               ['review', 'check', 'audit', 'analyze code']):
            return TaskType.CODE_REVIEW
        
        # Architecture keywords
        if any(keyword in description_lower for keyword in 
               ['design', 'architecture', 'system', 'structure', 'plan']):
            return TaskType.ARCHITECTURE
        
        # Documentation keywords
        if any(keyword in description_lower for keyword in 
               ['document', 'documentation', 'explain', 'describe']):
            return TaskType.DOCUMENTATION
        
        # Testing keywords
        if any(keyword in description_lower for keyword in 
               ['test', 'testing', 'unit test', 'integration test']):
            return TaskType.TESTING
        
        # Refactoring keywords
        if any(keyword in description_lower for keyword in 
               ['refactor', 'restructure', 'cleanup', 'optimize']):
            return TaskType.REFACTORING
        
        # Research keywords
        if any(keyword in description_lower for keyword in 
               ['research', 'investigate', 'find', 'search', 'learn']):
            return TaskType.RESEARCH
        
        # Analysis keywords
        if any(keyword in description_lower for keyword in 
               ['analyze', 'analysis', 'examine', 'evaluate']):
            return TaskType.ANALYSIS
        
        # Default to conversation
        return TaskType.CONVERSATION
    
    def _map_workflow_to_task_type(self, domain_name: str, workflow_name: str) -> TaskType:
        """Map domain workflow to task type."""
        
        workflow_mapping = {
            'architecture_review': TaskType.ARCHITECTURE,
            'code_quality_audit': TaskType.CODE_REVIEW,
            'system_analysis': TaskType.ANALYSIS,
            'migration_planning': TaskType.ARCHITECTURE,
            'comprehensive_project_planning': TaskType.ORCHESTRATION,
            'web_application_planning': TaskType.ARCHITECTURE,
            'microservices_planning': TaskType.ARCHITECTURE
        }
        
        return workflow_mapping.get(workflow_name, TaskType.ORCHESTRATION)
    
    def _prepare_system_context(
        self,
        requirements: Optional[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]]
    ) -> str:
        """Prepare system context from requirements and constraints."""
        
        context_parts = []
        
        if requirements:
            context_parts.append("Requirements:")
            for key, value in requirements.items():
                context_parts.append(f"- {key}: {value}")
        
        if constraints:
            context_parts.append("Constraints:")
            for key, value in constraints.items():
                context_parts.append(f"- {key}: {value}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _determine_complexity(
        self,
        task_description: str,
        requirements: Optional[Dict[str, Any]]
    ) -> str:
        """Determine task complexity."""
        
        # Simple heuristics for complexity
        complexity_indicators = {
            'high': ['complex', 'advanced', 'sophisticated', 'enterprise', 'scalable', 'distributed'],
            'medium': ['moderate', 'standard', 'typical', 'regular'],
            'low': ['simple', 'basic', 'easy', 'quick', 'small']
        }
        
        description_lower = task_description.lower()
        
        for level, keywords in complexity_indicators.items():
            if any(keyword in description_lower for keyword in keywords):
                return level
        
        # Check requirements for complexity indicators
        if requirements:
            req_str = str(requirements).lower()
            for level, keywords in complexity_indicators.items():
                if any(keyword in req_str for keyword in keywords):
                    return level
        
        # Default to medium complexity
        return "medium"
    
    # Status and management methods
    
    def get_routing_status(self) -> Dict[str, Any]:
        """Get status of the routing system."""
        status = {
            "router_status": self.router_agent.get_router_status(),
            "active_sessions": len(self.active_sessions),
            "semantic_search_enabled": self.semantic_search is not None,
            "rag_enabled": self.enable_rag,
            "memory_enabled": self.enable_memory
        }
        
        if self.semantic_search:
            status["semantic_search_stats"] = self.semantic_search.get_search_stats()
        
        return status
    
    def get_session_info(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        if session_id is None:
            session_id = self.default_session_id
        
        # Get router session info
        router_info = self.router_agent.get_session_info(session_id)
        
        # Add local session info
        local_info = self.active_sessions.get(session_id, {})
        
        if router_info:
            router_info.update(local_info)
            return router_info
        elif local_info:
            return local_info
        else:
            return None
    
    def clear_session(self, session_id: Optional[str] = None):
        """Clear a session."""
        if session_id is None:
            session_id = self.default_session_id
        
        # Clear from router agent
        self.router_agent.clear_session(session_id)
        
        # Clear local session info
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    async def shutdown(self):
        """Shutdown the agent and all systems."""
        
        await self.router_agent.shutdown()
        
        if self.semantic_search:
            await self.semantic_search.cleanup()
        
        # Shutdown base agent
        await super().shutdown()
        
        self.logger.info("RoutedSelfReflectingAgent shutdown complete")


# Convenience function to create a routed agent with automatic configuration detection
async def create_routed_agent(
    project_path: Optional[str] = None,
    auto_detect_models: bool = True,
    **kwargs
) -> RoutedSelfReflectingAgent:
    """
    Create a routed agent with automatic configuration.
    
    Args:
        project_path: Path to project directory
        auto_detect_models: Automatically detect available models from environment
        **kwargs: Additional arguments for agent initialization
        
    Returns:
        Initialized routed agent
    """
    
    # Auto-detect project path if not provided
    if project_path is None:
        project_path = os.getcwd()
    
    # Create agent
    agent = RoutedSelfReflectingAgent(
        project_path=project_path,
        **kwargs
    )
    
    # Initialize
    await agent.initialize()
    
    return agent