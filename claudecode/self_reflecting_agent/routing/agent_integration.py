"""
Agent Integration Layer

Integrates the context-aware model router with the existing self-reflecting agent system,
providing seamless model switching with RAG and semantic search capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import asdict

from .context_aware_router import ContextAwareRouter, ConversationTurn
from .model_router import TaskType, TaskContext, ModelProvider
from ..rag.hybrid_rag import HybridRAGSystem
from ..memory.mem0_integration import Mem0Memory


class RouterIntegratedAgent:
    """
    Enhanced agent that uses intelligent model routing with context preservation
    and RAG/semantic search integration.
    """
    
    def __init__(
        self,
        router_config_path: Optional[Path] = None,
        rag_config: Optional[Dict[str, Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        agent_id: str = "default"
    ):
        self.logger = logging.getLogger(__name__)
        self.agent_id = agent_id
        
        # Initialize context-aware router
        self.router = ContextAwareRouter(router_config_path)
        
        # Initialize RAG system
        self.rag_system = None
        if rag_config:
            try:
                self.rag_system = HybridRAGSystem(rag_config)
                self.logger.info("RAG system initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize RAG system: {e}")
        
        # Initialize memory system
        self.memory_system = None
        if memory_config:
            try:
                self.memory_system = Mem0Memory(memory_config)
                self.logger.info("Memory system initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize memory system: {e}")
        
        # Track active sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Model client cache
        self.model_clients: Dict[str, Any] = {}
    
    async def initialize(self):
        """Initialize all systems."""
        if self.rag_system:
            await self.rag_system.initialize()
        
        if self.memory_system:
            await self.memory_system.initialize()
        
        self.logger.info("RouterIntegratedAgent initialized")
    
    async def process_request(
        self,
        session_id: str,
        user_message: str,
        task_type: TaskType = TaskType.CONVERSATION,
        complexity: str = "medium",
        system_context: Optional[str] = None,
        use_rag: bool = True,
        use_memory: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user request with intelligent routing, RAG, and memory integration.
        
        Args:
            session_id: Unique session identifier
            user_message: User's message/request
            task_type: Type of task for routing
            complexity: Task complexity level
            system_context: System context/instructions
            use_rag: Whether to use RAG for context enhancement
            use_memory: Whether to use memory for context
            **kwargs: Additional parameters
            
        Returns:
            Response dictionary with model output and metadata
        """
        
        start_time = time.time()
        
        try:
            # Initialize session if needed
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    "created_at": start_time,
                    "total_requests": 0,
                    "models_used": set(),
                    "total_tokens": 0
                }
            
            session_info = self.active_sessions[session_id]
            session_info["total_requests"] += 1
            
            # Enhance context with RAG and memory if available
            enhanced_context = await self._enhance_context_with_rag_and_memory(
                user_message=user_message,
                session_id=session_id,
                use_rag=use_rag,
                use_memory=use_memory,
                system_context=system_context
            )
            
            # Create task context
            task_context = TaskContext(
                task_type=task_type,
                complexity=complexity,
                estimated_tokens=self._estimate_tokens(user_message, enhanced_context),
                requires_code=self._requires_code(user_message, task_type),
                requires_reasoning=self._requires_reasoning(user_message, task_type),
                requires_vision=self._requires_vision(user_message),
                requires_function_calling=self._requires_function_calling(user_message, task_type),
                latency_sensitive=kwargs.get("latency_sensitive", False),
                cost_sensitive=kwargs.get("cost_sensitive", False),
                context_data=kwargs
            )
            
            # Route with context
            routing_decision, context_state = await self.router.route_with_context(
                session_id=session_id,
                task_context=task_context,
                user_message=user_message,
                system_context=enhanced_context["system_context"]
            )
            
            self.logger.info(f"Routed to {routing_decision.selected_model}: {routing_decision.reasoning}")
            
            # Get model client and execute request
            model_client = await self._get_model_client(routing_decision.selected_model)
            formatted_context = self.router.get_context_for_model(session_id, routing_decision.selected_model)
            
            # Execute request with fallback handling
            response_result = await self._execute_with_fallback(
                session_id=session_id,
                routing_decision=routing_decision,
                formatted_context=formatted_context,
                task_context=task_context
            )
            
            # Update session tracking
            session_info["models_used"].add(routing_decision.selected_model)
            if "tokens_used" in response_result:
                session_info["total_tokens"] += response_result["tokens_used"]
            
            # Add assistant response to context
            await self.router.add_assistant_response(
                session_id=session_id,
                response=response_result["content"],
                model_used=routing_decision.selected_model,
                tokens_used=response_result.get("tokens_used")
            )
            
            # Update memory with new interaction if available
            if self.memory_system and use_memory:
                await self._update_memory(
                    session_id=session_id,
                    user_message=user_message,
                    assistant_response=response_result["content"],
                    model_used=routing_decision.selected_model
                )
            
            # Prepare response
            total_time = time.time() - start_time
            
            return {
                "content": response_result["content"],
                "model_used": routing_decision.selected_model,
                "provider": routing_decision.provider.value,
                "routing_reasoning": routing_decision.reasoning,
                "fallback_models": routing_decision.fallback_models,
                "execution_time_ms": int(total_time * 1000),
                "tokens_used": response_result.get("tokens_used"),
                "estimated_cost": routing_decision.estimated_cost,
                "context_stats": self.router.get_context_stats(session_id),
                "rag_enhanced": use_rag and bool(enhanced_context.get("rag_context")),
                "memory_enhanced": use_memory and bool(enhanced_context.get("memory_context")),
                "session_id": session_id,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Request processing failed: {e}")
            
            return {
                "content": f"I apologize, but I encountered an error processing your request: {str(e)}",
                "model_used": None,
                "provider": None,
                "routing_reasoning": f"Error: {str(e)}",
                "fallback_models": [],
                "execution_time_ms": int((time.time() - start_time) * 1000),
                "tokens_used": 0,
                "estimated_cost": 0.0,
                "context_stats": None,
                "rag_enhanced": False,
                "memory_enhanced": False,
                "session_id": session_id,
                "success": False,
                "error": str(e)
            }
    
    async def _enhance_context_with_rag_and_memory(
        self,
        user_message: str,
        session_id: str,
        use_rag: bool,
        use_memory: bool,
        system_context: Optional[str]
    ) -> Dict[str, Any]:
        """Enhance context using RAG and memory systems."""
        
        enhanced_context = {
            "system_context": system_context or "",
            "rag_context": "",
            "memory_context": "",
            "total_enhancement_tokens": 0
        }
        
        context_parts = []
        
        # Add original system context
        if system_context:
            context_parts.append(system_context)
        
        # Enhance with RAG if available
        if use_rag and self.rag_system:
            try:
                rag_results = await self.rag_system.search(
                    query=user_message,
                    k=5,
                    min_score=0.7
                )
                
                if rag_results:
                    rag_context_parts = []
                    for result in rag_results:
                        rag_context_parts.append(f"- {result['content'][:200]}...")
                    
                    rag_context = "Relevant context from knowledge base:\n" + "\n".join(rag_context_parts)
                    enhanced_context["rag_context"] = rag_context
                    context_parts.append(rag_context)
                    
                    self.logger.debug(f"Enhanced with {len(rag_results)} RAG results")
                
            except Exception as e:
                self.logger.warning(f"RAG enhancement failed: {e}")
        
        # Enhance with memory if available
        if use_memory and self.memory_system:
            try:
                memory_results = await self.memory_system.search(
                    query=user_message,
                    user_id=session_id,
                    limit=3
                )
                
                if memory_results:
                    memory_context_parts = []
                    for result in memory_results:
                        memory_context_parts.append(f"- Previous context: {result.get('text', '')[:150]}...")
                    
                    memory_context = "Relevant previous context:\n" + "\n".join(memory_context_parts)
                    enhanced_context["memory_context"] = memory_context
                    context_parts.append(memory_context)
                    
                    self.logger.debug(f"Enhanced with {len(memory_results)} memory results")
                
            except Exception as e:
                self.logger.warning(f"Memory enhancement failed: {e}")
        
        # Combine all context parts
        if len(context_parts) > 1:  # More than just the original system context
            enhanced_context["system_context"] = "\n\n".join(context_parts)
            enhanced_context["total_enhancement_tokens"] = len(enhanced_context["system_context"])
        
        return enhanced_context
    
    def _estimate_tokens(self, user_message: str, enhanced_context: Dict[str, Any]) -> int:
        """Estimate total tokens for the request."""
        # Rough estimation: 1 token ≈ 4 characters
        base_tokens = len(user_message) // 4
        context_tokens = enhanced_context.get("total_enhancement_tokens", 0) // 4
        
        # Add buffer for response
        return base_tokens + context_tokens + 1000
    
    def _requires_code(self, user_message: str, task_type: TaskType) -> bool:
        """Determine if the request requires code capabilities."""
        code_indicators = [
            "code", "function", "class", "method", "algorithm", "implement",
            "debug", "fix", "refactor", "programming", "script", "syntax"
        ]
        
        # Task type based determination
        if task_type in [TaskType.CODE_GENERATION, TaskType.CODE_REVIEW, 
                        TaskType.DEBUGGING, TaskType.REFACTORING, TaskType.TESTING]:
            return True
        
        # Content based determination
        return any(indicator in user_message.lower() for indicator in code_indicators)
    
    def _requires_reasoning(self, user_message: str, task_type: TaskType) -> bool:
        """Determine if the request requires reasoning capabilities."""
        reasoning_indicators = [
            "analyze", "explain", "why", "how", "compare", "evaluate",
            "decide", "choose", "recommend", "strategy", "plan", "design"
        ]
        
        # Most task types require reasoning
        if task_type in [TaskType.ORCHESTRATION, TaskType.ARCHITECTURE, 
                        TaskType.ANALYSIS, TaskType.REASONING]:
            return True
        
        return any(indicator in user_message.lower() for indicator in reasoning_indicators)
    
    def _requires_vision(self, user_message: str) -> bool:
        """Determine if the request requires vision capabilities."""
        vision_indicators = [
            "image", "picture", "photo", "screenshot", "diagram", "chart",
            "visual", "see", "look at", "analyze image"
        ]
        
        return any(indicator in user_message.lower() for indicator in vision_indicators)
    
    def _requires_function_calling(self, user_message: str, task_type: TaskType) -> bool:
        """Determine if the request requires function calling capabilities."""
        function_indicators = [
            "call", "execute", "run", "api", "function", "tool", "command"
        ]
        
        # Certain task types commonly use function calling
        if task_type in [TaskType.TESTING, TaskType.DEBUGGING]:
            return True
        
        return any(indicator in user_message.lower() for indicator in function_indicators)
    
    async def _get_model_client(self, model_name: str) -> Any:
        """Get or create a client for the specified model."""
        if model_name in self.model_clients:
            return self.model_clients[model_name]
        
        model_config = self.router.base_router.models[model_name]
        client = None
        
        try:
            if model_config.provider == ModelProvider.ANTHROPIC:
                import anthropic
                api_key = os.getenv(model_config.api_key_env)
                client = anthropic.Anthropic(api_key=api_key)
                
            elif model_config.provider == ModelProvider.OPENAI:
                import openai
                api_key = os.getenv(model_config.api_key_env)
                client = openai.OpenAI(api_key=api_key)
                
            elif model_config.provider == ModelProvider.GOOGLE:
                import google.generativeai as genai
                api_key = os.getenv(model_config.api_key_env)
                genai.configure(api_key=api_key)
                client = genai.GenerativeModel(model_name)
            
            if client:
                self.model_clients[model_name] = client
            
            return client
            
        except ImportError as e:
            self.logger.error(f"Required library not installed for {model_config.provider}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to create client for {model_name}: {e}")
            raise
    
    async def _execute_with_fallback(
        self,
        session_id: str,
        routing_decision,
        formatted_context: Dict[str, Any],
        task_context: TaskContext
    ) -> Dict[str, Any]:
        """Execute request with fallback handling."""
        
        models_to_try = [routing_decision.selected_model] + routing_decision.fallback_models
        last_error = None
        
        for model_name in models_to_try:
            try:
                self.logger.info(f"Attempting request with {model_name}")
                
                start_time = time.time()
                result = await self._execute_single_model(model_name, formatted_context, task_context)
                execution_time = int((time.time() - start_time) * 1000)
                
                # Record success
                await self.router.record_task_result(
                    session_id=session_id,
                    model_name=model_name,
                    success=True,
                    latency_ms=execution_time,
                    cost=result.get("cost", 0.0)
                )
                
                return result
                
            except Exception as e:
                last_error = e
                execution_time = int((time.time() - start_time) * 1000)
                
                # Record failure
                await self.router.record_task_result(
                    session_id=session_id,
                    model_name=model_name,
                    success=False,
                    latency_ms=execution_time,
                    error_reason=str(e)
                )
                
                self.logger.warning(f"Model {model_name} failed: {e}")
                
                # If this wasn't the last model, try the next one
                if model_name != models_to_try[-1]:
                    continue
        
        # All models failed
        raise RuntimeError(f"All models failed. Last error: {last_error}")
    
    async def _execute_single_model(
        self,
        model_name: str,
        formatted_context: Dict[str, Any],
        task_context: TaskContext
    ) -> Dict[str, Any]:
        """Execute request on a single model."""
        
        model_config = self.router.base_router.models[model_name]
        client = await self._get_model_client(model_name)
        
        if model_config.provider == ModelProvider.ANTHROPIC:
            return await self._execute_anthropic(client, model_name, formatted_context, model_config)
        elif model_config.provider == ModelProvider.OPENAI:
            return await self._execute_openai(client, model_name, formatted_context, model_config)
        elif model_config.provider == ModelProvider.GOOGLE:
            return await self._execute_google(client, model_name, formatted_context, model_config)
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
    
    async def _execute_anthropic(self, client, model_name: str, context: Dict[str, Any], config) -> Dict[str, Any]:
        """Execute request on Anthropic Claude."""
        
        # Prepare parameters
        params = {
            "model": model_name,
            "messages": context["messages"],
            "max_tokens": config.model_params.get("max_tokens", 8192),
            "temperature": config.model_params.get("temperature", 0.1)
        }
        
        response = client.messages.create(**params)
        
        return {
            "content": response.content[0].text,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
            "cost": self._calculate_cost(
                response.usage.input_tokens + response.usage.output_tokens,
                config.capabilities.cost_per_1k_tokens
            )
        }
    
    async def _execute_openai(self, client, model_name: str, context: Dict[str, Any], config) -> Dict[str, Any]:
        """Execute request on OpenAI GPT."""
        
        params = {
            "model": model_name,
            "messages": context["messages"],
            "max_tokens": config.model_params.get("max_tokens", 4096),
            "temperature": config.model_params.get("temperature", 0.1)
        }
        
        response = client.chat.completions.create(**params)
        
        total_tokens = response.usage.prompt_tokens + response.usage.completion_tokens
        
        return {
            "content": response.choices[0].message.content,
            "tokens_used": total_tokens,
            "cost": self._calculate_cost(total_tokens, config.capabilities.cost_per_1k_tokens)
        }
    
    async def _execute_google(self, client, model_name: str, context: Dict[str, Any], config) -> Dict[str, Any]:
        """Execute request on Google Gemini."""
        
        # Gemini has different API structure
        if context.get("history"):
            # Multi-turn conversation
            chat = client.start_chat(history=context["history"])
            response = chat.send_message(context["message"])
        else:
            # Single message
            response = client.generate_content(context["message"])
        
        # Google doesn't provide token usage in the same way, estimate it
        estimated_tokens = len(response.text) // 4  # Rough estimation
        
        return {
            "content": response.text,
            "tokens_used": estimated_tokens,
            "cost": self._calculate_cost(estimated_tokens, config.capabilities.cost_per_1k_tokens)
        }
    
    def _calculate_cost(self, tokens: int, cost_per_1k: Optional[float]) -> float:
        """Calculate cost for token usage."""
        if not cost_per_1k:
            return 0.0
        return (tokens / 1000.0) * cost_per_1k
    
    async def _update_memory(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        model_used: str
    ):
        """Update memory system with the interaction."""
        try:
            await self.memory_system.add(
                messages=[
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_response}
                ],
                user_id=session_id,
                metadata={
                    "model_used": model_used,
                    "timestamp": time.time(),
                    "agent_id": self.agent_id
                }
            )
        except Exception as e:
            self.logger.warning(f"Failed to update memory: {e}")
    
    # Convenience methods for common operations
    
    async def chat(
        self,
        session_id: str,
        message: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Simple chat interface."""
        return await self.process_request(
            session_id=session_id,
            user_message=message,
            task_type=TaskType.CONVERSATION,
            system_context=system_prompt,
            **kwargs
        )
    
    async def generate_code(
        self,
        session_id: str,
        request: str,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate code interface."""
        system_context = f"You are an expert programmer. Generate high-quality {language or 'code'} that follows best practices."
        
        return await self.process_request(
            session_id=session_id,
            user_message=request,
            task_type=TaskType.CODE_GENERATION,
            complexity="medium",
            system_context=system_context,
            **kwargs
        )
    
    async def debug_code(
        self,
        session_id: str,
        code_or_error: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Debug code interface."""
        system_context = "You are an expert debugger. Analyze the code or error and provide clear solutions."
        
        return await self.process_request(
            session_id=session_id,
            user_message=code_or_error,
            task_type=TaskType.DEBUGGING,
            complexity="high",
            system_context=system_context,
            latency_sensitive=True,
            **kwargs
        )
    
    async def review_code(
        self,
        session_id: str,
        code: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Code review interface."""
        system_context = "You are an expert code reviewer. Provide detailed feedback on code quality, best practices, and potential improvements."
        
        return await self.process_request(
            session_id=session_id,
            user_message=code,
            task_type=TaskType.CODE_REVIEW,
            complexity="medium",
            system_context=system_context,
            **kwargs
        )
    
    async def plan_architecture(
        self,
        session_id: str,
        requirements: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Architecture planning interface."""
        system_context = "You are a senior software architect. Design scalable, maintainable systems that follow architectural best practices."
        
        return await self.process_request(
            session_id=session_id,
            user_message=requirements,
            task_type=TaskType.ARCHITECTURE,
            complexity="high",
            system_context=system_context,
            **kwargs
        )
    
    # System management methods
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session."""
        if session_id not in self.active_sessions:
            return None
        
        session_info = self.active_sessions[session_id].copy()
        session_info["models_used"] = list(session_info["models_used"])
        session_info["context_stats"] = self.router.get_context_stats(session_id)
        
        return session_info
    
    def get_router_status(self) -> Dict[str, Any]:
        """Get status of the model router."""
        return self.router.base_router.get_model_status()
    
    def clear_session(self, session_id: str):
        """Clear a session completely."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        self.router.clear_context(session_id)
    
    async def shutdown(self):
        """Shutdown the agent and all systems."""
        if self.rag_system:
            await self.rag_system.cleanup()
        
        if self.memory_system:
            await self.memory_system.cleanup()
        
        # Close model clients
        for client in self.model_clients.values():
            if hasattr(client, 'close'):
                try:
                    await client.close()
                except:
                    pass
        
        self.logger.info("RouterIntegratedAgent shutdown complete")