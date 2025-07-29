"""
Context-Aware Model Router

Extends the base model router to maintain context continuity when switching
between models with different capabilities and context windows.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .model_router import (
    ModelRouter, TaskContext, RoutingDecision, TaskType, ModelProvider
)


@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None


@dataclass
class ContextState:
    """Current context state for a conversation."""
    conversation_history: List[ConversationTurn]
    system_context: str
    task_context: Dict[str, Any]
    current_model: Optional[str] = None
    total_tokens_used: int = 0
    
    def get_total_context_length(self) -> int:
        """Calculate total context length."""
        total_length = len(self.system_context)
        
        for turn in self.conversation_history:
            total_length += len(turn.content)
            
        return total_length
    
    def get_conversation_text(self) -> str:
        """Get conversation as formatted text."""
        parts = []
        
        if self.system_context:
            parts.append(f"System: {self.system_context}")
        
        for turn in self.conversation_history:
            parts.append(f"{turn.role.title()}: {turn.content}")
            
        return "\n\n".join(parts)


class ContextCompressor:
    """Handles context compression when switching to models with smaller context windows."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def compress_context(
        self, 
        context_state: ContextState, 
        target_context_length: int,
        compression_strategy: str = "summarize"
    ) -> ContextState:
        """
        Compress context to fit within target length.
        
        Args:
            context_state: Current context state
            target_context_length: Target maximum context length
            compression_strategy: "summarize" or "truncate"
            
        Returns:
            Compressed context state
        """
        
        current_length = context_state.get_total_context_length()
        
        if current_length <= target_context_length:
            return context_state  # No compression needed
        
        self.logger.info(f"Compressing context from {current_length} to {target_context_length} chars")
        
        if compression_strategy == "summarize":
            return await self._summarize_context(context_state, target_context_length)
        else:
            return self._truncate_context(context_state, target_context_length)
    
    async def _summarize_context(self, context_state: ContextState, target_length: int) -> ContextState:
        """Summarize context using intelligent summarization."""
        
        # Keep system context and recent turns, summarize older conversation
        compressed_state = ContextState(
            conversation_history=[],
            system_context=context_state.system_context,
            task_context=context_state.task_context.copy(),
            current_model=context_state.current_model,
            total_tokens_used=context_state.total_tokens_used
        )
        
        # Always keep the last few turns
        recent_turns = context_state.conversation_history[-3:] if len(context_state.conversation_history) > 3 else context_state.conversation_history
        older_turns = context_state.conversation_history[:-3] if len(context_state.conversation_history) > 3 else []
        
        # Calculate space for recent turns and system context
        system_length = len(context_state.system_context)
        recent_length = sum(len(turn.content) for turn in recent_turns)
        
        available_for_summary = target_length - system_length - recent_length - 500  # Buffer
        
        if older_turns and available_for_summary > 200:
            # Create summary of older conversation
            summary_text = await self._create_conversation_summary(older_turns, available_for_summary)
            
            summary_turn = ConversationTurn(
                role="system",
                content=f"[Previous conversation summary: {summary_text}]",
                metadata={"type": "summary", "original_turns": len(older_turns)}
            )
            
            compressed_state.conversation_history.append(summary_turn)
        
        # Add recent turns
        compressed_state.conversation_history.extend(recent_turns)
        
        return compressed_state
    
    def _truncate_context(self, context_state: ContextState, target_length: int) -> ContextState:
        """Truncate context by removing older conversation turns."""
        
        compressed_state = ContextState(
            conversation_history=[],
            system_context=context_state.system_context,
            task_context=context_state.task_context.copy(),
            current_model=context_state.current_model,
            total_tokens_used=context_state.total_tokens_used
        )
        
        # Start with system context
        current_length = len(context_state.system_context)
        
        # Add conversation turns from most recent backwards
        for turn in reversed(context_state.conversation_history):
            turn_length = len(turn.content)
            
            if current_length + turn_length <= target_length:
                compressed_state.conversation_history.insert(0, turn)
                current_length += turn_length
            else:
                break
        
        return compressed_state
    
    async def _create_conversation_summary(self, turns: List[ConversationTurn], max_length: int) -> str:
        """Create a summary of conversation turns."""
        
        # Simple extractive summary for now
        # In a full implementation, this would use a summarization model
        
        conversation_text = "\n".join([f"{turn.role}: {turn.content}" for turn in turns])
        
        if len(conversation_text) <= max_length:
            return conversation_text
        
        # Extract key points (simple heuristic)
        sentences = conversation_text.split('. ')
        key_sentences = []
        current_length = 0
        
        # Prioritize sentences with certain keywords
        priority_keywords = ['error', 'issue', 'problem', 'solution', 'implement', 'create', 'design', 'fix']
        
        # First pass: add high-priority sentences
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in priority_keywords):
                if current_length + len(sentence) <= max_length:
                    key_sentences.append(sentence)
                    current_length += len(sentence)
        
        # Second pass: fill remaining space with other sentences
        for sentence in sentences:
            if sentence not in key_sentences:
                if current_length + len(sentence) <= max_length:
                    key_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break
        
        return '. '.join(key_sentences)


class ContextAwareRouter:
    """
    Model router that maintains context continuity across model switches.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize base router
        self.base_router = ModelRouter(config_path)
        
        # Initialize context management
        self.context_compressor = ContextCompressor()
        
        # Load context management configuration
        self.context_config = self.base_router.config.get('context_management', {})
        self.preserve_context = self.context_config.get('preserve_context', True)
        self.max_preserved_context = self.context_config.get('max_preserved_context', 50000)
        self.compression_strategy = self.context_config.get('compression_strategy', 'summarize')
        self.include_conversation_history = self.context_config.get('include_conversation_history', True)
        self.max_conversation_turns = self.context_config.get('max_conversation_turns', 10)
        
        # Active context states (keyed by session/conversation ID)
        self.context_states: Dict[str, ContextState] = {}
    
    async def route_with_context(
        self, 
        session_id: str,
        task_context: TaskContext,
        user_message: str,
        system_context: Optional[str] = None
    ) -> Tuple[RoutingDecision, ContextState]:
        """
        Route a task while maintaining context continuity.
        
        Args:
            session_id: Unique identifier for the conversation session
            task_context: Task context for routing
            user_message: The user's message
            system_context: System context/instructions
            
        Returns:
            Tuple of (routing_decision, updated_context_state)
        """
        
        # Get or create context state
        if session_id not in self.context_states:
            self.context_states[session_id] = ContextState(
                conversation_history=[],
                system_context=system_context or "",
                task_context=asdict(task_context)
            )
        
        context_state = self.context_states[session_id]
        
        # Update task context if provided
        if system_context:
            context_state.system_context = system_context
        
        # Add user message to conversation history
        user_turn = ConversationTurn(
            role="user",
            content=user_message,
            timestamp=str(asyncio.get_event_loop().time())
        )
        
        # Estimate total context length including new message
        estimated_context_length = (
            context_state.get_total_context_length() + 
            len(user_message) + 
            task_context.estimated_tokens
        )
        
        # Update task context with estimated length
        task_context.estimated_tokens = estimated_context_length
        
        # Get routing decision
        routing_decision = await self.base_router.route_task(task_context)
        
        # Check if we need to compress context for the selected model
        selected_model = self.base_router.models[routing_decision.selected_model]
        max_context_length = selected_model.capabilities.context_length
        
        # Leave some buffer for the response
        available_context = max_context_length - task_context.estimated_tokens - 1000
        
        if estimated_context_length > available_context:
            self.logger.info(f"Context too large ({estimated_context_length}), compressing for {routing_decision.selected_model}")
            
            context_state = await self.context_compressor.compress_context(
                context_state,
                available_context,
                self.compression_strategy
            )
        
        # Add user turn to context
        context_state.conversation_history.append(user_turn)
        
        # Limit conversation history if needed
        if len(context_state.conversation_history) > self.max_conversation_turns:
            # Keep system messages and most recent turns
            system_turns = [turn for turn in context_state.conversation_history if turn.role == "system"]
            recent_turns = [turn for turn in context_state.conversation_history if turn.role != "system"][-self.max_conversation_turns:]
            
            context_state.conversation_history = system_turns + recent_turns
        
        # Update current model
        context_state.current_model = routing_decision.selected_model
        
        # Store updated context state
        self.context_states[session_id] = context_state
        
        return routing_decision, context_state
    
    async def add_assistant_response(
        self, 
        session_id: str, 
        response: str, 
        model_used: str,
        tokens_used: Optional[int] = None
    ):
        """
        Add assistant response to conversation history.
        
        Args:
            session_id: Session identifier
            response: Assistant's response
            model_used: Model that generated the response
            tokens_used: Number of tokens used (optional)
        """
        
        if session_id not in self.context_states:
            self.logger.warning(f"No context state found for session {session_id}")
            return
        
        context_state = self.context_states[session_id]
        
        assistant_turn = ConversationTurn(
            role="assistant",
            content=response,
            model_used=model_used,
            tokens_used=tokens_used,
            timestamp=str(asyncio.get_event_loop().time())
        )
        
        context_state.conversation_history.append(assistant_turn)
        
        if tokens_used:
            context_state.total_tokens_used += tokens_used
    
    def get_context_for_model(self, session_id: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get formatted context for a specific model.
        
        Args:
            session_id: Session identifier
            model_name: Target model name
            
        Returns:
            Formatted context for the model
        """
        
        if session_id not in self.context_states:
            return None
        
        context_state = self.context_states[session_id]
        model_config = self.base_router.models.get(model_name)
        
        if not model_config:
            return None
        
        # Format context based on model provider
        if model_config.provider == ModelProvider.ANTHROPIC:
            return self._format_context_for_anthropic(context_state)
        elif model_config.provider == ModelProvider.OPENAI:
            return self._format_context_for_openai(context_state)
        elif model_config.provider == ModelProvider.GOOGLE:
            return self._format_context_for_google(context_state)
        else:
            return self._format_context_generic(context_state)
    
    def _format_context_for_anthropic(self, context_state: ContextState) -> Dict[str, Any]:
        """Format context for Anthropic Claude."""
        messages = []
        
        # Add system context if present
        if context_state.system_context:
            messages.append({
                "role": "system",
                "content": context_state.system_context
            })
        
        # Add conversation history
        for turn in context_state.conversation_history:
            if turn.role in ["user", "assistant"]:
                messages.append({
                    "role": turn.role,
                    "content": turn.content
                })
        
        return {
            "messages": messages,
            "provider_format": "anthropic"
        }
    
    def _format_context_for_openai(self, context_state: ContextState) -> Dict[str, Any]:
        """Format context for OpenAI GPT."""
        messages = []
        
        # Add system context if present
        if context_state.system_context:
            messages.append({
                "role": "system",
                "content": context_state.system_context
            })
        
        # Add conversation history
        for turn in context_state.conversation_history:
            if turn.role in ["user", "assistant", "system"]:
                messages.append({
                    "role": turn.role,
                    "content": turn.content
                })
        
        return {
            "messages": messages,
            "provider_format": "openai"
        }
    
    def _format_context_for_google(self, context_state: ContextState) -> Dict[str, Any]:
        """Format context for Google Gemini."""
        # Gemini uses a different format
        history = []
        
        # Combine system context with first user message if present
        system_context = context_state.system_context
        
        current_user_content = None
        
        for turn in context_state.conversation_history:
            if turn.role == "user":
                if current_user_content is not None:
                    # We have a pending user message, this shouldn't happen in proper conversation
                    history.append({"role": "user", "parts": [current_user_content]})
                current_user_content = turn.content
            elif turn.role == "assistant" and current_user_content is not None:
                # Add user-assistant pair
                user_content = current_user_content
                if system_context and len(history) == 0:
                    # Add system context to first user message
                    user_content = f"{system_context}\n\n{user_content}"
                    system_context = None  # Don't add again
                
                history.append({"role": "user", "parts": [user_content]})
                history.append({"role": "model", "parts": [turn.content]})
                current_user_content = None
        
        # Handle any remaining user message
        if current_user_content is not None:
            user_content = current_user_content
            if system_context:
                user_content = f"{system_context}\n\n{user_content}"
            
            # For Gemini, we need to format the last user message separately
            return {
                "history": history[:-1] if history else [],  # All but the last exchange
                "message": user_content,  # Current user message
                "provider_format": "google"
            }
        
        return {
            "history": history,
            "message": "",
            "provider_format": "google"
        }
    
    def _format_context_generic(self, context_state: ContextState) -> Dict[str, Any]:
        """Generic context formatting."""
        return {
            "system_context": context_state.system_context,
            "conversation_history": [asdict(turn) for turn in context_state.conversation_history],
            "task_context": context_state.task_context,
            "provider_format": "generic"
        }
    
    def get_context_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics about the context state."""
        if session_id not in self.context_states:
            return None
        
        context_state = self.context_states[session_id]
        
        return {
            "session_id": session_id,
            "total_turns": len(context_state.conversation_history),
            "total_tokens_used": context_state.total_tokens_used,
            "current_context_length": context_state.get_total_context_length(),
            "current_model": context_state.current_model,
            "has_system_context": bool(context_state.system_context),
            "conversation_summary": self._get_conversation_summary(context_state)
        }
    
    def _get_conversation_summary(self, context_state: ContextState) -> str:
        """Get a brief summary of the conversation."""
        if not context_state.conversation_history:
            return "No conversation history"
        
        user_messages = [turn.content for turn in context_state.conversation_history if turn.role == "user"]
        
        if not user_messages:
            return "No user messages"
        
        # Simple summary based on first and last user messages
        if len(user_messages) == 1:
            return f"Single request: {user_messages[0][:100]}..."
        else:
            return f"Conversation from '{user_messages[0][:50]}...' to '{user_messages[-1][:50]}...'"
    
    def clear_context(self, session_id: str):
        """Clear context for a session."""
        if session_id in self.context_states:
            del self.context_states[session_id]
            self.logger.info(f"Cleared context for session {session_id}")
    
    def export_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export context state for persistence or analysis."""
        if session_id not in self.context_states:
            return None
        
        context_state = self.context_states[session_id]
        
        return {
            "session_id": session_id,
            "context_state": {
                "conversation_history": [asdict(turn) for turn in context_state.conversation_history],
                "system_context": context_state.system_context,
                "task_context": context_state.task_context,
                "current_model": context_state.current_model,
                "total_tokens_used": context_state.total_tokens_used
            },
            "export_timestamp": asyncio.get_event_loop().time()
        }
    
    def import_context(self, context_data: Dict[str, Any]):
        """Import previously exported context state."""
        session_id = context_data["session_id"]
        state_data = context_data["context_state"]
        
        # Reconstruct conversation history
        conversation_history = []
        for turn_data in state_data["conversation_history"]:
            conversation_history.append(ConversationTurn(**turn_data))
        
        # Reconstruct context state
        context_state = ContextState(
            conversation_history=conversation_history,
            system_context=state_data["system_context"],
            task_context=state_data["task_context"],
            current_model=state_data.get("current_model"),
            total_tokens_used=state_data.get("total_tokens_used", 0)
        )
        
        self.context_states[session_id] = context_state
        self.logger.info(f"Imported context for session {session_id}")
    
    async def record_task_result(
        self,
        session_id: str,
        model_name: str,
        success: bool,
        latency_ms: int,
        cost: float = 0.0,
        error_reason: Optional[str] = None
    ):
        """Record task result and update context if needed."""
        # Delegate to base router
        await self.base_router.record_task_result(
            model_name, success, latency_ms, cost, error_reason
        )
        
        # Update context state if there was an error and we need to try a different model
        if not success and session_id in self.context_states:
            context_state = self.context_states[session_id]
            
            # Add error information to context for potential retry with different model
            error_turn = ConversationTurn(
                role="system",
                content=f"[Error with {model_name}: {error_reason}]",
                metadata={"type": "error", "model": model_name, "error": error_reason}
            )
            
            # Only add if this isn't a duplicate error message
            if (not context_state.conversation_history or 
                context_state.conversation_history[-1].metadata != error_turn.metadata):
                context_state.conversation_history.append(error_turn)


# Convenience functions for common routing scenarios with context

async def route_with_context_for_orchestration(
    router: ContextAwareRouter,
    session_id: str,
    user_message: str,
    system_context: Optional[str] = None,
    complexity: str = "medium"
) -> Tuple[RoutingDecision, ContextState]:
    """Route orchestration task with context."""
    task_context = TaskContext(
        task_type=TaskType.ORCHESTRATION,
        complexity=complexity,
        estimated_tokens=4000,
        requires_reasoning=True,
        latency_sensitive=False
    )
    
    return await router.route_with_context(session_id, task_context, user_message, system_context)


async def route_with_context_for_debugging(
    router: ContextAwareRouter,
    session_id: str,
    user_message: str,
    system_context: Optional[str] = None,
    complexity: str = "high"
) -> Tuple[RoutingDecision, ContextState]:
    """Route debugging task with context."""
    task_context = TaskContext(
        task_type=TaskType.DEBUGGING,
        complexity=complexity,
        estimated_tokens=8000,
        requires_code=True,
        requires_reasoning=True,
        latency_sensitive=True
    )
    
    return await router.route_with_context(session_id, task_context, user_message, system_context)


async def route_with_context_for_code_generation(
    router: ContextAwareRouter,
    session_id: str,
    user_message: str,
    system_context: Optional[str] = None,
    complexity: str = "medium"
) -> Tuple[RoutingDecision, ContextState]:
    """Route code generation task with context."""
    task_context = TaskContext(
        task_type=TaskType.CODE_GENERATION,
        complexity=complexity,
        estimated_tokens=6000,
        requires_code=True,
        requires_reasoning=True
    )
    
    return await router.route_with_context(session_id, task_context, user_message, system_context)