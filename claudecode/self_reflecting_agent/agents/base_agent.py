"""
Base agent class providing common functionality for all agent types.

This module implements the foundation for all agents in the system, providing:
- DSPy integration for optimizable prompting
- Memory integration for persistent context
- Context management for efficient token usage
- Evaluation and self-improvement capabilities
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

import dspy
from pydantic import BaseModel, Field

from ..memory import AgentMemory
from ..context import ContextManager
from ..evaluation import AgentEvaluator


@dataclass
class AgentState:
    """Represents the current state of an agent."""
    agent_id: str
    current_task: Optional[str] = None
    context: Dict[str, Any] = None
    memory_context: List[str] = None
    last_update: datetime = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.memory_context is None:
            self.memory_context = []
        if self.last_update is None:
            self.last_update = datetime.now()


class AgentMessage(BaseModel):
    """Structured message format for agent communication."""
    sender: str = Field(description="Agent ID of the sender")
    recipient: str = Field(description="Agent ID of the recipient") 
    message_type: str = Field(description="Type of message (task, result, query, etc.)")
    content: Any = Field(description="Message content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now)


class BaseAgent(ABC):
    """
    Base class for all agents in the Self-Reflecting Claude Code Agent system.
    
    Provides common functionality including:
    - DSPy integration for optimizable prompting
    - Memory management and context awareness
    - Communication with other agents
    - Self-evaluation and improvement
    """
    
    def __init__(
        self,
        agent_id: str,
        config: Dict[str, Any],
        memory: Optional[AgentMemory] = None,
        context_manager: Optional[ContextManager] = None,
        evaluator: Optional[AgentEvaluator] = None,
        dspy_lm: Optional[dspy.LM] = None
    ):
        self.agent_id = agent_id
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
        # Core components
        self.memory = memory
        self.context_manager = context_manager
        self.evaluator = evaluator
        
        # DSPy setup
        if dspy_lm:
            dspy.configure(lm=dspy_lm)
            self.dspy_enabled = True
        else:
            self.dspy_enabled = False
            
        # Agent state
        self.state = AgentState(agent_id=agent_id)
        self.message_queue = asyncio.Queue()
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "avg_response_time": 0.0,
            "success_rate": 0.0,
            "last_evaluation_score": 0.0
        }
        
        self.logger.info(f"Initialized {self.__class__.__name__} with ID: {agent_id}")
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task assigned to this agent.
        
        Args:
            task: Task specification containing type, requirements, and context
            
        Returns:
            Result dictionary containing output, metadata, and status
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent type.
        
        Returns:
            System prompt string defining the agent's role and capabilities
        """
        pass
    
    async def send_message(self, recipient: str, message_type: str, content: Any, **metadata) -> None:
        """Send a message to another agent."""
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            metadata=metadata
        )
        
        # In a real implementation, this would route to the recipient
        # For now, we'll log the message
        self.logger.info(f"Sending {message_type} message to {recipient}: {str(content)[:100]}...")
    
    async def receive_message(self) -> Optional[AgentMessage]:
        """Receive a message from the message queue."""
        try:
            message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
            return message
        except asyncio.TimeoutError:
            return None
    
    async def update_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update agent memory with new information."""
        if self.memory:
            await self.memory.add_memory(
                content=content,
                metadata={
                    "agent_id": self.agent_id,
                    "timestamp": datetime.now().isoformat(),
                    **(metadata or {})
                }
            )
    
    async def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search agent memory for relevant information."""
        if self.memory:
            return await self.memory.search(query, limit=limit)
        return []
    
    def update_context(self, key: str, value: Any) -> None:
        """Update the agent's context state."""
        self.state.context[key] = value
        self.state.last_update = datetime.now()
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a value from the agent's context."""
        return self.state.context.get(key, default)
    
    async def compress_context(self, content: str) -> str:
        """Compress context content to fit within token limits."""
        if self.context_manager:
            return await self.context_manager.compress_content(content)
        return content
    
    async def evaluate_performance(self, task: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Evaluate the agent's performance on a completed task."""
        if self.evaluator:
            score = await self.evaluator.evaluate_agent_performance(
                agent_id=self.agent_id,
                task=task,
                result=result
            )
            self.metrics["last_evaluation_score"] = score
            return score
        return 0.0
    
    def update_metrics(self, response_time: float, success: bool) -> None:
        """Update agent performance metrics."""
        self.metrics["tasks_completed"] += 1
        
        # Update average response time
        current_avg = self.metrics["avg_response_time"]
        task_count = self.metrics["tasks_completed"]
        self.metrics["avg_response_time"] = (current_avg * (task_count - 1) + response_time) / task_count
        
        # Update success rate
        if task_count == 1:
            self.metrics["success_rate"] = 1.0 if success else 0.0
        else:
            current_successes = self.metrics["success_rate"] * (task_count - 1)
            new_successes = current_successes + (1 if success else 0)
            self.metrics["success_rate"] = new_successes / task_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.metrics,
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "state": {
                "current_task": self.state.current_task,
                "context_size": len(self.state.context),
                "memory_items": len(self.state.memory_context),
                "last_update": self.state.last_update.isoformat()
            }
        }
    
    async def self_improve(self) -> None:
        """Trigger self-improvement based on recent performance."""
        if not self.evaluator:
            return
            
        # Analyze recent performance
        improvement_suggestions = await self.evaluator.get_improvement_suggestions(
            agent_id=self.agent_id,
            metrics=self.metrics
        )
        
        if improvement_suggestions:
            self.logger.info(f"Self-improvement suggestions: {improvement_suggestions}")
            
            # Apply improvements (in a real implementation, this might involve
            # DSPy optimization, prompt updates, or parameter tuning)
            await self._apply_improvements(improvement_suggestions)
    
    async def _apply_improvements(self, suggestions: List[str]) -> None:
        """Apply self-improvement suggestions."""
        # This is a placeholder for actual improvement implementation
        for suggestion in suggestions:
            self.logger.info(f"Applying improvement: {suggestion}")
            # TODO: Implement actual improvement logic
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, tasks_completed={self.metrics['tasks_completed']})"