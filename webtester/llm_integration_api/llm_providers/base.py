from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage

class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """Generate text from the LLM"""
        pass

    def _create_messages(self, prompt: str, system_message: Optional[str] = None):
        """Create messages array with optional system message"""
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        return messages