from typing import Optional
from .base import LLMProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider

def create_llm_provider(
    provider: str, 
    api_key: str, 
    model: Optional[str] = None, 
    temperature: float = 0.7, 
    max_tokens: int = 2048
) -> LLMProvider:
    """
    Factory method to create an LLM provider based on the specified provider.

    Args:
        provider: The name of the LLM provider (e.g., 'openai', 'anthropic', 'gemini')
        api_key: The API key for the specified provider
        model: Optional model name. If not provided, a default model will be used.
        temperature: Controls randomness of the output (default: 0.7)
        max_tokens: Maximum number of tokens to generate (default: 2048)

    Returns:
        An instance of the specified LLM provider

    Raises:
        ValueError: If an unsupported provider is specified
    """
    provider = provider.lower()

    # Default model selection if not provided
    if not model:
        model = {
            'openai': 'gpt-4o',
            'anthropic': 'claude-3-sonnet-20240229',
            'gemini': 'gemini-1.5-flash'
        }.get(provider, '')

    if provider == 'openai':
        return OpenAIProvider(api_key, model, temperature, max_tokens)
    elif provider == 'anthropic':
        return AnthropicProvider(api_key, model, temperature, max_tokens)
    elif provider == 'gemini':
        return GeminiProvider(api_key, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

__all__ = ['create_llm_provider', 'LLMProvider']