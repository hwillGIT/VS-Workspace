"""
OpenRouter Client Integration

Provides a unified interface for interacting with OpenRouter's diverse model ecosystem.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, AsyncGenerator
import aiohttp
from dataclasses import dataclass


@dataclass
class OpenRouterResponse:
    """Response from OpenRouter API."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Dict[str, Any]


@dataclass
class OpenRouterStreamChunk:
    """Streaming response chunk from OpenRouter."""
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class OpenRouterClient:
    """
    Client for interacting with OpenRouter API.
    
    Provides both streaming and non-streaming interfaces for diverse model access.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        app_name: str = "Self-Reflecting Agent",
        timeout: int = 300
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.app_name = app_name
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_name,
            "HTTP-Referer": "https://github.com/your-org/self-reflecting-agent",
        }
    
    async def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> OpenRouterResponse:
        """
        Send a completion request to OpenRouter.
        
        Args:
            model: Model identifier (e.g., 'anthropic/claude-3-5-sonnet')
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stop: List of stop sequences
            **kwargs: Additional model-specific parameters
            
        Returns:
            OpenRouterResponse with generated content and metadata
        """
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        
        # Add any additional parameters
        payload.update(kwargs)
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_response(result)
                    
                    elif response.status == 429:
                        error_text = await response.text()
                        raise Exception(f"Rate limited by OpenRouter: {error_text}")
                    
                    elif response.status == 401:
                        raise Exception("Invalid OpenRouter API key")
                    
                    elif response.status == 400:
                        error_text = await response.text()
                        raise Exception(f"Bad request to OpenRouter: {error_text}")
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenRouter API error {response.status}: {error_text}")
            
            except asyncio.TimeoutError:
                raise Exception(f"OpenRouter request timed out after {self.timeout}s")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error calling OpenRouter: {e}")
    
    async def stream_complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> AsyncGenerator[OpenRouterStreamChunk, None]:
        """
        Stream a completion request from OpenRouter.
        
        Args:
            Same as complete() method
            
        Yields:
            OpenRouterStreamChunk objects with incremental content
        """
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": True
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop
        
        payload.update(kwargs)
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"OpenRouter streaming error {response.status}: {error_text}")
                    
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if line.startswith('data: '):
                            data = line[6:]
                            
                            if data == '[DONE]':
                                break
                            
                            try:
                                chunk_data = json.loads(data)
                                chunk = self._parse_stream_chunk(chunk_data)
                                if chunk:
                                    yield chunk
                            except json.JSONDecodeError:
                                self.logger.warning(f"Could not parse streaming chunk: {data}")
                                continue
            
            except asyncio.TimeoutError:
                raise Exception(f"OpenRouter streaming request timed out after {self.timeout}s")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error during OpenRouter streaming: {e}")
    
    def _parse_response(self, response_data: Dict[str, Any]) -> OpenRouterResponse:
        """Parse a complete response from OpenRouter."""
        try:
            choice = response_data["choices"][0]
            message = choice["message"]
            
            return OpenRouterResponse(
                content=message["content"],
                model=response_data["model"],
                usage=response_data.get("usage", {}),
                finish_reason=choice["finish_reason"],
                raw_response=response_data
            )
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected OpenRouter response format: {e}")
    
    def _parse_stream_chunk(self, chunk_data: Dict[str, Any]) -> Optional[OpenRouterStreamChunk]:
        """Parse a streaming chunk from OpenRouter."""
        try:
            if "choices" not in chunk_data or not chunk_data["choices"]:
                return None
            
            choice = chunk_data["choices"][0]
            delta = choice.get("delta", {})
            
            # Skip chunks without content
            if "content" not in delta:
                return None
            
            return OpenRouterStreamChunk(
                content=delta["content"],
                model=chunk_data["model"],
                finish_reason=choice.get("finish_reason"),
                usage=chunk_data.get("usage")
            )
        except (KeyError, IndexError):
            self.logger.warning(f"Could not parse streaming chunk: {chunk_data}")
            return None
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from OpenRouter.
        
        Returns:
            List of model information dictionaries
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        return result.get("data", [])
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to get models: {error_text}")
            
            except asyncio.TimeoutError:
                raise Exception("Timeout getting models from OpenRouter")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error getting models: {e}")
    
    async def get_credits(self) -> Dict[str, Any]:
        """
        Get current credit balance from OpenRouter.
        
        Returns:
            Dictionary with credit information
        """
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.base_url}/auth/key",
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        raise Exception(f"Failed to get credits: {error_text}")
            
            except asyncio.TimeoutError:
                raise Exception("Timeout getting credits from OpenRouter")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error getting credits: {e}")


# Convenience functions for common model access patterns

async def create_openrouter_client() -> OpenRouterClient:
    """Create an OpenRouter client with environment configuration."""
    return OpenRouterClient()


async def quick_complete(
    model: str,
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> str:
    """Quick completion using OpenRouter."""
    client = await create_openrouter_client()
    
    messages = [{"role": "user", "content": prompt}]
    
    response = await client.complete(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response.content


async def get_available_models() -> List[str]:
    """Get list of available model names from OpenRouter."""
    client = await create_openrouter_client()
    models = await client.get_models()
    
    return [model["id"] for model in models if model.get("id")]


# Example usage and testing
async def test_openrouter_client():
    """Test the OpenRouter client functionality."""
    try:
        client = await create_openrouter_client()
        
        # Test getting models
        print("Available models:")
        models = await client.get_models()
        for model in models[:5]:  # Show first 5
            print(f"  - {model.get('id', 'Unknown')}")
        
        # Test completion
        print("\nTesting completion...")
        response = await client.complete(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "Hello! How are you?"}],
            max_tokens=50
        )
        print(f"Response: {response.content}")
        print(f"Usage: {response.usage}")
        
        # Test streaming
        print("\nTesting streaming...")
        full_response = ""
        async for chunk in client.stream_complete(
            model="anthropic/claude-3-haiku",
            messages=[{"role": "user", "content": "Count to 5."}],
            max_tokens=100
        ):
            full_response += chunk.content
            print(chunk.content, end="", flush=True)
        print(f"\nFull streamed response: {full_response}")
        
        print("\nOpenRouter client test completed successfully!")
        
    except Exception as e:
        print(f"OpenRouter client test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_openrouter_client())