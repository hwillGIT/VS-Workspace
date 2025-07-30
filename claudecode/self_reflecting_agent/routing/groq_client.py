"""
Groq Client Integration

Provides ultra-fast inference through Groq's specialized hardware.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Any, AsyncGenerator
import aiohttp
from dataclasses import dataclass


@dataclass
class GroqResponse:
    """Response from Groq API."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Dict[str, Any]


@dataclass  
class GroqStreamChunk:
    """Streaming response chunk from Groq."""
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class GroqClient:
    """
    Client for interacting with Groq API.
    
    Provides ultra-fast inference with streaming and non-streaming interfaces.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: int = 60
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = base_url
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
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
        seed: Optional[int] = None,
        **kwargs
    ) -> GroqResponse:
        """
        Send a completion request to Groq.
        
        Args:
            model: Model identifier (e.g., 'llama-3.3-70b-versatile')
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0-2)
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stop: List of stop sequences
            seed: Random seed for reproducibility
            **kwargs: Additional model-specific parameters
            
        Returns:
            GroqResponse with generated content and metadata
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
        if seed is not None:
            payload["seed"] = seed
        
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
                        # Extract rate limit headers if available
                        retry_after = response.headers.get('retry-after')
                        if retry_after:
                            raise Exception(f"Rate limited by Groq. Retry after {retry_after}s: {error_text}")
                        else:
                            raise Exception(f"Rate limited by Groq: {error_text}")
                    
                    elif response.status == 401:
                        raise Exception("Invalid Groq API key")
                    
                    elif response.status == 400:
                        error_text = await response.text()
                        raise Exception(f"Bad request to Groq: {error_text}")
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"Groq API error {response.status}: {error_text}")
            
            except asyncio.TimeoutError:
                raise Exception(f"Groq request timed out after {self.timeout}s")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error calling Groq: {e}")
    
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
        seed: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[GroqStreamChunk, None]:
        """
        Stream a completion request from Groq.
        
        Args:
            Same as complete() method
            
        Yields:
            GroqStreamChunk objects with incremental content
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
        if seed is not None:
            payload["seed"] = seed
        
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
                        raise Exception(f"Groq streaming error {response.status}: {error_text}")
                    
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
                raise Exception(f"Groq streaming request timed out after {self.timeout}s")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error during Groq streaming: {e}")
    
    def _parse_response(self, response_data: Dict[str, Any]) -> GroqResponse:
        """Parse a complete response from Groq."""
        try:
            choice = response_data["choices"][0]
            message = choice["message"]
            
            return GroqResponse(
                content=message["content"],
                model=response_data["model"],
                usage=response_data.get("usage", {}),
                finish_reason=choice["finish_reason"],
                raw_response=response_data
            )
        except (KeyError, IndexError) as e:
            raise Exception(f"Unexpected Groq response format: {e}")
    
    def _parse_stream_chunk(self, chunk_data: Dict[str, Any]) -> Optional[GroqStreamChunk]:
        """Parse a streaming chunk from Groq."""
        try:
            if "choices" not in chunk_data or not chunk_data["choices"]:
                return None
            
            choice = chunk_data["choices"][0]
            delta = choice.get("delta", {})
            
            # Skip chunks without content
            if "content" not in delta:
                return None
            
            return GroqStreamChunk(
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
        Get list of available models from Groq.
        
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
                raise Exception("Timeout getting models from Groq")
            except aiohttp.ClientError as e:
                raise Exception(f"Network error getting models: {e}")


# Convenience functions for common model access patterns

async def create_groq_client() -> GroqClient:
    """Create a Groq client with environment configuration."""
    return GroqClient()


async def quick_complete(
    model: str,
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.7
) -> str:
    """Quick completion using Groq's ultra-fast inference."""
    client = await create_groq_client()
    
    messages = [{"role": "user", "content": prompt}]
    
    response = await client.complete(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response.content


async def get_available_models() -> List[str]:
    """Get list of available model names from Groq."""
    client = await create_groq_client()
    models = await client.get_models()
    
    return [model["id"] for model in models if model.get("id")]


# Example usage and testing
async def test_groq_client():
    """Test the Groq client functionality."""
    try:
        client = await create_groq_client()
        
        # Test getting models
        print("Available Groq models:")
        models = await client.get_models()
        for model in models:
            print(f"  - {model.get('id', 'Unknown')}: {model.get('owned_by', 'Unknown')}")
        
        # Test completion with ultra-fast model
        print("\nTesting ultra-fast completion...")
        response = await client.complete(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "What makes Groq special for AI inference?"}],
            max_tokens=100,
            temperature=0.3
        )
        print(f"Response: {response.content}")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        
        # Test streaming
        print("\nTesting ultra-fast streaming...")
        full_response = ""
        start_time = asyncio.get_event_loop().time()
        
        async for chunk in client.stream_complete(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Count to 10 quickly."}],
            max_tokens=100,
            temperature=0.1
        ):
            full_response += chunk.content
            print(chunk.content, end="", flush=True)
        
        end_time = asyncio.get_event_loop().time()
        print(f"\n\nStreaming completed in {(end_time - start_time):.2f} seconds")
        print(f"Full response length: {len(full_response)} characters")
        
        print("\n✅ Groq client test completed successfully!")
        
    except Exception as e:
        print(f"❌ Groq client test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_groq_client())