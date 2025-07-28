from typing import Optional
from langchain_anthropic import ChatAnthropic
from .base import LLMProvider

class AnthropicProvider(LLMProvider):
    """Anthropic LLM provider implementation."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        """Initializes the Anthropic provider."""
        super().__init__(api_key, model, temperature, max_tokens)
        # Initialize the LangChain ChatAnthropic client
        self.client = ChatAnthropic(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Generates text using the configured Anthropic model.

        Args:
            prompt: The main user prompt.
            system_message: An optional system message to provide context.

        Returns:
            The generated text content.

        Raises:
            Exception: If any error occurs during the API call.
        """
        try:
            # Prepare the messages list
            messages = self._create_messages(prompt, system_message)
            # Invoke the LangChain client
            response = self.client.invoke(messages)
            # Return the content of the response
            return response.content
        except Exception as e:
            # Log the error and re-raise a more specific exception or handle it
            print(f"Error generating content with Anthropic: {e}")
            # Re-raising the exception to be handled by the main API endpoint
            raise e