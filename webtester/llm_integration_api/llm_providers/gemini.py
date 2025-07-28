from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from .base import LLMProvider

class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider implementation."""

    def __init__(self, api_key: str, model: str, temperature: float, max_tokens: int):
        """Initializes the Gemini provider."""
        super().__init__(api_key, model, temperature, max_tokens)
        # Initialize the LangChain ChatGoogleGenerativeAI client
        self.client = ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens # Note: Langchain might handle max_tokens differently for Gemini, check docs if needed
        )

    def generate(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Generates text using the configured Gemini model.

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
            # Log the error with more context and raise a custom exception
            import logging
            logging.error(f"Gemini API Error: {e}", exc_info=True)
            # Raise a more specific custom exception with additional context
            raise RuntimeError(f"Failed to generate content with Gemini: {str(e)}") from e