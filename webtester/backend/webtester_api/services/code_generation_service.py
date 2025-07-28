class CodeGenerationService:
    def generate_code(self, prompt):
        # Implementation using OpenRouter API
        import requests
        import json

        api_url = "https://openrouter.ai/api/v1/generate"
        api_key = "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"  # Replace with your actual API key

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.7
        }

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()
            # Adjust this based on OpenRouter's actual response format
            return result.get("generated_text", "No code generated.")
        except Exception as e:
            return f"Error generating code: {str(e)}"

    def generate_text(self, prompt):
        # Enhanced implementation for text generation
        try:
            from transformers import pipeline
            generator = pipeline('text-generation', model='gpt2')
            generated_text = generator(prompt, max_length=50)[0]['generated_text']
            return generated_text
        except Exception as e:
            return f"Error generating text: {str(e)}"