import os
import time
from functools import wraps
from typing import Optional

import redis
import prometheus_client
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# Import LLM providers
from .llm_providers import create_llm_provider
from langchain_core.messages import HumanMessage

# Prometheus Metrics
REQUESTS = prometheus_client.Counter('llm_requests_total', 'Total LLM API Requests')
REQUEST_LATENCY = prometheus_client.Histogram('llm_request_latency_seconds', 'LLM Request Latency')
ERROR_COUNTER = prometheus_client.Counter('llm_errors_total', 'Total LLM API Errors')

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Retrieve API keys and configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 10))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))  # seconds

# Redis rate limiting
redis_client = redis.from_url(REDIS_URL)

def rate_limit(func):
    """
    Decorator to implement rate limiting using Redis.
    Limits the number of requests per IP address within a specified time window.
    """
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        client_ip = request.client.host
        key = f"rate_limit:{client_ip}"
        
        # Increment request count and get current count
        current_count = redis_client.incr(key)
        
        # Set expiration if this is the first request
        if current_count == 1:
            redis_client.expire(key, RATE_LIMIT_WINDOW)
        
        # Check if request count exceeds limit
        if current_count > RATE_LIMIT_REQUESTS:
            raise HTTPException(
                status_code=429, 
                detail=f"Too many requests. Limit is {RATE_LIMIT_REQUESTS} per {RATE_LIMIT_WINDOW} seconds."
            )
        
        return await func(request, *args, **kwargs)
    return wrapper

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Webtester LLM Integration API",
    description="API for interacting with various LLMs for code/test generation.",
    version="0.2.0",  # Updated version to reflect new features
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models (Request/Response Schemas) ---
class GenerationRequest(BaseModel):
    """Request model for code/test generation."""
    model_provider: str  # e.g., "openai", "anthropic", "gemini"
    prompt: str
    model_name: str = ""  # Optional model name override (e.g., "gpt-4", "gemini-1.5-pro")
    temperature: float = 0.7  # Controls randomness: lower = more deterministic
    max_tokens: int = 2048  # Maximum number of tokens to generate
    system_message: str = ""  # Optional system message for context/instructions

class GenerationResponse(BaseModel):
    """Response model for code/test generation."""
    generated_text: str
    model_used: str
    request_id: str  # Unique identifier for tracking requests

# --- API Endpoints ---
@app.get("/", summary="Health Check")
async def read_root():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "LLM Integration API is running"}

@app.post("/generate", response_model=GenerationResponse, summary="Generate Code or Tests")
@rate_limit
async def generate_content(request: Request, generation_request: GenerationRequest):
    """
    Endpoint to generate content (code or tests) using a specified LLM provider.
    Includes rate limiting and Prometheus metrics tracking.
    """
    start_time = time.time()
    request_id = str(hash(time.time()))  # Simple unique request ID
    
    REQUESTS.inc()  # Increment request counter
    
    # Determine the appropriate API key
    api_key = {
        'openai': OPENAI_API_KEY,
        'anthropic': ANTHROPIC_API_KEY,
        'gemini': GEMINI_API_KEY
    }.get(generation_request.model_provider)

    # Check if API key is configured
    if not api_key:
        ERROR_COUNTER.inc()
        raise HTTPException(
            status_code=500, 
            detail=f"{generation_request.model_provider.capitalize()} API key not configured"
        )

    try:
        # Create the LLM provider using the factory method
        llm_provider = create_llm_provider(
            provider=generation_request.model_provider,
            api_key=api_key,
            model=generation_request.model_name,
            temperature=generation_request.temperature,
            max_tokens=generation_request.max_tokens
        )

        # Generate content using the provider
        generated_output = llm_provider.generate(
            prompt=generation_request.prompt, 
            system_message=generation_request.system_message
        )
        model_name = llm_provider.model

    except Exception as e:
        ERROR_COUNTER.inc()
        # Log the exception details for debugging
        print(f"Error during LLM generation: {e}")
        # Return a generic error to the client
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during generation with {generation_request.model_provider}: {str(e)}"
        )
    finally:
        # Record request latency
        REQUEST_LATENCY.observe(time.time() - start_time)

    return GenerationResponse(
        generated_text=generated_output, 
        model_used=model_name,
        request_id=request_id
    )

# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Endpoint to expose Prometheus metrics."""
    return prometheus_client.generate_latest()

# --- Uvicorn Runner ---
if __name__ == "__main__":
    # Initialize Prometheus metrics server
    prometheus_client.start_http_server(8000)
    
    # Run the FastAPI app using uvicorn
    # Host '0.0.0.0' makes it accessible on the network
    # Reload=True is useful for development, automatically restarts on code changes
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)