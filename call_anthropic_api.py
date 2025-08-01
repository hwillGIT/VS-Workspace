import os
import requests
import json

# Function to load environment variables from a .env file
def load_env(env_path):
    env_vars = {}
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    env_vars[key] = value.strip('\'\"')
    except FileNotFoundError:
        pass # .env file might not exist, which is fine if env vars are set elsewhere
    return env_vars

# Load environment variables from the .env file
env_vars = load_env('D:\\VS Workspace\\.env')
ANTHROPIC_API_KEY = env_vars.get("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    print("Error: ANTHROPIC_API_KEY not found in .env file or environment variables.")
    exit(1)

API_ENDPOINT = "https://api.anthropic.com/v1/messages"

# Message from SELF_DEBUGGER.md
message_content = """
Problem Description: The `SelfReflectingAgent`'s `HybridRAG` component continues to fail initialization, preventing the `cli.py info` command from completing. While `sentence-transformers` is installed and its embedding model (`sentence-transformers/all-MiniLM-L6-v2`) loads successfully when tested in isolation, the `HybridRAG` error persists. This suggests the issue is not with `sentence-transformers` itself, but rather how it's integrated or a deeper dependency problem within the RAG system.

Error Message & Stack Trace (from previous `cli.py info` run):
```
ClaudeCode.self_reflecting_agent.rag.hybrid_rag - ERROR - Failed
...
Traceback (most recent call last):
  File "D:\VS Workspace\ClaudeCode\self_reflecting_agent\cli.py", line 390, in main
    asyncio.run(handle_info_command(args))
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.10_3.10.3056.0_x64__qbz5n2kfra8p0\lib\asyncio\runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.10_3.10.3056.0_x64__qbz5n2kfra8p0\lib\asyncio\base_events.py", line 649, in run_until_complete
    return future.result()
  File "D:\VS Workspace\ClaudeCode\self_reflecting_agent\main.py", line 79, in initialize
    await self._initialize_core_components()
  File "D:\VS Workspace\ClaudeCode\self_reflecting_agent\main.py", line 120, in _initialize_core_components
    await self.rag_system.initialize()
  File "D:\VS Workspace\ClaudeCode\self_reflecting_agent\rag\hybrid_rag.py", line 70, in initialize
    results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
  File "D:\VS Workspace\ClaudeCode\self_reflecting_agent\rag\hybrid_rag.py", line 74, in initialize
    self.logger.error(f"Failed to initialize {component_names[i]}: {result}")
```

Relevant Code Snippets:
*   `D:\VS Workspace\ClaudeCode\self_reflecting_agent\rag\hybrid_rag.py` (especially the `initialize` method, which calls `vector_store.initialize()` and `bm25_search.initialize()`)
*   `D:\VS Workspace\ClaudeCode\self_reflecting_agent\rag\vector_store.py` (especially `_initialize_embedding_model` and `_initialize_backend`)
*   `D:\VS Workspace\ClaudeCode\self_reflecting_agent\rag\bm25_search.py` (for completeness, as it's also initialized in `HybridRAG`)

Contextual Information:
*   Current working directory: `D:\VS Workspace`
*   Operating system: `win32`
*   `sentence-transformers` is confirmed installed and working in isolation.
*   The `UnicodeEncodeError` was observed previously, but the core RAG initialization failure is the primary concern.

Specific Questions for Anthropic:
"I have confirmed that `sentence-transformers` is installed and its embedding model loads successfully in isolation. However, the `HybridRAG` component of the `SelfReflectingAgent` still fails to initialize with the error `ClaudeCode.self_reflecting_agent.rag.hybrid_rag - ERROR - Failed`. Given that the embedding model itself is not the direct cause, what are the next steps to debug this RAG initialization failure? Could there be other missing dependencies for `faiss` or `chromadb` (as used by `VectorStore`), or for `BM25Search`? How can I systematically identify which specific sub-component (`vector_store` or `bm25_search`) is causing the `HybridRAG` initialization to fail, and what are common reasons for their failure in a Windows environment?"
"""


headers = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

data = {
    "model": "claude-3-opus-20240229",
    "max_tokens": 4096,
    "temperature": 0.7,
    "messages": [
        {
            "role": "user",
            "content": message_content
        }
    ]
}

try:
    response = requests.post(API_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()  # Raise an exception for HTTP errors
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
    if response is not None:
        print(f"Response content: {response.text}")
    exit(1)
