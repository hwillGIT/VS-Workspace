# Intelligent Model Routing System

## Overview

The Self-Reflecting Agent now includes an intelligent model routing system that automatically selects the optimal AI model for each task based on:

- **Task Type**: Different models excel at different types of tasks
- **Model Availability**: Automatic fallback when models hit rate limits or are unavailable
- **Performance History**: Learning from past successes and failures
- **Cost Optimization**: Balance between quality and cost
- **Context Requirements**: Match model context windows to task needs
- **Specialized Capabilities**: Route to models optimized for specific tasks

## Key Features

🎯 **Intelligent Routing Logic**
- Orchestration tasks → Claude 3.5 Sonnet (best reasoning)
- Debugging tasks → Gemini 2.5 Pro (optimized for debugging)
- Code generation → Claude 3.5 Sonnet or GPT-4o (excellent coding)
- Documentation → Claude Haiku or GPT-4o-mini (cost-effective)

🔄 **Context Preservation**
- Maintains conversation history across model switches
- Intelligent context compression for smaller context windows
- Seamless transitions between models

🔍 **RAG & Semantic Search**
- Enhances responses with relevant project context
- Hybrid search combining semantic and keyword matching
- Automatic indexing of project files and documentation

📊 **Performance Monitoring**
- Tracks success rates, latency, and costs per model
- Automatic optimization based on performance history
- Detailed analytics and reporting

## Quick Start

### 1. Setup Environment Variables

Copy `.env.example` to `.env` and configure your API keys:

```bash
# Primary models (configure the ones you have access to)
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Enable intelligent routing
ENABLE_MODEL_ROUTING=true
ENABLE_RAG=true
ENABLE_SEMANTIC_SEARCH=true
```

### 2. Basic Usage

```python
from routed_agent import create_routed_agent

# Create agent with intelligent routing
agent = await create_routed_agent(
    project_path="./my_project",
    enable_rag=True,
    enable_semantic_search=True
)

# The system automatically routes to optimal models
result = await agent.execute_task(
    task_description="Debug this connection error",
    task_type=TaskType.DEBUGGING
)

print(f"Used {result['model_used']}: {result['routing_reasoning']}")
```

### 3. Specialized Interfaces

```python
# Code generation (routes to best coding models)
code_result = await agent.generate_code(
    request="Create a thread-safe cache implementation",
    language="python"
)

# Debugging (routes to Gemini 2.5 Pro - optimized for debugging)
debug_result = await agent.debug_code(
    code_or_error="Getting KeyError when accessing user data"
)

# Architecture (routes to Claude Sonnet - best system thinking)
arch_result = await agent.plan_architecture(
    requirements="Design scalable microservices for e-commerce"
)

# Search and answer with RAG enhancement
answer = await agent.search_and_answer(
    question="How does the authentication system work?"
)
```

## Supported Models

### Anthropic Claude Models
- **claude-3-5-sonnet-20241022**: Premium model, excellent for complex reasoning and code
- **claude-3-5-haiku-20241022**: Fast and cost-effective, good for simple tasks
- **claude-3-haiku-20240307**: Most cost-effective option for documentation and conversation

### OpenAI GPT Models
- **gpt-4o**: Strong general capability, good for coding and analysis
- **gpt-4o-mini**: Cost-effective, suitable for simple tasks
- **gpt-4-turbo**: High-quality but expensive, used sparingly

### Google Gemini Models
- **gemini-2.0-flash-exp**: **Optimized for debugging tasks**, large context window
- **gemini-1.5-pro**: Very large context (2M tokens), good for large codebases
- **gemini-1.5-flash**: Fast and cost-effective

## Routing Logic

### Task Type → Model Preferences

| Task Type | Primary Model | Secondary | Use Case |
|-----------|---------------|-----------|----------|
| **Orchestration** | Claude 3.5 Sonnet | GPT-4o | High-level planning, coordination |
| **Debugging** | **Gemini 2.5 Pro** | Claude Sonnet | Error analysis, troubleshooting |
| **Code Generation** | Claude 3.5 Sonnet | GPT-4o | Writing new code, algorithms |
| **Code Review** | Claude 3.5 Sonnet | GPT-4o | Analyzing code quality, suggestions |
| **Architecture** | Claude 3.5 Sonnet | GPT-4 Turbo | System design, scalability planning |
| **Documentation** | Claude Haiku | GPT-4o Mini | Writing docs, explanations |
| **Testing** | Claude 3.5 Sonnet | GPT-4o | Creating tests, test strategies |
| **Conversation** | Claude Haiku | GPT-4o Mini | General chat, simple questions |

### Intelligent Fallback

When a model is unavailable (rate limits, outages, quota exceeded):

1. **Primary Model** fails → Try **Secondary Model**
2. **Secondary Model** fails → Try **Tertiary Model**
3. **All preferred models** fail → Use **Emergency Fallback**
   - Claude Haiku (most reliable)
   - GPT-4o Mini (backup)
   - Gemini Flash (final fallback)

### Context Management

**Context Preservation Across Models:**

```python
# Start conversation with Claude Sonnet
response1 = await agent.chat("I'm building a REST API with authentication")

# Switch to debugging model while preserving context
response2 = await agent.debug_code("Getting 'Invalid token' error")
# → Routes to Gemini 2.5 Pro but maintains full conversation context

# Switch back to code generation
response3 = await agent.generate_code("Add password reset functionality")
# → Routes to Claude Sonnet with full context preserved
```

**Context Compression:**
When switching to models with smaller context windows, the system:
- Keeps recent conversation turns
- Summarizes older conversation history
- Preserves important context and system instructions
- Maintains conversation continuity

## Performance Optimization

### Automatic Learning

The system tracks performance metrics for each model:
- **Success Rate**: How often the model completes tasks successfully
- **Average Latency**: Response time for different task types
- **Cost Per Request**: Actual usage costs
- **Failure Patterns**: Common failure reasons and times

### Cost Management

Set daily spending limits:
```bash
DAILY_COST_LIMIT_ANTHROPIC=20.00
DAILY_COST_LIMIT_OPENAI=15.00
DAILY_COST_LIMIT_GOOGLE=10.00
```

Enable cost-sensitive routing:
```python
result = await agent.execute_task(
    task_description="Write documentation for this function",
    cost_sensitive=True  # Will prefer cheaper models
)
```

### Latency Optimization

Enable latency-sensitive routing for time-critical tasks:
```python
result = await agent.debug_code(
    code_or_error="Production error needs immediate attention",
    latency_sensitive=True  # Will prefer faster models
)
```

## RAG and Semantic Search

### Automatic Project Indexing

The system automatically indexes:
- **Code files**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, etc.
- **Documentation**: `.md`, `.rst`, `.txt` files
- **Configuration**: `README.md`, `CLAUDE.md`, etc.

### Hybrid Search

Combines **semantic similarity** and **keyword matching**:
```python
# Semantic search finds conceptually related content
results = await agent.semantic_search(
    query="authentication implementation",
    search_type="hybrid"  # Best of both approaches
)
```

### Enhanced Responses

RAG automatically enhances responses with relevant context:
```python
# Question gets enhanced with relevant project context
answer = await agent.search_and_answer(
    question="How should I implement user sessions?"
)
# → Finds relevant auth code, security docs, etc.
```

## Advanced Configuration

### Custom Routing Rules

Override default routing in `routing/router_config.yaml`:

```yaml
routing_rules:
  debugging:
    - gemini-2.0-flash-exp    # First choice for debugging
    - claude-3-5-sonnet-20241022
    - gpt-4o
  
  code_generation:
    - claude-3-5-sonnet-20241022  # Best for code quality
    - gpt-4o
    - gemini-2.0-flash-exp
```

### Performance Tuning

```yaml
monitoring:
  performance_weight: 0.4      # How much to weight performance history
  capability_weight: 0.4       # How much to weight model capabilities  
  priority_weight: 0.2         # How much to weight manual priorities
  
  min_success_rate: 0.7        # Minimum success rate to use a model
  max_acceptable_latency_ms: 30000  # Max latency before trying alternatives
```

### Cost Controls

```yaml
cost_management:
  enable_cost_optimization: true
  
  # Daily limits per model (USD)
  daily_limits:
    gpt-4-turbo: 10.0
    claude-3-5-sonnet-20241022: 15.0
  
  # Tasks that should prefer cheaper models
  cost_sensitive_tasks:
    - documentation
    - conversation
    - testing
```

## Monitoring and Analytics

### Runtime Status

```python
# Check router status
status = agent.get_routing_status()
print(f"Models available: {len(status['router_status']['models'])}")

# Check session statistics
session_info = agent.get_session_info("my_session")
print(f"Tasks completed: {session_info['task_count']}")
print(f"Models used: {session_info['models_used']}")
```

### Performance Metrics

```python
# View model performance
for model_name, model_info in status['router_status']['models'].items():
    if 'performance' in model_info:
        perf = model_info['performance']
        print(f"{model_name}: {perf['success_rate']:.1%} success, "
              f"{perf['avg_latency_ms']:.0f}ms avg latency")
```

## Best Practices

### 1. API Key Management
- Set up keys for multiple providers for best fallback coverage
- Use environment variables, never hardcode keys
- Monitor usage and set appropriate rate limits

### 2. Task Type Selection
- Be specific about task types for better routing
- Use `TaskType.DEBUGGING` for error analysis → routes to Gemini 2.5 Pro
- Use `TaskType.ARCHITECTURE` for system design → routes to Claude Sonnet
- Use `TaskType.DOCUMENTATION` for docs → routes to cost-effective models

### 3. Context Management
- Use consistent session IDs for related tasks
- Let the system handle context compression automatically
- Provide clear, specific task descriptions

### 4. Cost Optimization
- Use `cost_sensitive=True` for non-critical tasks
- Set appropriate daily limits
- Monitor usage through the status dashboard

### 5. Performance Monitoring
- Check routing decisions and adjust preferences if needed
- Monitor success rates and latency
- Update model priorities based on performance

## Troubleshooting

### Common Issues

**No models available:**
```bash
# Check API keys are set
echo $ANTHROPIC_API_KEY
echo $OPENAI_API_KEY
echo $GOOGLE_API_KEY

# Enable at least one provider
ENABLE_MODEL_ROUTING=true
```

**High latency:**
```python
# Enable latency-sensitive routing
result = await agent.execute_task(
    task_description="Your task",
    latency_sensitive=True
)
```

**High costs:**
```python
# Enable cost optimization
result = await agent.execute_task(
    task_description="Your task", 
    cost_sensitive=True
)
```

**Context too large:**
The system automatically compresses context when switching to smaller models. Check compression strategy:
```bash
CONTEXT_COMPRESSION_STRATEGY=summarize  # or 'truncate'
MAX_PRESERVED_CONTEXT=50000
```

### Debug Mode

Enable detailed logging:
```bash
LOG_LEVEL=DEBUG
DEVELOPMENT_MODE=true
```

## Migration Guide

### From Basic Agent

Replace:
```python
from main import SelfReflectingAgent
agent = SelfReflectingAgent()
```

With:
```python
from routed_agent import create_routed_agent
agent = await create_routed_agent()
```

### Existing Code Compatibility

The routed agent maintains full compatibility with existing agent methods:
- `execute_task()` - now with intelligent routing
- `chat()` - now with optimal model selection
- `generate_code()` - routes to best coding models
- `debug_code()` - routes to debugging-optimized models

## Example Use Cases

### 1. Full-Stack Development
```python
# Architecture planning → Claude Sonnet
arch = await agent.plan_architecture("Design a real-time chat app")

# Code generation → Claude Sonnet
code = await agent.generate_code("Implement WebSocket handler")

# Debugging → Gemini 2.5 Pro
debug = await agent.debug_code("WebSocket connection fails after 30s")

# Documentation → Claude Haiku (cost-effective)
docs = await agent.execute_task("Document the WebSocket API", 
                                task_type=TaskType.DOCUMENTATION)
```

### 2. Code Review Workflow
```python
# Review with premium model
review = await agent.review_code(code_content)

# Generate tests with quality model  
tests = await agent.execute_task("Create unit tests for this code",
                                task_type=TaskType.TESTING)

# Debug test failures with specialized model
debug = await agent.debug_code("Test fails with 'Mock not found'")
```

### 3. Research and Analysis
```python
# Research with large context model
research = await agent.execute_task("Research GraphQL vs REST trade-offs",
                                   task_type=TaskType.RESEARCH)

# Analysis with reasoning model
analysis = await agent.execute_task("Analyze our API performance bottlenecks",
                                   task_type=TaskType.ANALYSIS)
```

The intelligent routing system ensures you always get the best model for each task while maintaining cost efficiency and context continuity. 🚀