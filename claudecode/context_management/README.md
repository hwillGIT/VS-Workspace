# ChromaDB Context Management System

A hierarchical context management system for AI agents using ChromaDB as an embedded vector database.

## Features

✨ **4-Level Context Hierarchy**
- **Immediate**: Current conversation context
- **Session**: Current work session memory
- **Project**: Project-specific knowledge
- **Global**: Universal patterns and best practices

🔍 **Semantic Search**
- Vector similarity search
- Metadata filtering
- Cross-level querying
- Automatic embeddings with Sentence Transformers

💾 **Persistent Storage**
- Embedded database (no server required)
- Survives application restarts
- Efficient disk-based storage
- Automatic persistence

🏗️ **Project Isolation**
- Separate contexts per project
- Easy project switching
- Shared global knowledge

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from chroma_context_manager import ChromaContextManager, ContextLevel

# Initialize the context manager
cm = ChromaContextManager(persist_directory="./my_context_db")

# Set current project
cm.set_project("trading_system")

# Add context at different levels
cm.add_context(
    "User wants to build a real-time trading system",
    ContextLevel.IMMEDIATE
)

cm.add_context(
    "Using WebSocket for market data streaming",
    ContextLevel.SESSION,
    metadata={"decision_type": "technical"}
)

cm.add_context(
    "All monetary values must use Decimal type",
    ContextLevel.PROJECT,
    metadata={"rule_type": "data_handling"}
)

# Search across all contexts
results = cm.search_context("data streaming implementation", n_results=5)

# Get all context for current session
session_context = cm.get_session_context()
```

## API Reference

### ChromaContextManager

#### `__init__(persist_directory, embedding_model)`
Initialize the context manager with persistent storage.

#### `add_context(content, level, metadata, project)`
Add a new context entry.

#### `search_context(query, level, n_results, filters)`
Search for relevant context using semantic similarity.

#### `get_session_context(session_id, max_items)`
Retrieve all context for a specific session.

#### `promote_context(doc_id, from_level, to_level)`
Promote valuable context to a higher level.

#### `clear_immediate_context()`
Clear conversation-level context.

#### `set_project(project_name)`
Switch to a different project context.

#### `new_session()`
Start a new work session.

## Use Cases

### 1. **AI Code Assistant**
```python
# Track conversation flow
cm.add_context(user_request, ContextLevel.IMMEDIATE)
cm.add_context(assistant_response, ContextLevel.IMMEDIATE)

# Remember technical decisions
cm.add_context(
    "Chose PostgreSQL for ACID compliance", 
    ContextLevel.SESSION,
    metadata={"decision": "database"}
)
```

### 2. **Learning from Interactions**
```python
# Promote successful patterns
if task_successful:
    cm.promote_context(
        doc_id, 
        from_level=ContextLevel.SESSION,
        to_level=ContextLevel.PROJECT
    )
```

### 3. **Multi-Project Support**
```python
# Switch between projects
cm.set_project("web_app")
# All subsequent operations scoped to web_app

cm.set_project("ml_pipeline")  
# Now working in ml_pipeline context
```

### 4. **Intelligent Retrieval**
```python
# Get context with metadata filtering
security_context = cm.search_context(
    "authentication",
    filters={"category": "security"}
)

# Get recent decisions
recent_decisions = cm.search_context(
    "",  # Empty query returns all
    filters={"decision_type": "technical"},
    n_results=10
)
```

## Architecture

```
┌─────────────────────────────────┐
│        Global Context           │  Universal patterns
├─────────────────────────────────┤
│       Project Context           │  Project-specific
├─────────────────────────────────┤
│       Session Context           │  Work session
├─────────────────────────────────┤
│      Immediate Context          │  Current conversation
└─────────────────────────────────┘
         ↓                ↑
    ChromaDB          Embeddings
```

## Performance

- **Fast**: In-memory indices with disk persistence
- **Scalable**: Handles millions of embeddings
- **Efficient**: Only loads needed data
- **Concurrent**: Thread-safe operations

## Best Practices

1. **Clear immediate context** between unrelated conversations
2. **Promote valuable insights** to higher levels
3. **Use metadata** for better filtering
4. **Set meaningful project names** for isolation
5. **Regular cleanup** of old sessions

## Integration with AI Agents

```python
class AIAssistant:
    def __init__(self):
        self.context_manager = ChromaContextManager()
    
    def process_request(self, user_input):
        # Add to immediate context
        self.context_manager.add_context(
            user_input, 
            ContextLevel.IMMEDIATE
        )
        
        # Get relevant context
        context = self.context_manager.search_context(
            user_input,
            n_results=10
        )
        
        # Use context for response generation
        response = self.generate_response(user_input, context)
        
        # Store response
        self.context_manager.add_context(
            response,
            ContextLevel.IMMEDIATE,
            metadata={"type": "assistant_response"}
        )
        
        return response
```

## Troubleshooting

### "Collection not found"
- Ensure ChromaDB is properly initialized
- Check persist_directory permissions

### "Embedding dimension mismatch"
- Don't change embedding models after initialization
- Clear database if model change needed

### Performance Issues
- Limit immediate context size
- Use metadata filters to narrow searches
- Consider batch operations for bulk adds