const neo4j = require('neo4j-driver');

async function createGeminiCipherTutorial() {
    console.log('Creating Cipher Tutorial for Gemini...\n');
    
    const driver = neo4j.driver(
        'bolt://localhost:7687',
        neo4j.auth.basic('neo4j', 'architecture123')
    );
    
    try {
        const session = driver.session();
        
        // 1. Store Tutorial Overview
        console.log('1. Storing Cipher tutorial overview...');
        await session.run(`
            MERGE (tutorial:Tutorial {name: "Cipher_Memory_System"})
            SET tutorial.audience = "Gemini AI",
                tutorial.purpose = "Learn to use Cipher for memory storage and retrieval",
                tutorial.system_type = "Dual storage: Neo4j (knowledge graph) + SQLite (conversations)",
                tutorial.key_concept = "Persistent memory across AI sessions",
                tutorial.location = "D:\\VS Workspace\\cipher-byterover",
                tutorial.created = datetime()
        `);
        
        // 2. Store Memory Storage Tutorial
        console.log('2. Storing memory storage tutorial...');
        await session.run(`
            MERGE (storage:TutorialStep {step: "Memory_Storage"})
            SET storage.title = "How to Store Memories in Cipher",
                storage.method1 = "Direct Neo4j: CREATE (m:Memory {content: 'your_knowledge', type: 'fact', tags: ['tag1', 'tag2']})",
                storage.method2 = "Via Cipher Tools: Use cipher_store_reasoning_memory function",
                storage.method3 = "Knowledge Graph: CREATE nodes with relationships",
                storage.best_practice = "Tag memories with relevant categories for easy retrieval",
                storage.example = "Store code patterns, user preferences, project context",
                storage.updated = datetime()
        `);
        
        // 3. Store Memory Retrieval Tutorial  
        console.log('3. Storing memory retrieval tutorial...');
        await session.run(`
            MERGE (retrieval:TutorialStep {step: "Memory_Retrieval"})
            SET retrieval.title = "How to Retrieve Memories from Cipher",
                retrieval.method1 = "Search Graph: MATCH (m:Memory) WHERE m.content CONTAINS 'keyword' RETURN m",
                retrieval.method2 = "Via Cipher Tools: Use cipher_search_memory or cipher_enhanced_search",
                retrieval.method3 = "Semantic Search: Find related concepts using embeddings",
                retrieval.best_practice = "Use specific keywords and tags for precise results",
                retrieval.example = "Search for 'python functions' or 'user preference'",
                retrieval.updated = datetime()
        `);
        
        // 4. Store Practical Examples
        console.log('4. Storing practical examples...');
        await session.run(`
            MERGE (examples:TutorialStep {step: "Practical_Examples"})
            SET examples.title = "Real-world Cipher Usage Examples",
                examples.example1 = "Code Patterns: Store reusable code snippets with descriptions",
                examples.example2 = "User Context: Remember user's coding style and preferences", 
                examples.example3 = "Project Memory: Store project architecture and decisions",
                examples.example4 = "Learning Notes: Store new concepts and their relationships",
                examples.example5 = "Bug Solutions: Store problem-solution pairs for future reference",
                examples.updated = datetime()
        `);
        
        // 5. Store Usage Strategies
        console.log('5. Storing usage strategies...');
        await session.run(`
            MERGE (strategies:TutorialStep {step: "Usage_Strategies"})
            SET strategies.title = "How Claude Will Use Cipher",
                strategies.strategy1 = "Session Continuity: Store conversation context between sessions",
                strategies.strategy2 = "Code Intelligence: Remember codebase patterns and architecture",
                strategies.strategy3 = "User Modeling: Learn and remember user preferences and style",
                strategies.strategy4 = "Knowledge Accumulation: Build up expertise over time",
                strategies.strategy5 = "Problem Solving: Reference previous solutions to similar problems",
                strategies.benefit = "Provides persistent, searchable memory across all interactions",
                strategies.updated = datetime()
        `);
        
        // 6. Store Gemini-Specific Tutorial
        console.log('6. Creating Gemini-specific tutorial...');
        await session.run(`
            MERGE (gemini:Tutorial {name: "Gemini_Cipher_Guide"})
            SET gemini.introduction = "Cipher is your persistent memory system. Think of it as external RAM that survives between conversations.",
                gemini.core_concept = "Two storage layers: Neo4j for structured knowledge, SQLite for conversation history",
                gemini.when_to_store = "Store important facts, code patterns, user preferences, project context, solutions to problems",
                gemini.when_to_retrieve = "Before answering questions, check if you have relevant previous knowledge stored",
                gemini.tagging_system = "Use descriptive tags like 'python', 'api_design', 'user_preference', 'bug_fix'",
                gemini.search_tips = "Search by keywords, tags, or concepts. Use semantic search for related ideas",
                gemini.example_workflow = "1. Check existing memory 2. Process new information 3. Store key insights 4. Tag appropriately",
                gemini.updated = datetime()
        `);
        
        // 7. Create relationships between tutorial components
        console.log('7. Creating tutorial relationships...');
        await session.run(`
            MATCH (tutorial:Tutorial {name: "Cipher_Memory_System"})
            MATCH (storage:TutorialStep {step: "Memory_Storage"})
            MATCH (retrieval:TutorialStep {step: "Memory_Retrieval"})
            MATCH (examples:TutorialStep {step: "Practical_Examples"})
            MATCH (strategies:TutorialStep {step: "Usage_Strategies"})
            MATCH (gemini:Tutorial {name: "Gemini_Cipher_Guide"})
            
            MERGE (tutorial)-[:INCLUDES]->(storage)
            MERGE (tutorial)-[:INCLUDES]->(retrieval)
            MERGE (tutorial)-[:INCLUDES]->(examples)
            MERGE (tutorial)-[:INCLUDES]->(strategies)
            MERGE (gemini)-[:REFERENCES]->(tutorial)
            MERGE (storage)-[:LEADS_TO]->(retrieval)
            MERGE (retrieval)-[:DEMONSTRATES]->(examples)
            MERGE (examples)-[:SUPPORTS]->(strategies)
        `);
        
        // 8. Store Claude's Intended Usage Pattern
        console.log('8. Storing Claude usage patterns...');
        await session.run(`
            MERGE (claude_usage:UsagePattern {agent: "Claude"})
            SET claude_usage.pattern = "Intelligent Memory Assistant",
                claude_usage.behavior1 = "Always check memory before responding to complex questions",
                claude_usage.behavior2 = "Store new insights and code patterns for future reference", 
                claude_usage.behavior3 = "Remember user's coding style and project preferences",
                claude_usage.behavior4 = "Build context across multiple conversations",
                claude_usage.behavior5 = "Use semantic search to find related previous solutions",
                claude_usage.memory_types = "Code patterns, architectural decisions, bug fixes, user preferences, project context",
                claude_usage.retrieval_triggers = "Complex questions, code requests, architecture discussions, debugging",
                claude_usage.updated = datetime()
        `);
        
        await session.close();
        console.log('\n✅ Cipher tutorial for Gemini stored in knowledge graph');
        
    } catch (error) {
        console.error('Error creating tutorial:', error.message);
    } finally {
        await driver.close();
    }
}

// Also create a human-readable tutorial
async function generateTutorialDocument() {
    console.log('\nGenerating human-readable tutorial...\n');
    
    const tutorial = `
# Cipher Memory System Tutorial for Gemini

## Overview
Cipher is a persistent memory system that allows AI agents to store and retrieve knowledge across sessions. Think of it as external RAM that survives between conversations.

## Architecture
- **Neo4j**: Knowledge graph for structured information and relationships
- **SQLite**: Conversation history and message storage
- **Dual Storage**: Enables both semantic search and conversation continuity

## How to Store Memories

### Method 1: Direct Neo4j Queries
\`\`\`cypher
CREATE (m:Memory {
    content: "Python functions should follow PEP 8 naming conventions",
    type: "coding_standard",
    tags: ["python", "pep8", "naming"],
    timestamp: datetime()
})
\`\`\`

### Method 2: Cipher Tools (Recommended)
- Use \`cipher_store_reasoning_memory\` function
- Automatically handles tagging and relationships
- Integrates with conversation context

### Method 3: Knowledge Graph Relationships
\`\`\`cypher
MERGE (user:User {name: "current_user"})
MERGE (pref:Preference {type: "coding_style", value: "functional_programming"})
MERGE (user)-[:PREFERS]->(pref)
\`\`\`

## How to Retrieve Memories

### Search by Keywords
\`\`\`cypher
MATCH (m:Memory) 
WHERE m.content CONTAINS "python" 
RETURN m.content, m.tags
\`\`\`

### Search by Tags
\`\`\`cypher
MATCH (m:Memory) 
WHERE "api_design" IN m.tags 
RETURN m
\`\`\`

### Semantic Search (via Cipher tools)
- Use \`cipher_enhanced_search\` for related concepts
- Finds memories by meaning, not just keywords

## Practical Examples

### 1. Code Patterns
\`\`\`
Store: "User prefers async/await over Promises for API calls"
Tags: ["javascript", "async", "user_preference", "api"]
\`\`\`

### 2. Project Context
\`\`\`
Store: "This trading system uses 11 core agents with centralized configuration"
Tags: ["trading_system", "architecture", "agents", "config"]
\`\`\`

### 3. Bug Solutions
\`\`\`
Store: "ChromaDB connection issues fixed by checking localhost:8000 first"
Tags: ["chromadb", "bug_fix", "connection", "localhost"]
\`\`\`

## Usage Strategies for AI Agents

### 1. Session Continuity
- Store conversation context at end of sessions
- Retrieve relevant context at start of new sessions

### 2. Learning Accumulation
- Store new concepts and their relationships
- Build expertise over time through accumulated knowledge

### 3. User Modeling
- Remember user's coding style and preferences
- Adapt responses based on stored user context

### 4. Problem Solving
- Store successful solutions with problem descriptions
- Search for similar problems before solving new ones

## Best Practices

### Storage
1. **Tag Appropriately**: Use descriptive, searchable tags
2. **Be Specific**: Store concrete, actionable information
3. **Create Relationships**: Link related concepts in the knowledge graph
4. **Include Context**: Store why something was important

### Retrieval
1. **Search First**: Always check existing memory before responding
2. **Use Multiple Methods**: Try keyword, tag, and semantic search
3. **Combine Results**: Merge insights from different memory searches
4. **Update Memory**: Refine stored information based on new insights

## Claude's Usage Pattern

As Claude, I will use Cipher to:

1. **Remember Project Context**: Store architecture decisions, code patterns, user preferences
2. **Build Conversation Continuity**: Reference previous discussions and decisions
3. **Accumulate Expertise**: Learn from each interaction and store insights
4. **Provide Better Assistance**: Use stored knowledge to give more relevant, personalized help
5. **Avoid Repetition**: Remember what's already been explained or solved

## Example Workflow

1. **Check Memory**: Search for relevant previous knowledge
2. **Process Request**: Understand current question/task
3. **Combine Knowledge**: Merge stored insights with current analysis
4. **Provide Response**: Give informed, context-aware answer
5. **Store Insights**: Save new knowledge for future reference
6. **Tag Appropriately**: Ensure future searchability

This creates a persistent, evolving knowledge base that improves assistance quality over time.
`;

    console.log(tutorial);
    return tutorial;
}

// Run tutorial creation
createGeminiCipherTutorial()
    .then(() => generateTutorialDocument())
    .catch(console.error);