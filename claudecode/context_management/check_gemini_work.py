from chroma_context_manager import ChromaContextManager, ContextLevel

cm = ChromaContextManager(persist_directory='./chroma_context_db')

# Search for Gemini's specific terms
print("Searching for Gemini's work...")

# Try different search terms
searches = [
    "self-debugger",
    "KeyError MEMORY", 
    "send_llm_prompt_via_task_tool",
    "Gemini CLI",
    "model router pattern"
]

for search_term in searches:
    print(f"\nSearching: '{search_term}'")
    results = cm.search_context(search_term, n_results=3)
    
    for i, result in enumerate(results, 1):
        content = result['content'][:150].replace('\u2192', '->')
        print(f"  {i}. [{result['level']}] {content}...")
        
        # Check if this looks like Gemini's content
        if any(term in result['content'].lower() for term in ['self-debugger', 'gemini cli', 'keyerror']):
            print(f"     ** FOUND GEMINI WORK **")
            print(f"     Full content: {result['content']}")
            print(f"     Metadata: {result['metadata']}")

# Also check current project context
print(f"\nCurrent project: {cm.current_project}")
print(f"Current session: {cm.current_session_id}")

# Get stats
stats = cm.get_statistics()
print(f"\nDatabase stats: {stats}")