#!/usr/bin/env python3
"""
Search ChromaDB for existing Gemini-Anthropic debug messaging work
"""

from chroma_context_manager import ChromaContextManager, ContextLevel
import json

def search_debug_context():
    # Initialize context manager
    cm = ChromaContextManager(persist_directory='./chroma_context_db')
    
    # Search queries
    search_queries = [
        'gemini debug anthropic messaging',
        'task tool subagent llm routing',
        'gemini to anthropic communication',
        'debug messaging system',
        'cross llm integration'
    ]
    
    all_results = {}
    
    for query in search_queries:
        print(f"\nSearching: '{query}'")
        print("=" * 50)
        
        results = cm.search_context(query, n_results=5)
        all_results[query] = results
        
        if results:
            for i, result in enumerate(results, 1):
                # Clean content to avoid Unicode issues
                content = result['content'].replace('\u2192', '->')
                content = content.replace('\u2022', '*')
                content = content.replace('\u2013', '-')
                content = content.replace('\u2014', '--')
                
                print(f"{i}. [{result['level']}] {content[:150]}...")
                print(f"   Distance: {result['distance']:.4f}")
                if result.get('metadata'):
                    print(f"   Metadata: {result['metadata']}")
                print()
        else:
            print("   No results found")
    
    # Also get recent session context
    print("\nRecent Session Context:")
    print("=" * 50)
    
    session_context = cm.get_session_context(max_items=10)
    
    for i, ctx in enumerate(session_context, 1):
        content = ctx['content'].replace('\u2192', '->')
        content = content.replace('\u2022', '*')
        content = content.replace('\u2013', '-')
        content = content.replace('\u2014', '--')
        
        print(f"{i}. [{ctx['level']}] {content[:100]}...")
        if ctx.get('metadata'):
            print(f"   Metadata: {ctx['metadata']}")
        print()
    
    return all_results

if __name__ == "__main__":
    results = search_debug_context()
    
    # Save results to file for analysis
    with open('debug_context_search_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to debug_context_search_results.json")