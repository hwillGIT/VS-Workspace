from chroma_context_manager import ChromaContextManager, ContextLevel

cm = ChromaContextManager(persist_directory='./chroma_context_db')

print("Getting most recent entries from all levels...")
print("=" * 60)

# Get recent entries from current session
session_context = cm.get_session_context(max_items=20)

print(f"Recent session context ({len(session_context)} entries):")
for i, ctx in enumerate(session_context, 1):
    content = ctx['content'][:200].replace('\u2192', '->')
    print(f"{i}. [{ctx['level']}] {content}...")
    print(f"   Timestamp: {ctx['metadata'].get('timestamp', 'N/A')}")
    if 'source' in ctx['metadata']:
        print(f"   Source: {ctx['metadata']['source']}")
    print()

# Also do a broad search to get most recent across all projects
print("\nBroad search across all projects (most recent):")
print("=" * 60)

all_results = cm.search_context('', n_results=30)  # Empty query to get recent entries

for i, result in enumerate(all_results[:10], 1):  # Show top 10
    content = result['content'][:200].replace('\u2192', '->')
    print(f"{i}. [{result['level']}] {content}...")
    print(f"   Project: {result['metadata'].get('project', 'N/A')}")
    print(f"   Timestamp: {result['metadata'].get('timestamp', 'N/A')}")
    if 'source' in result['metadata']:
        print(f"   Source: {result['metadata']['source']}")
    print()