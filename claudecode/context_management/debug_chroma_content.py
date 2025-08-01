from chroma_context_manager import ChromaContextManager, ContextLevel

cm = ChromaContextManager(persist_directory='./chroma_context_db')

print("Debugging ChromaDB content...")
print("=" * 60)

# Get raw collection data
for level in ContextLevel:
    collection = cm.collections[level]
    print(f"\n{level.value.upper()} Collection:")
    print(f"  Document count: {collection.count()}")
    
    # Get a few recent entries from this collection
    try:
        results = collection.get(limit=5)
        print(f"  Sample entries:")
        
        if results['documents']:
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                content = doc[:100].replace('\u2192', '->')
                timestamp = metadata.get('timestamp', 'N/A')
                source = metadata.get('source', 'N/A')
                print(f"    {i+1}. {content}...")
                print(f"       Timestamp: {timestamp}, Source: {source}")
        else:
            print("    No documents found")
            
    except Exception as e:
        print(f"    Error accessing collection: {e}")

# Try to find entries with "Gemini" in metadata source
print(f"\nSearching for entries with 'Gemini' in source...")
for level in ContextLevel:
    collection = cm.collections[level]
    try:
        results = collection.get(
            where={"source": {"$eq": "Gemini CLI"}},
            limit=10
        )
        
        if results['documents']:
            print(f"Found {len(results['documents'])} Gemini entries in {level.value}:")
            for doc, metadata in zip(results['documents'], results['metadatas']):
                print(f"  - {doc[:150]}...")
                print(f"    Metadata: {metadata}")
                
    except Exception as e:
        print(f"Error searching {level.value}: {e}")

print(f"\nCurrent project: {cm.current_project}")
print(f"Current session: {cm.current_session_id}")