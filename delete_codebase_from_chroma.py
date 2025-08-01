import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'ClaudeCode', 'context_management'))
from chroma_context_manager import ChromaContextManager, ContextLevel

async def delete_codebase_documents():
    cm = ChromaContextManager()
    
    # Iterate through all context levels and delete documents with the specified source
    for level in ContextLevel:
        try:
            print(f"Attempting to delete documents from {level.value} collection...")
            if cm.collections[level]:
                cm.collections[level].delete(where={"source": "directory:VS Workspace"})
                print(f"Successfully attempted deletion from {level.value} collection.")
            else:
                print(f"Skipping deletion for {level.value} collection: not initialized.")
        except Exception as e:
            print(f"Error deleting documents from {level.value} collection: {e}")

    print("Deletion process complete.")

if __name__ == "__main__":
    asyncio.run(delete_codebase_documents())
