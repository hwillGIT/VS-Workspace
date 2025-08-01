import sys
import os
import traceback
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'ClaudeCode', 'context_management'))
from chroma_context_manager import ChromaContextManager, ContextLevel

try:
    # Read arguments from JSON file
    args_file_path = sys.argv[1]
    with open(args_file_path, 'r') as f:
        args = json.load(f)

    cm = ChromaContextManager()
    content = args[0]
    level = ContextLevel[args[1].upper()]
    metadata = args[2] if len(args) > 2 else {}

    doc_id = cm.add_context(content, level, metadata)
    print(f"Added to ChromaDB with ID: {doc_id}")
except Exception as e:
    print(f"Error adding to ChromaDB: {e}")
    traceback.print_exc()
    sys.exit(1)