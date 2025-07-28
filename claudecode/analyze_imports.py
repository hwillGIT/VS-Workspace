import os
import re
from collections import defaultdict

def find_imports_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find relative imports between agents
        agent_imports = []
        
        # Match imports like 'from ...agents.xxx import' or 'from ..xxx import' etc
        patterns = [
            r'from\s+\.+agents\.([^.\s]+)(?:\.([^.\s]+))?\s+import',
            r'from\s+\.\.([^.\s]+)\s+import',
            r'from\s+\.([^.\s]+)\s+import'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    agent_imports.append(match)
                else:
                    agent_imports.append((match,))
        
        return agent_imports
    except Exception as e:
        return []

# Get all Python files in agents directory
agents_dir = 'trading_system/agents'
import_graph = defaultdict(list)

for root, dirs, files in os.walk(agents_dir):
    for file in files:
        if file.endswith('.py') and not file.startswith('test_') and file != '__init__.py':
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, agents_dir)
            
            imports = find_imports_in_file(filepath)
            if imports:
                import_graph[rel_path] = imports

print('IMPORT ANALYSIS FOR TRADING SYSTEM AGENTS')
print('=' * 60)
for file, imports in import_graph.items():
    if imports:
        print(f'{file}:')
        for imp in imports:
            print(f'  -> {imp}')
        print()

print('\nSUMMARY:')
print('=' * 30)
total_files = len(import_graph)
files_with_imports = len([f for f, i in import_graph.items() if i])
print(f'Total agent files analyzed: {total_files}')
print(f'Files with relative imports: {files_with_imports}')