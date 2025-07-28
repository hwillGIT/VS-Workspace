import os
import re
from collections import defaultdict, deque
from typing import Set, Dict, List, Tuple

def extract_imports_from_file(filepath: str) -> List[str]:
    """Extract all agent imports from a Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        imports = []
        
        # Pattern to match relative imports within agents
        patterns = [
            # from ..other_agent import
            r'from\s+\.\.([a-zA-Z_][a-zA-Z0-9_]*)\s+import',
            # from .other_agent import  
            r'from\s+\.([a-zA-Z_][a-zA-Z0-9_]*)\s+import',
            # from ...agents.other_agent import
            r'from\s+\.\.\.agents\.([a-zA-Z_][a-zA-Z0-9_]*)\s+import',
            # from ....agents.other_agent import (for strategies)
            r'from\s+\.\.\.\.agents\.([a-zA-Z_][a-zA-Z0-9_]*)\s+import',
            # Direct agent imports
            r'from\s+agents\.([a-zA-Z_][a-zA-Z0-9_]*)\s+import'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            imports.extend(matches)
        
        return imports
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

def get_agent_name_from_path(filepath: str) -> str:
    """Extract agent name from file path."""
    # Convert path separators to forward slashes for consistency
    filepath = filepath.replace('\\', '/')
    
    # Extract the agent module path
    if 'system_architect' in filepath:
        return 'system_architect.' + os.path.basename(filepath).replace('.py', '')
    elif 'strategies' in filepath:
        # For strategy agents like momentum/momentum_agent.py
        parts = filepath.split('/')
        if 'strategies' in parts:
            idx = parts.index('strategies')
            if len(parts) > idx + 1:
                strategy_type = parts[idx + 1]
                filename = os.path.basename(filepath).replace('.py', '')
                return f'strategies.{strategy_type}.{filename}'
    else:
        # For main agents like data_universe/data_universe_agent.py
        parts = filepath.split('/')
        if 'agents' in parts:
            idx = parts.index('agents')
            if len(parts) > idx + 1:
                agent_type = parts[idx + 1]
                filename = os.path.basename(filepath).replace('.py', '')
                return f'{agent_type}.{filename}'
    
    # Fallback
    return os.path.basename(filepath).replace('.py', '')

def build_dependency_graph() -> Dict[str, List[str]]:
    """Build a graph of dependencies between agents."""
    graph = defaultdict(list)
    
    agents_dir = 'trading_system/agents'
    
    for root, dirs, files in os.walk(agents_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py' and not file.startswith('test_'):
                filepath = os.path.join(root, file)
                agent_name = get_agent_name_from_path(filepath)
                
                imports = extract_imports_from_file(filepath)
                
                # Filter and clean imports to get actual agent references
                for imported_module in imports:
                    if imported_module and imported_module != agent_name:
                        graph[agent_name].append(imported_module)
    
    return graph

def find_circular_dependencies(graph: Dict[str, List[str]]) -> List[List[str]]:
    """Find circular dependencies using DFS."""
    visited = set()
    rec_stack = set()
    cycles = []
    
    def dfs(node: str, path: List[str]) -> bool:
        if node in rec_stack:
            # Found a cycle
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return True
        
        if node in visited:
            return False
        
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        # Check all dependencies
        for neighbor in graph.get(node, []):
            if dfs(neighbor, path):
                pass  # Continue to find all cycles
        
        rec_stack.remove(node)
        path.pop()
        return False
    
    for node in graph.keys():
        if node not in visited:
            dfs(node, [])
    
    return cycles

def main():
    print("CIRCULAR DEPENDENCY ANALYSIS FOR TRADING SYSTEM")
    print("=" * 60)
    
    # Build dependency graph
    graph = build_dependency_graph()
    
    print(f"Found {len(graph)} agents with dependencies:")
    print()
    
    # Show all dependencies
    for agent, deps in graph.items():
        if deps:
            print(f"{agent}:")
            for dep in deps:
                print(f"  -> {dep}")
            print()
    
    # Find circular dependencies
    cycles = find_circular_dependencies(graph)
    
    print("\nCIRCULAR DEPENDENCY ANALYSIS:")
    print("=" * 40)
    
    if cycles:
        print(f"WARNING: Found {len(cycles)} circular dependencies:")
        print()
        
        for i, cycle in enumerate(cycles, 1):
            print(f"Cycle {i}:")
            for j in range(len(cycle) - 1):
                print(f"  {cycle[j]} -> {cycle[j+1]}")
            print()
    else:
        print("OK: No circular dependencies found!")
    
    # Additional analysis for broken imports
    print("\nBROKEN IMPORT ANALYSIS:")
    print("=" * 30)
    
    broken_imports = []
    agents_dir = 'trading_system/agents'
    
    for root, dirs, files in os.walk(agents_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py' and not file.startswith('test_'):
                filepath = os.path.join(root, file)
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for ..base.agent imports (should be ...core.base.agent)
                    if '..base.agent' in content:
                        broken_imports.append((filepath, '..base.agent should be ...core.base.agent'))
                        
                except Exception as e:
                    print(f"Error checking {filepath}: {e}")
    
    if broken_imports:
        print(f"WARNING: Found {len(broken_imports)} broken imports:")
        for filepath, issue in broken_imports:
            print(f"  {filepath}: {issue}")
    else:
        print("OK: No broken imports found!")

if __name__ == "__main__":
    main()