#!/usr/bin/env python3
"""
Circular Dependency Detection and Analysis Tool

Analyzes Python codebases for circular dependencies and provides
refactoring suggestions with architectural discussions.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import networkx as nx
import matplotlib.pyplot as plt


@dataclass
class DependencyNode:
    """Represents a module/file in the dependency graph"""
    module_path: str
    module_name: str
    imports: List[str]
    circular_groups: List[List[str]]
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class CircularDependency:
    """Represents a detected circular dependency"""
    cycle: List[str]
    severity: str
    impact_analysis: str
    refactoring_options: List[str]
    discussion_points: List[str]


class DependencyAnalyzer:
    """Analyzes Python code for circular dependencies"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dependency_graph = nx.DiGraph()
        self.module_imports = {}
        self.circular_dependencies = []
        self.analysis_results = {}
        
    def analyze_project(self) -> Dict:
        """Main analysis method - returns comprehensive results"""
        print("[ANALYZER] Starting circular dependency analysis...")
        
        # Step 1: Discover all Python files
        python_files = self._discover_python_files()
        print(f"[DISCOVERY] Found {len(python_files)} Python files")
        
        # Step 2: Extract imports from each file
        self._extract_imports(python_files)
        print(f"[IMPORTS] Analyzed imports for {len(self.module_imports)} modules")
        
        # Step 3: Build dependency graph
        self._build_dependency_graph()
        print(f"[GRAPH] Built dependency graph with {self.dependency_graph.number_of_nodes()} nodes")
        
        # Step 4: Detect circular dependencies
        cycles = self._detect_cycles()
        print(f"[DETECTION] Found {len(cycles)} circular dependency cycles")
        
        # Step 5: Analyze each cycle
        self._analyze_cycles(cycles)
        
        # Step 6: Generate refactoring recommendations
        recommendations = self._generate_recommendations()
        
        # Step 7: Create discussion points
        discussions = self._create_discussion_points()
        
        return {
            'summary': {
                'total_files': len(python_files),
                'total_modules': len(self.module_imports),
                'circular_cycles': len(cycles),
                'critical_cycles': len([c for c in self.circular_dependencies if c.severity == 'critical']),
                'high_priority_cycles': len([c for c in self.circular_dependencies if c.severity == 'high'])
            },
            'cycles': [self._cycle_to_dict(c) for c in self.circular_dependencies],
            'recommendations': recommendations,
            'discussions': discussions,
            'graph_metrics': self._calculate_graph_metrics()
        }
    
    def _discover_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []
        
        # Skip common directories that shouldn't be analyzed
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 
                    '.env', '.venv', 'env', 'venv', 'build', 'dist'}
        
        for root, dirs, files in os.walk(self.project_root):
            # Remove skip directories from search
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def _extract_imports(self, python_files: List[Path]):
        """Extract import statements from Python files"""
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to extract imports
                tree = ast.parse(content)
                imports = self._extract_imports_from_ast(tree, file_path)
                
                # Convert to relative module name
                module_name = self._path_to_module_name(file_path)
                self.module_imports[module_name] = imports
                
            except Exception as e:
                print(f"[WARNING] Could not analyze {file_path}: {e}")
    
    def _extract_imports_from_ast(self, tree: ast.AST, file_path: Path) -> List[str]:
        """Extract imports from AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # Handle relative imports
                    if node.level > 0:
                        # Resolve relative import
                        relative_module = self._resolve_relative_import(
                            node.module, node.level, file_path
                        )
                        if relative_module:
                            imports.append(relative_module)
                    else:
                        imports.append(node.module)
        
        # Filter to only include internal project imports
        return [imp for imp in imports if self._is_internal_import(imp)]
    
    def _resolve_relative_import(self, module: Optional[str], level: int, file_path: Path) -> Optional[str]:
        """Resolve relative imports to absolute module names"""
        try:
            # Get the current module's package path
            current_module = self._path_to_module_name(file_path)
            parts = current_module.split('.')
            
            # Go up 'level' directories
            if level >= len(parts):
                return None
            
            base_parts = parts[:-level] if level > 0 else parts[:-1]
            
            if module:
                return '.'.join(base_parts + [module])
            else:
                return '.'.join(base_parts)
        except:
            return None
    
    def _path_to_module_name(self, file_path: Path) -> str:
        """Convert file path to Python module name"""
        # Make path relative to project root
        try:
            relative_path = file_path.relative_to(self.project_root)
        except ValueError:
            # File is outside project root
            return str(file_path.stem)
        
        # Convert path separators to dots and remove .py extension
        parts = list(relative_path.parts)
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        
        # Handle __init__.py files
        if parts[-1] == '__init__':
            parts.pop()
        
        return '.'.join(parts)
    
    def _is_internal_import(self, import_name: str) -> bool:
        """Check if import is internal to the project"""
        # Skip standard library and third-party imports
        stdlib_modules = {
            'os', 'sys', 'json', 'ast', 'pathlib', 'typing', 'collections',
            'dataclasses', 'networkx', 'matplotlib', 'datetime', 'asyncio',
            'logging', 'traceback', 'unittest', 'pytest', 'click', 'rich'
        }
        
        if import_name.split('.')[0] in stdlib_modules:
            return False
        
        # Check if it's a relative import within our project
        return True
    
    def _build_dependency_graph(self):
        """Build NetworkX graph from import relationships"""
        # Add all modules as nodes
        for module in self.module_imports.keys():
            self.dependency_graph.add_node(module)
        
        # Add edges for imports
        for module, imports in self.module_imports.items():
            for imported_module in imports:
                if imported_module in self.module_imports:
                    self.dependency_graph.add_edge(module, imported_module)
    
    def _detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies using NetworkX"""
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            return cycles
        except Exception as e:
            print(f"[ERROR] Cycle detection failed: {e}")
            return []
    
    def _analyze_cycles(self, cycles: List[List[str]]):
        """Analyze each circular dependency for severity and impact"""
        for cycle in cycles:
            severity = self._assess_cycle_severity(cycle)
            impact = self._analyze_cycle_impact(cycle)
            options = self._generate_refactoring_options(cycle)
            discussions = self._generate_cycle_discussions(cycle)
            
            circular_dep = CircularDependency(
                cycle=cycle,
                severity=severity,
                impact_analysis=impact,
                refactoring_options=options,
                discussion_points=discussions
            )
            
            self.circular_dependencies.append(circular_dep)
    
    def _assess_cycle_severity(self, cycle: List[str]) -> str:
        """Assess the severity of a circular dependency"""
        # Factors that increase severity:
        # 1. Cycle length (longer = more complex)
        # 2. Core/infrastructure modules involved
        # 3. Cross-layer dependencies
        
        cycle_length = len(cycle)
        
        # Check for core modules
        core_modules = {'core', 'base', 'framework', 'engine', 'main'}
        has_core_module = any(any(core in module.lower() for core in core_modules) 
                             for module in cycle)
        
        # Check for cross-layer dependencies
        layers = {'ui', 'api', 'service', 'data', 'core', 'util'}
        involved_layers = set()
        for module in cycle:
            module_lower = module.lower()
            for layer in layers:
                if layer in module_lower:
                    involved_layers.add(layer)
        
        cross_layer = len(involved_layers) > 2
        
        # Determine severity
        if cycle_length >= 4 or has_core_module or cross_layer:
            return 'critical'
        elif cycle_length == 3 or len(involved_layers) > 1:
            return 'high'
        elif cycle_length == 2:
            return 'medium'
        else:
            return 'low'
    
    def _analyze_cycle_impact(self, cycle: List[str]) -> str:
        """Analyze the impact of a circular dependency"""
        impacts = []
        
        # Testing impact
        impacts.append("Testing: Difficult to unit test modules in isolation")
        
        # Maintenance impact
        impacts.append("Maintenance: Changes in one module may require changes in all cycle members")
        
        # Deployment impact
        if len(cycle) > 2:
            impacts.append("Deployment: Cannot deploy/update modules independently")
        
        # Performance impact
        impacts.append("Performance: May cause import-time circular loading issues")
        
        # Architecture impact
        impacts.append("Architecture: Violates separation of concerns and layered architecture principles")
        
        return '; '.join(impacts)
    
    def _generate_refactoring_options(self, cycle: List[str]) -> List[str]:
        """Generate refactoring options for breaking the cycle"""
        options = []
        
        # Option 1: Extract common interface/base class
        options.append(
            "Extract Interface: Create a common interface/abstract base class "
            "that both modules can depend on, eliminating direct dependencies"
        )
        
        # Option 2: Dependency injection
        options.append(
            "Dependency Injection: Use dependency injection to inject dependencies "
            "at runtime rather than import time"
        )
        
        # Option 3: Event-driven architecture
        options.append(
            "Event-Driven: Replace direct calls with event publishing/subscribing "
            "to decouple modules"
        )
        
        # Option 4: Extract shared functionality
        options.append(
            "Extract Shared Module: Move shared functionality to a separate module "
            "that both can depend on"
        )
        
        # Option 5: Merge modules (if they're too tightly coupled)
        if len(cycle) == 2:
            options.append(
                "Merge Modules: If modules are too tightly coupled, consider merging "
                "them into a single cohesive module"
            )
        
        # Option 6: Layered architecture
        options.append(
            "Layered Architecture: Reorganize modules into clear layers where "
            "dependencies only flow downward"
        )
        
        return options
    
    def _generate_cycle_discussions(self, cycle: List[str]) -> List[str]:
        """Generate discussion points for the development team"""
        discussions = []
        
        discussions.append(
            f"Business Impact: How does this circular dependency between "
            f"{' -> '.join(cycle)} affect our ability to deliver features?"
        )
        
        discussions.append(
            "Refactoring Priority: What is the cost/benefit of fixing this cycle "
            "versus other technical debt?"
        )
        
        discussions.append(
            "Architecture Vision: How does this cycle align with our target "
            "architecture and design principles?"
        )
        
        discussions.append(
            "Team Ownership: Which team(s) own these modules and how should "
            "we coordinate the refactoring effort?"
        )
        
        discussions.append(
            "Testing Strategy: How can we ensure the refactoring doesn't break "
            "existing functionality?"
        )
        
        discussions.append(
            "Migration Plan: Should we fix this incrementally or as a big-bang "
            "refactoring?"
        )
        
        return discussions
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate prioritized refactoring recommendations"""
        recommendations = []
        
        # Sort cycles by severity
        sorted_cycles = sorted(
            self.circular_dependencies, 
            key=lambda x: {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity],
            reverse=True
        )
        
        for i, cycle_dep in enumerate(sorted_cycles[:5], 1):  # Top 5 priorities
            rec = {
                'priority': i,
                'severity': cycle_dep.severity,
                'cycle': ' -> '.join(cycle_dep.cycle),
                'recommended_approach': cycle_dep.refactoring_options[0],  # Top option
                'effort_estimate': self._estimate_refactoring_effort(cycle_dep),
                'business_value': self._estimate_business_value(cycle_dep)
            }
            recommendations.append(rec)
        
        return recommendations
    
    def _estimate_refactoring_effort(self, cycle_dep: CircularDependency) -> str:
        """Estimate effort required to fix the cycle"""
        cycle_length = len(cycle_dep.cycle)
        
        if cycle_length == 2:
            return "Medium (1-2 weeks)"
        elif cycle_length == 3:
            return "High (2-4 weeks)"
        else:
            return "Very High (1-2 months)"
    
    def _estimate_business_value(self, cycle_dep: CircularDependency) -> str:
        """Estimate business value of fixing the cycle"""
        if cycle_dep.severity in ['critical', 'high']:
            return "High - Improves maintainability and reduces technical debt"
        else:
            return "Medium - Incremental improvement to code quality"
    
    def _create_discussion_points(self) -> List[Dict]:
        """Create team discussion points"""
        discussions = []
        
        if len(self.circular_dependencies) > 0:
            discussions.append({
                'topic': 'Circular Dependency Prevention Strategy',
                'description': 'How can we prevent circular dependencies in future development?',
                'proposals': [
                    'Implement pre-commit hooks that detect circular dependencies',
                    'Add dependency analysis to CI/CD pipeline',
                    'Create architecture guidelines and review processes',
                    'Use static analysis tools in IDEs'
                ]
            })
        
        if any(cd.severity in ['critical', 'high'] for cd in self.circular_dependencies):
            discussions.append({
                'topic': 'Technical Debt Sprint Planning',
                'description': 'Should we dedicate sprint capacity to fixing critical circular dependencies?',
                'proposals': [
                    'Allocate 20% of sprint capacity to technical debt',
                    'Create dedicated technical debt epics',
                    'Fix cycles as part of feature development in affected areas'
                ]
            })
        
        return discussions
    
    def _calculate_graph_metrics(self) -> Dict:
        """Calculate useful graph metrics"""
        if self.dependency_graph.number_of_nodes() == 0:
            return {}
        
        return {
            'total_nodes': self.dependency_graph.number_of_nodes(),
            'total_edges': self.dependency_graph.number_of_edges(),
            'density': nx.density(self.dependency_graph),
            'strongly_connected_components': len(list(nx.strongly_connected_components(self.dependency_graph))),
            'average_in_degree': sum(dict(self.dependency_graph.in_degree()).values()) / self.dependency_graph.number_of_nodes(),
            'average_out_degree': sum(dict(self.dependency_graph.out_degree()).values()) / self.dependency_graph.number_of_nodes()
        }
    
    def _cycle_to_dict(self, cycle_dep: CircularDependency) -> Dict:
        """Convert CircularDependency to dictionary"""
        return {
            'cycle': cycle_dep.cycle,
            'severity': cycle_dep.severity,
            'impact_analysis': cycle_dep.impact_analysis,
            'refactoring_options': cycle_dep.refactoring_options,
            'discussion_points': cycle_dep.discussion_points
        }
    
    def export_results(self, output_path: Path, format: str = 'json'):
        """Export analysis results"""
        results = self.analysis_results
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        print(f"[EXPORT] Results exported to {output_path}")
    
    def visualize_dependencies(self, output_path: Path = None, show_cycles_only: bool = False):
        """Create a visual representation of dependencies"""
        try:
            import matplotlib.pyplot as plt
            
            if show_cycles_only:
                # Create subgraph with only cycles
                cycle_nodes = set()
                for cycle_dep in self.circular_dependencies:
                    cycle_nodes.update(cycle_dep.cycle)
                subgraph = self.dependency_graph.subgraph(cycle_nodes)
            else:
                subgraph = self.dependency_graph
            
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(subgraph, pos, node_color='lightblue', 
                                 node_size=500, alpha=0.7)
            
            # Draw edges
            nx.draw_networkx_edges(subgraph, pos, edge_color='gray', 
                                 arrows=True, arrowsize=20, alpha=0.5)
            
            # Draw labels
            nx.draw_networkx_labels(subgraph, pos, font_size=8)
            
            plt.title("Module Dependency Graph" + (" - Cycles Only" if show_cycles_only else ""))
            plt.axis('off')
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"[VISUALIZATION] Graph saved to {output_path}")
            else:
                plt.show()
                
        except ImportError:
            print("[WARNING] matplotlib not available for visualization")


def main():
    """Main entry point for dependency analysis"""
    if len(sys.argv) < 2:
        print("Usage: python dependency_analyzer.py <project_path> [--export output.json] [--visualize graph.png]")
        sys.exit(1)
    
    project_path = Path(sys.argv[1])
    
    if not project_path.exists():
        print(f"[ERROR] Project path does not exist: {project_path}")
        sys.exit(1)
    
    # Run analysis
    analyzer = DependencyAnalyzer(project_path)
    results = analyzer.analyze_project()
    analyzer.analysis_results = results
    
    # Print summary
    print("\n" + "=" * 70)
    print("CIRCULAR DEPENDENCY ANALYSIS RESULTS")
    print("=" * 70)
    
    summary = results['summary']
    print(f"Total Files Analyzed: {summary['total_files']}")
    print(f"Total Modules: {summary['total_modules']}")
    print(f"Circular Dependency Cycles: {summary['circular_cycles']}")
    print(f"Critical Cycles: {summary['critical_cycles']}")
    print(f"High Priority Cycles: {summary['high_priority_cycles']}")
    
    if results['cycles']:
        print(f"\nTop Priority Cycles:")
        for i, cycle in enumerate(results['cycles'][:3], 1):
            print(f"  {i}. {' -> '.join(cycle['cycle'])} (Severity: {cycle['severity']})")
    
    # Handle command line arguments
    if '--export' in sys.argv:
        export_idx = sys.argv.index('--export')
        if export_idx + 1 < len(sys.argv):
            export_path = Path(sys.argv[export_idx + 1])
            analyzer.export_results(export_path)
    
    if '--visualize' in sys.argv:
        viz_idx = sys.argv.index('--visualize')
        if viz_idx + 1 < len(sys.argv):
            viz_path = Path(sys.argv[viz_idx + 1])
            analyzer.visualize_dependencies(viz_path, show_cycles_only=True)


if __name__ == "__main__":
    main()