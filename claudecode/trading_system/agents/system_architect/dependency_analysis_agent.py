"""
Dependency Analysis Agent

This agent performs comprehensive dependency analysis including circular dependency detection,
dependency graph generation, and dependency impact analysis for the trading system.
"""

import ast
import re
import os
import json
import logging
import networkx as nx
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import importlib.util
from collections import defaultdict

from ...core.base.agent import BaseAgent


@dataclass
class DependencyNode:
    """Represents a dependency node in the graph"""
    name: str
    type: str  # 'module', 'class', 'function', 'package', 'external'
    file_path: str
    line_number: int
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)
    is_circular: bool = False
    circular_path: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    coupling_score: float = 0.0


@dataclass
class CircularDependency:
    """Represents a circular dependency"""
    cycle_id: str
    nodes: List[str]  # Nodes involved in the cycle
    severity: str  # 'low', 'medium', 'high', 'critical'
    impact_score: float
    resolution_strategies: List[str]
    automated_fix_available: bool
    affected_files: List[str]
    cycle_length: int
    entry_points: List[str]  # Potential break points


@dataclass
class DependencyImpact:
    """Represents the impact of a dependency change"""
    target_node: str
    affected_nodes: List[str]
    impact_level: str  # 'direct', 'indirect', 'transitive'
    risk_score: float
    propagation_depth: int
    test_requirements: List[str]
    deployment_risk: str


@dataclass
class DependencyMetrics:
    """Comprehensive dependency metrics"""
    total_dependencies: int
    circular_dependencies: int
    coupling_index: float
    stability_index: float
    abstraction_level: float
    fan_in: Dict[str, int]
    fan_out: Dict[str, int]
    depth_metrics: Dict[str, int]
    critical_path_length: int
    dependency_clusters: List[List[str]]


class DependencyAnalysisAgent(BaseAgent):
    """
    Dependency Analysis Agent
    
    Analyzes project dependencies including:
    - Circular dependency detection and resolution
    - Dependency graph generation and visualization
    - Impact analysis for dependency changes
    - Coupling and cohesion metrics
    - Dependency optimization recommendations
    - Architecture layering validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DependencyAnalysis", config.get('dependency_analysis', {}))
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_circular_chain_length = config.get('max_circular_chain_length', 10)
        self.coupling_threshold = config.get('coupling_threshold', 5)
        self.include_external_deps = config.get('include_external_deps', True)
        self.exclude_patterns = config.get('exclude_patterns', ['test_', '__pycache__', '.git'])
        
        # Analysis state
        self.dependency_graph = nx.DiGraph()
        self.module_map: Dict[str, DependencyNode] = {}
        self.import_patterns = self._load_import_patterns()
        
    async def analyze_dependencies(self, target_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive dependency analysis
        
        Args:
            target_path: Path to analyze
            
        Returns:
            Dependency analysis results
        """
        self.logger.info(f"Starting dependency analysis of {target_path}")
        
        # Clear previous state
        self.dependency_graph.clear()
        self.module_map.clear()
        
        # Build dependency graph
        await self._build_dependency_graph(target_path)
        
        # Detect circular dependencies
        circular_deps = self._detect_circular_dependencies()
        
        # Calculate metrics
        metrics = self._calculate_dependency_metrics()
        
        # Analyze impact
        impact_analysis = self._analyze_dependency_impact()
        
        # Generate recommendations
        recommendations = self._generate_dependency_recommendations(circular_deps, metrics)
        
        return {
            'dependency_graph': self._graph_to_dict(),
            'circular_dependencies': [self._circular_dep_to_dict(cd) for cd in circular_deps],
            'metrics': self._metrics_to_dict(metrics),
            'impact_analysis': impact_analysis,
            'recommendations': recommendations,
            'critical_dependencies': self._identify_critical_dependencies(),
            'optimization_opportunities': self._identify_optimization_opportunities(),
            'architecture_violations': self._detect_architecture_violations(),
            'refactoring_priorities': self._prioritize_dependency_refactoring(circular_deps)
        }
    
    async def _build_dependency_graph(self, target_path: str) -> None:
        """Build the dependency graph from source code"""
        path = Path(target_path)
        
        if path.is_file() and path.suffix == '.py':
            await self._analyze_file_dependencies(path)
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                if not self._should_skip_file(py_file):
                    await self._analyze_file_dependencies(py_file)
        
        # Add edges for all dependencies
        self._add_dependency_edges()
    
    async def _analyze_file_dependencies(self, file_path: Path) -> None:
        """Analyze dependencies in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(content)
                node = self._create_dependency_node(file_path, tree)
                self.module_map[str(file_path)] = node
                self.dependency_graph.add_node(str(file_path), **self._node_to_dict(node))
            except SyntaxError as e:
                self.logger.warning(f"Syntax error in {file_path}: {e}")
                
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
    
    def _create_dependency_node(self, file_path: Path, tree: ast.AST) -> DependencyNode:
        """Create a dependency node from AST analysis"""
        imports = []
        exports = []
        dependencies = []
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.name
                    imports.append(import_name)
                    dependencies.append(import_name)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    from_module = node.module
                    imports.append(from_module)
                    dependencies.append(from_module)
                    
                    for alias in node.names:
                        item_name = f"{from_module}.{alias.name}"
                        imports.append(item_name)
                        dependencies.append(item_name)
        
        # Extract exports (functions, classes, variables)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):  # Public functions
                    exports.append(f"{file_path.stem}.{node.name}")
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith('_'):  # Public classes
                    exports.append(f"{file_path.stem}.{node.name}")
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and not target.id.startswith('_'):
                        exports.append(f"{file_path.stem}.{target.id}")
        
        return DependencyNode(
            name=file_path.stem,
            type='module',
            file_path=str(file_path),
            line_number=1,
            imports=imports,
            exports=exports,
            dependencies=list(set(dependencies))  # Remove duplicates
        )
    
    def _add_dependency_edges(self) -> None:
        """Add edges to the dependency graph"""
        for file_path, node in self.module_map.items():
            for dep in node.dependencies:
                # Try to find the dependency in our module map
                dep_node = self._find_dependency_node(dep)
                if dep_node:
                    self.dependency_graph.add_edge(file_path, dep_node.file_path, dependency=dep)
                    dep_node.dependents.append(file_path)
    
    def _find_dependency_node(self, dep_name: str) -> Optional[DependencyNode]:
        """Find a dependency node by name"""
        # Simple matching - in practice, this would be more sophisticated
        for node in self.module_map.values():
            if (node.name == dep_name or 
                dep_name in node.exports or
                any(exp.endswith(f".{dep_name}") for exp in node.exports)):
                return node
        
        # Check if it's a relative import
        if dep_name.startswith('.'):
            # Handle relative imports
            pass
            
        return None
    
    def _detect_circular_dependencies(self) -> List[CircularDependency]:
        """Detect circular dependencies in the graph"""
        circular_deps = []
        
        try:
            # Find strongly connected components
            sccs = list(nx.strongly_connected_components(self.dependency_graph))
            
            for i, scc in enumerate(sccs):
                if len(scc) > 1:  # Circular dependency found
                    cycle_nodes = list(scc)
                    
                    # Find the actual cycle path
                    cycle_path = self._find_cycle_path(cycle_nodes)
                    
                    # Calculate severity
                    severity = self._calculate_cycle_severity(cycle_nodes, cycle_path)
                    
                    # Generate resolution strategies
                    strategies = self._generate_resolution_strategies(cycle_nodes, cycle_path)
                    
                    circular_dep = CircularDependency(
                        cycle_id=f"cycle_{i}",
                        nodes=cycle_nodes,
                        severity=severity,
                        impact_score=self._calculate_cycle_impact(cycle_nodes),
                        resolution_strategies=strategies,
                        automated_fix_available=self._can_auto_fix_cycle(cycle_nodes, cycle_path),
                        affected_files=[self.module_map[node].file_path for node in cycle_nodes if node in self.module_map],
                        cycle_length=len(cycle_path),
                        entry_points=self._find_cycle_entry_points(cycle_nodes)
                    )
                    
                    circular_deps.append(circular_dep)
                    
                    # Mark nodes as circular
                    for node in cycle_nodes:
                        if node in self.module_map:
                            self.module_map[node].is_circular = True
                            self.module_map[node].circular_path = cycle_path
                            
        except Exception as e:
            self.logger.error(f"Error detecting circular dependencies: {e}")
        
        return circular_deps
    
    def _find_cycle_path(self, cycle_nodes: List[str]) -> List[str]:
        """Find the actual path of a circular dependency"""
        if not cycle_nodes:
            return []
        
        try:
            # Create subgraph with only cycle nodes
            subgraph = self.dependency_graph.subgraph(cycle_nodes)
            
            # Find a cycle using DFS
            start_node = cycle_nodes[0]
            visited = set()
            path = []
            
            def dfs(node, target):
                if node in visited:
                    if node == target:
                        return True
                    return False
                
                visited.add(node)
                path.append(node)
                
                for neighbor in subgraph.successors(node):
                    if dfs(neighbor, target):
                        return True
                
                path.pop()
                return False
            
            if dfs(start_node, start_node):
                return path + [start_node]  # Complete the cycle
                
        except Exception as e:
            self.logger.error(f"Error finding cycle path: {e}")
        
        return cycle_nodes  # Fallback
    
    def _calculate_cycle_severity(self, cycle_nodes: List[str], cycle_path: List[str]) -> str:
        """Calculate the severity of a circular dependency"""
        # Factors: cycle length, coupling, number of public interfaces
        cycle_length = len(cycle_path)
        
        if cycle_length <= 2:
            return 'high'  # Direct circular dependency
        elif cycle_length <= 4:
            return 'medium'
        elif cycle_length <= 6:
            return 'low'
        else:
            return 'low'  # Long cycles are often less problematic
    
    def _calculate_cycle_impact(self, cycle_nodes: List[str]) -> float:
        """Calculate the impact score of a circular dependency"""
        impact = 0.0
        
        for node in cycle_nodes:
            if node in self.module_map:
                dep_node = self.module_map[node]
                # Impact based on number of dependents and exports
                impact += len(dep_node.dependents) * 0.3
                impact += len(dep_node.exports) * 0.2
                impact += len(dep_node.dependencies) * 0.1
        
        return min(impact, 10.0)  # Cap at 10
    
    def _generate_resolution_strategies(self, cycle_nodes: List[str], cycle_path: List[str]) -> List[str]:
        """Generate strategies to resolve circular dependencies"""
        strategies = []
        
        # Common resolution strategies
        strategies.append("Extract common functionality into a shared module")
        strategies.append("Use dependency injection to break direct dependencies")
        strategies.append("Introduce interfaces or abstract base classes")
        strategies.append("Move shared code to a higher-level module")
        strategies.append("Merge modules if they are tightly coupled")
        strategies.append("Use late imports (import inside functions)")
        
        # Specific strategies based on cycle characteristics
        if len(cycle_path) == 2:
            strategies.insert(0, "Consider merging the two modules if they are closely related")
        elif len(cycle_path) > 5:
            strategies.insert(0, "Break the longest dependency chain by extracting interfaces")
        
        return strategies[:4]  # Return top 4 strategies
    
    def _can_auto_fix_cycle(self, cycle_nodes: List[str], cycle_path: List[str]) -> bool:
        """Check if a circular dependency can be automatically fixed"""
        # Simple heuristics for auto-fixing
        if len(cycle_path) == 2:
            # Check if one of the dependencies is just for type hints
            for node in cycle_nodes:
                if node in self.module_map:
                    dep_node = self.module_map[node]
                    # Look for TYPE_CHECKING imports
                    content_path = Path(dep_node.file_path)
                    if content_path.exists():
                        try:
                            with open(content_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            if 'TYPE_CHECKING' in content:
                                return True
                        except:
                            pass
        
        return False
    
    def _find_cycle_entry_points(self, cycle_nodes: List[str]) -> List[str]:
        """Find potential entry points to break the cycle"""
        entry_points = []
        
        for node in cycle_nodes:
            if node in self.module_map:
                dep_node = self.module_map[node]
                # Nodes with fewer dependencies are better entry points
                if len(dep_node.dependencies) <= 2:
                    entry_points.append(node)
        
        return entry_points[:3]  # Return top 3 entry points
    
    def _calculate_dependency_metrics(self) -> DependencyMetrics:
        """Calculate comprehensive dependency metrics"""
        total_deps = self.dependency_graph.number_of_edges()
        circular_deps = len([scc for scc in nx.strongly_connected_components(self.dependency_graph) if len(scc) > 1])
        
        # Calculate fan-in and fan-out
        fan_in = {}
        fan_out = {}
        
        for node in self.dependency_graph.nodes():
            fan_in[node] = self.dependency_graph.in_degree(node)
            fan_out[node] = self.dependency_graph.out_degree(node)
        
        # Calculate coupling index (average fan-out)
        coupling_index = sum(fan_out.values()) / len(fan_out) if fan_out else 0.0
        
        # Calculate stability index (fan-out / (fan-in + fan-out))
        stability_scores = []
        for node in self.dependency_graph.nodes():
            total_coupling = fan_in[node] + fan_out[node]
            if total_coupling > 0:
                stability = fan_out[node] / total_coupling
                stability_scores.append(stability)
        
        stability_index = sum(stability_scores) / len(stability_scores) if stability_scores else 0.0
        
        # Calculate abstraction level (percentage of abstract nodes)
        abstract_nodes = 0
        for node_path in self.dependency_graph.nodes():
            if node_path in self.module_map:
                node = self.module_map[node_path]
                # Check if node contains abstract classes or interfaces
                if self._is_abstract_module(node):
                    abstract_nodes += 1
        
        abstraction_level = abstract_nodes / len(self.module_map) if self.module_map else 0.0
        
        # Calculate depth metrics
        depth_metrics = {}
        try:
            for node in self.dependency_graph.nodes():
                # Calculate the longest path from this node
                paths = nx.single_source_shortest_path_length(self.dependency_graph, node)
                depth_metrics[node] = max(paths.values()) if paths else 0
        except:
            depth_metrics = {node: 0 for node in self.dependency_graph.nodes()}
        
        # Find critical path
        critical_path_length = max(depth_metrics.values()) if depth_metrics else 0
        
        # Find dependency clusters
        clusters = self._find_dependency_clusters()
        
        return DependencyMetrics(
            total_dependencies=total_deps,
            circular_dependencies=circular_deps,
            coupling_index=coupling_index,
            stability_index=stability_index,
            abstraction_level=abstraction_level,
            fan_in=fan_in,
            fan_out=fan_out,
            depth_metrics=depth_metrics,
            critical_path_length=critical_path_length,
            dependency_clusters=clusters
        )
    
    def _is_abstract_module(self, node: DependencyNode) -> bool:
        """Check if a module contains abstract classes or interfaces"""
        try:
            with open(node.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for abstract classes, protocols, or interfaces
            abstract_patterns = [
                r'class.*\(ABC\)',
                r'class.*\(Protocol\)',
                r'@abc\.abstractmethod',
                r'from abc import',
                r'from typing import Protocol'
            ]
            
            for pattern in abstract_patterns:
                if re.search(pattern, content):
                    return True
                    
        except Exception:
            pass
        
        return False
    
    def _find_dependency_clusters(self) -> List[List[str]]:
        """Find clusters of tightly coupled dependencies"""
        clusters = []
        
        try:
            # Use community detection algorithm
            undirected_graph = self.dependency_graph.to_undirected()
            
            # Simple clustering based on connected components
            components = list(nx.connected_components(undirected_graph))
            
            # Filter clusters with more than 2 nodes
            clusters = [list(component) for component in components if len(component) > 2]
            
        except Exception as e:
            self.logger.error(f"Error finding dependency clusters: {e}")
        
        return clusters
    
    def _analyze_dependency_impact(self) -> Dict[str, Any]:
        """Analyze the impact of potential dependency changes"""
        impact_analysis = {
            'high_impact_nodes': [],
            'bottleneck_nodes': [],
            'isolated_nodes': [],
            'hub_nodes': []
        }
        
        for node_path in self.dependency_graph.nodes():
            if node_path in self.module_map:
                node = self.module_map[node_path]
                
                fan_in = self.dependency_graph.in_degree(node_path)
                fan_out = self.dependency_graph.out_degree(node_path)
                
                # High impact nodes (many dependents)
                if fan_in > 5:
                    impact_analysis['high_impact_nodes'].append({
                        'name': node.name,
                        'path': node_path,
                        'dependents': fan_in,
                        'risk_level': 'high' if fan_in > 10 else 'medium'
                    })
                
                # Bottleneck nodes (high fan-in and fan-out)
                if fan_in > 3 and fan_out > 3:
                    impact_analysis['bottleneck_nodes'].append({
                        'name': node.name,
                        'path': node_path,
                        'coupling_score': fan_in + fan_out
                    })
                
                # Isolated nodes (no dependencies)
                if fan_in == 0 and fan_out == 0:
                    impact_analysis['isolated_nodes'].append({
                        'name': node.name,
                        'path': node_path
                    })
                
                # Hub nodes (many dependencies)
                if fan_out > 5:
                    impact_analysis['hub_nodes'].append({
                        'name': node.name,
                        'path': node_path,
                        'dependencies': fan_out
                    })
        
        return impact_analysis
    
    def _generate_dependency_recommendations(self, circular_deps: List[CircularDependency], 
                                           metrics: DependencyMetrics) -> List[str]:
        """Generate dependency optimization recommendations"""
        recommendations = []
        
        # Circular dependency recommendations
        if circular_deps:
            recommendations.append(f"Resolve {len(circular_deps)} circular dependencies to improve maintainability")
            
            critical_cycles = [cd for cd in circular_deps if cd.severity in ['high', 'critical']]
            if critical_cycles:
                recommendations.append(f"Priority: Fix {len(critical_cycles)} critical circular dependencies first")
        
        # Coupling recommendations
        if metrics.coupling_index > self.coupling_threshold:
            recommendations.append(f"Reduce coupling (current: {metrics.coupling_index:.1f}, target: <{self.coupling_threshold})")
            recommendations.append("Consider using dependency injection or interfaces to reduce direct dependencies")
        
        # Stability recommendations
        if metrics.stability_index < 0.3:
            recommendations.append("Improve stability by reducing outgoing dependencies in stable modules")
        elif metrics.stability_index > 0.8:
            recommendations.append("Some modules may be too unstable - consider consolidating responsibilities")
        
        # Abstraction recommendations
        if metrics.abstraction_level < 0.2:
            recommendations.append("Increase abstraction level by introducing interfaces or abstract base classes")
        
        # Cluster recommendations
        if len(metrics.dependency_clusters) > 5:
            recommendations.append("Consider consolidating some dependency clusters to reduce complexity")
        
        return recommendations
    
    def _identify_critical_dependencies(self) -> List[Dict[str, Any]]:
        """Identify critical dependencies that pose high risk"""
        critical_deps = []
        
        for node_path in self.dependency_graph.nodes():
            if node_path in self.module_map:
                node = self.module_map[node_path]
                
                fan_in = self.dependency_graph.in_degree(node_path)
                fan_out = self.dependency_graph.out_degree(node_path)
                
                # Critical if many modules depend on it
                criticality_score = fan_in * 2 + fan_out
                
                if criticality_score > 8:
                    critical_deps.append({
                        'name': node.name,
                        'path': node_path,
                        'criticality_score': criticality_score,
                        'dependents': fan_in,
                        'dependencies': fan_out,
                        'risk_factors': self._identify_risk_factors(node),
                        'mitigation_strategies': self._generate_mitigation_strategies(node)
                    })
        
        return sorted(critical_deps, key=lambda x: x['criticality_score'], reverse=True)
    
    def _identify_risk_factors(self, node: DependencyNode) -> List[str]:
        """Identify risk factors for a dependency"""
        risk_factors = []
        
        if node.is_circular:
            risk_factors.append("Part of circular dependency")
        
        if len(node.dependents) > 10:
            risk_factors.append("High number of dependents")
        
        if len(node.dependencies) > 15:
            risk_factors.append("High number of dependencies")
        
        # Check if it's in a critical path
        try:
            with open(node.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'TODO' in content or 'FIXME' in content:
                risk_factors.append("Contains technical debt markers")
                
            if len(content.split('\n')) > 500:
                risk_factors.append("Large file size")
                
        except Exception:
            pass
        
        return risk_factors
    
    def _generate_mitigation_strategies(self, node: DependencyNode) -> List[str]:
        """Generate mitigation strategies for critical dependencies"""
        strategies = []
        
        if len(node.dependents) > 5:
            strategies.append("Create interface to abstract implementation details")
            strategies.append("Add comprehensive unit tests")
        
        if len(node.dependencies) > 10:
            strategies.append("Split module into smaller, focused modules")
            strategies.append("Use dependency injection to reduce coupling")
        
        if node.is_circular:
            strategies.extend(node.circular_path[:2])  # Use resolution strategies
        
        strategies.append("Implement version compatibility checks")
        strategies.append("Add integration tests for critical paths")
        
        return strategies[:3]  # Return top 3 strategies
    
    def _identify_optimization_opportunities(self) -> Dict[str, List[str]]:
        """Identify opportunities for dependency optimization"""
        opportunities = {
            'merge_candidates': [],
            'split_candidates': [],
            'interface_candidates': [],
            'lazy_loading_candidates': []
        }
        
        for node_path in self.module_map:
            node = self.module_map[node_path]
            
            fan_in = self.dependency_graph.in_degree(node_path)
            fan_out = self.dependency_graph.out_degree(node_path)
            
            # Merge candidates (small, tightly coupled modules)
            if fan_out <= 2 and len(node.exports) <= 3:
                opportunities['merge_candidates'].append(node.name)
            
            # Split candidates (large modules with many responsibilities)
            if fan_out > 10 and len(node.exports) > 20:
                opportunities['split_candidates'].append(node.name)
            
            # Interface candidates (modules with many dependents)
            if fan_in > 5:
                opportunities['interface_candidates'].append(node.name)
            
            # Lazy loading candidates (modules with expensive dependencies)
            if fan_out > 8:
                opportunities['lazy_loading_candidates'].append(node.name)
        
        return opportunities
    
    def _detect_architecture_violations(self) -> List[Dict[str, Any]]:
        """Detect violations of architectural principles"""
        violations = []
        
        # Layer violations (lower layers depending on higher layers)
        layer_violations = self._detect_layer_violations()
        violations.extend(layer_violations)
        
        # Abstraction violations (concrete depending on concrete)
        abstraction_violations = self._detect_abstraction_violations()
        violations.extend(abstraction_violations)
        
        # Coupling violations (too many dependencies)
        coupling_violations = self._detect_coupling_violations()
        violations.extend(coupling_violations)
        
        return violations
    
    def _detect_layer_violations(self) -> List[Dict[str, Any]]:
        """Detect architectural layer violations"""
        violations = []
        
        # Define typical layers based on directory structure
        layer_hierarchy = {
            'ui': 4,
            'api': 3,
            'service': 2,
            'data': 1,
            'core': 0
        }
        
        for edge in self.dependency_graph.edges():
            source, target = edge
            
            source_layer = self._get_module_layer(source, layer_hierarchy)
            target_layer = self._get_module_layer(target, layer_hierarchy)
            
            # Violation if lower layer depends on higher layer
            if source_layer < target_layer:
                violations.append({
                    'type': 'layer_violation',
                    'source': source,
                    'target': target,
                    'source_layer': source_layer,
                    'target_layer': target_layer,
                    'severity': 'high',
                    'description': f"Lower layer ({source_layer}) depends on higher layer ({target_layer})"
                })
        
        return violations
    
    def _get_module_layer(self, module_path: str, layer_hierarchy: Dict[str, int]) -> int:
        """Determine the architectural layer of a module"""
        path_parts = Path(module_path).parts
        
        # Check for layer indicators in path
        for part in reversed(path_parts):
            for layer, level in layer_hierarchy.items():
                if layer in part.lower():
                    return level
        
        # Default layer
        return 2
    
    def _detect_abstraction_violations(self) -> List[Dict[str, Any]]:
        """Detect abstraction principle violations"""
        violations = []
        
        # Find concrete classes depending on other concrete classes
        # This is a simplified check - in practice, would need more sophisticated analysis
        
        return violations  # Placeholder
    
    def _detect_coupling_violations(self) -> List[Dict[str, Any]]:
        """Detect excessive coupling violations"""
        violations = []
        
        for node_path in self.dependency_graph.nodes():
            fan_out = self.dependency_graph.out_degree(node_path)
            
            if fan_out > self.coupling_threshold:
                if node_path in self.module_map:
                    node = self.module_map[node_path]
                    violations.append({
                        'type': 'coupling_violation',
                        'module': node.name,
                        'path': node_path,
                        'coupling_count': fan_out,
                        'threshold': self.coupling_threshold,
                        'severity': 'high' if fan_out > self.coupling_threshold * 2 else 'medium',
                        'description': f"Module has {fan_out} dependencies (threshold: {self.coupling_threshold})"
                    })
        
        return violations
    
    def _prioritize_dependency_refactoring(self, circular_deps: List[CircularDependency]) -> List[Dict[str, Any]]:
        """Prioritize dependency refactoring tasks"""
        priorities = []
        
        # Add circular dependency fixes
        for cd in circular_deps:
            impact = {'critical': 10, 'high': 8, 'medium': 5, 'low': 2}[cd.severity]
            effort = 5 if cd.automated_fix_available else 8
            
            priorities.append({
                'type': 'circular_dependency_fix',
                'target': f"Cycle: {' -> '.join(cd.nodes[:3])}{'...' if len(cd.nodes) > 3 else ''}",
                'description': f"Fix circular dependency involving {len(cd.nodes)} modules",
                'impact': impact,
                'effort': effort,
                'automated_fix': cd.automated_fix_available,
                'strategies': cd.resolution_strategies[:2]
            })
        
        # Add high coupling fixes
        for node_path in self.dependency_graph.nodes():
            fan_out = self.dependency_graph.out_degree(node_path)
            
            if fan_out > self.coupling_threshold * 1.5:
                if node_path in self.module_map:
                    node = self.module_map[node_path]
                    priorities.append({
                        'type': 'coupling_reduction',
                        'target': f"{node.name} ({node_path})",
                        'description': f"Reduce coupling from {fan_out} dependencies",
                        'impact': min(8, int(fan_out / 2)),
                        'effort': 6,
                        'automated_fix': False,
                        'strategies': ['Extract interfaces', 'Use dependency injection', 'Split module responsibilities']
                    })
        
        return sorted(priorities, key=lambda x: (x['impact'], -x['effort']), reverse=True)
    
    def _load_import_patterns(self) -> Dict[str, str]:
        """Load patterns for import analysis"""
        return {
            'relative_import': r'^from\s+\.+',
            'absolute_import': r'^(from|import)\s+[a-zA-Z]',
            'standard_library': r'^(from|import)\s+(os|sys|json|re|datetime|collections)',
            'third_party': r'^(from|import)\s+[a-z_]+',
        }
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis"""
        file_str = str(file_path)
        return any(pattern in file_str for pattern in self.exclude_patterns)
    
    def _graph_to_dict(self) -> Dict[str, Any]:
        """Convert dependency graph to dictionary"""
        return {
            'nodes': [
                {
                    'id': node,
                    'name': self.module_map[node].name if node in self.module_map else Path(node).stem,
                    'type': self.module_map[node].type if node in self.module_map else 'module',
                    'file_path': node,
                    'fan_in': self.dependency_graph.in_degree(node),
                    'fan_out': self.dependency_graph.out_degree(node)
                }
                for node in self.dependency_graph.nodes()
            ],
            'edges': [
                {
                    'source': edge[0],
                    'target': edge[1],
                    'dependency': self.dependency_graph.edges[edge].get('dependency', '')
                }
                for edge in self.dependency_graph.edges()
            ]
        }
    
    def _circular_dep_to_dict(self, cd: CircularDependency) -> Dict[str, Any]:
        """Convert circular dependency to dictionary"""
        return {
            'cycle_id': cd.cycle_id,
            'nodes': cd.nodes,
            'severity': cd.severity,
            'impact_score': cd.impact_score,
            'resolution_strategies': cd.resolution_strategies,
            'automated_fix_available': cd.automated_fix_available,
            'affected_files': cd.affected_files,
            'cycle_length': cd.cycle_length,
            'entry_points': cd.entry_points
        }
    
    def _metrics_to_dict(self, metrics: DependencyMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'total_dependencies': metrics.total_dependencies,
            'circular_dependencies': metrics.circular_dependencies,
            'coupling_index': round(metrics.coupling_index, 2),
            'stability_index': round(metrics.stability_index, 2),
            'abstraction_level': round(metrics.abstraction_level, 2),
            'critical_path_length': metrics.critical_path_length,
            'dependency_clusters': len(metrics.dependency_clusters),
            'fan_in_stats': {
                'max': max(metrics.fan_in.values()) if metrics.fan_in else 0,
                'avg': sum(metrics.fan_in.values()) / len(metrics.fan_in) if metrics.fan_in else 0
            },
            'fan_out_stats': {
                'max': max(metrics.fan_out.values()) if metrics.fan_out else 0,
                'avg': sum(metrics.fan_out.values()) / len(metrics.fan_out) if metrics.fan_out else 0
            }
        }
    
    def _node_to_dict(self, node: DependencyNode) -> Dict[str, Any]:
        """Convert dependency node to dictionary"""
        return {
            'name': node.name,
            'type': node.type,
            'file_path': node.file_path,
            'line_number': node.line_number,
            'imports': node.imports,
            'exports': node.exports,
            'dependencies': node.dependencies,
            'dependents': node.dependents,
            'is_circular': node.is_circular,
            'circular_path': node.circular_path,
            'complexity_score': node.complexity_score,
            'coupling_score': node.coupling_score
        }
    
    async def fix_circular_dependency(self, cycle_id: str, strategy: str) -> Dict[str, Any]:
        """
        Attempt to fix a circular dependency
        
        Args:
            cycle_id: ID of the circular dependency to fix
            strategy: Resolution strategy to apply
            
        Returns:
            Result of the fix attempt
        """
        self.logger.info(f"Attempting to fix circular dependency {cycle_id} using strategy: {strategy}")
        
        # This would implement actual fixes based on the strategy
        # For now, return guidance for manual fixes
        
        return {
            'success': False,
            'message': f"Circular dependency fixes require manual intervention",
            'strategy': strategy,
            'next_steps': [
                "1. Identify the weakest link in the dependency chain",
                "2. Extract shared functionality to a common module",
                "3. Use interfaces or abstract base classes",
                "4. Consider using late imports (import inside functions)",
                "5. Run tests to verify the fix doesn't break functionality"
            ]
        }
    
    async def generate_dependency_diagram(self, output_format: str = 'png') -> str:
        """
        Generate a visual dependency diagram
        
        Args:
            output_format: Output format ('png', 'svg', 'html')
            
        Returns:
            Path to generated diagram file
        """
        self.logger.info(f"Generating dependency diagram in {output_format} format")
        
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Create a simplified graph for visualization
            plt.figure(figsize=(16, 12))
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(self.dependency_graph, k=1, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.dependency_graph, pos, 
                                 node_color='lightblue', 
                                 node_size=1000, 
                                 alpha=0.8)
            
            # Draw edges
            nx.draw_networkx_edges(self.dependency_graph, pos, 
                                 edge_color='gray', 
                                 arrows=True, 
                                 arrowsize=20, 
                                 alpha=0.6)
            
            # Draw labels
            labels = {node: self.module_map[node].name if node in self.module_map else Path(node).stem 
                     for node in self.dependency_graph.nodes()}
            nx.draw_networkx_labels(self.dependency_graph, pos, labels, font_size=8)
            
            plt.title("Dependency Graph", size=16)
            plt.axis('off')
            
            # Save diagram
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"dependency_diagram_{timestamp}.{output_format}"
            plt.savefig(output_path, format=output_format, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_path
            
        except ImportError:
            return "Error: matplotlib required for diagram generation"
        except Exception as e:
            self.logger.error(f"Error generating dependency diagram: {e}")
            return f"Error generating diagram: {str(e)}"