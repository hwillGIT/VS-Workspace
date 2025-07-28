"""
Architecture Diagram Manager

This agent automatically generates and maintains architecture diagrams including
component diagrams, dependency graphs, sequence diagrams, and system overviews.
"""

import ast
import re
import logging
import subprocess
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import graphviz
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml

from ...core.base.agent import BaseAgent


@dataclass
class Component:
    """Represents a system component"""
    name: str
    type: str  # 'module', 'class', 'service', 'database', 'external'
    description: str
    responsibilities: List[str]
    interfaces: List[str]
    dependencies: List[str]
    location: str
    complexity_score: float
    lines_of_code: int


@dataclass
class Dependency:
    """Represents a dependency relationship"""
    source: str
    target: str
    type: str  # 'imports', 'calls', 'inherits', 'uses', 'contains'
    strength: float  # 0.0 to 1.0
    description: str
    bi_directional: bool


@dataclass
class DiagramConfig:
    """Configuration for diagram generation"""
    diagram_type: str
    output_format: str  # 'png', 'svg', 'html', 'pdf'
    style: str  # 'modern', 'classic', 'minimal'
    include_labels: bool
    include_metrics: bool
    color_scheme: str
    layout_algorithm: str


class ArchitectureDiagramManager(BaseAgent):
    """
    Architecture Diagram Manager
    
    Automatically generates and maintains various types of architecture diagrams:
    - Component diagrams showing system structure
    - Dependency graphs showing relationships
    - Sequence diagrams showing interactions
    - System overview diagrams
    - Module hierarchy diagrams
    - Data flow diagrams
    - Deployment diagrams
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ArchitectureDiagramManager", config.get('diagram_manager', {}))
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.diagrams_dir = Path(config.get('diagrams_directory', 'docs/architecture/diagrams'))
        self.output_formats = config.get('output_formats', ['png', 'svg', 'html'])
        self.auto_update = config.get('auto_update_diagrams', True)
        self.include_metrics = config.get('include_metrics', True)
        
        # Diagram styles
        self.color_schemes = {
            'modern': {
                'primary': '#2E86AB',
                'secondary': '#A23B72',
                'accent': '#F18F01',
                'background': '#F8F9FA',
                'text': '#333333',
                'border': '#DEE2E6'
            },
            'classic': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'accent': '#2ca02c',
                'background': '#ffffff',
                'text': '#000000',
                'border': '#cccccc'
            },
            'minimal': {
                'primary': '#6c757d',
                'secondary': '#adb5bd',
                'accent': '#495057',
                'background': '#ffffff',
                'text': '#212529',
                'border': '#e9ecef'
            }
        }
        
        # Initialize directories
        self.diagrams_dir.mkdir(parents=True, exist_ok=True)
        
        # Diagram generators
        self.diagram_generators = {
            'component': self._generate_component_diagram,
            'dependency': self._generate_dependency_graph,
            'sequence': self._generate_sequence_diagram,
            'overview': self._generate_system_overview,
            'hierarchy': self._generate_module_hierarchy,
            'dataflow': self._generate_data_flow_diagram,
            'deployment': self._generate_deployment_diagram,
            'class': self._generate_class_diagram
        }
    
    async def generate_component_diagrams(self, component_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive component diagrams
        
        Args:
            component_path: Path to analyze
            
        Returns:
            Generated diagram information
        """
        self.logger.info(f"Generating component diagrams for {component_path}")
        
        # Analyze system components
        components = await self._analyze_components(component_path)
        dependencies = await self._analyze_dependencies(component_path)
        
        # Generate different types of diagrams
        diagrams_generated = {}
        
        for diagram_type, generator in self.diagram_generators.items():
            try:
                diagram_paths = await generator(components, dependencies, component_path)
                diagrams_generated[diagram_type] = diagram_paths
                self.logger.info(f"Generated {diagram_type} diagram")
            except Exception as e:
                self.logger.error(f"Failed to generate {diagram_type} diagram: {e}")
                diagrams_generated[diagram_type] = {'error': str(e)}
        
        # Generate interactive dashboard
        dashboard_path = await self._generate_interactive_dashboard(components, dependencies, component_path)
        diagrams_generated['dashboard'] = dashboard_path
        
        # Generate diagram index
        index_path = await self._generate_diagram_index(diagrams_generated, component_path)
        
        return {
            'components_analyzed': len(components),
            'dependencies_found': len(dependencies),
            'diagrams_generated': list(diagrams_generated.keys()),
            'diagram_paths': diagrams_generated,
            'index_path': index_path,
            'formats': self.output_formats
        }
    
    async def update_all_diagrams(self) -> Dict[str, str]:
        """
        Update all system architecture diagrams
        
        Returns:
            Paths to updated diagrams
        """
        self.logger.info("Updating all architecture diagrams")
        
        # Find all components to update
        components_to_update = self._find_components_to_update()
        
        updated_diagrams = {}
        
        for component_path in components_to_update:
            try:
                result = await self.generate_component_diagrams(component_path)
                component_name = Path(component_path).stem
                updated_diagrams[component_name] = result['diagram_paths']
            except Exception as e:
                self.logger.error(f"Failed to update diagrams for {component_path}: {e}")
        
        # Generate master system diagram
        master_diagram = await self._generate_master_system_diagram(updated_diagrams)
        updated_diagrams['master_system'] = master_diagram
        
        return updated_diagrams
    
    async def _analyze_components(self, target_path: str) -> List[Component]:
        """Analyze system components"""
        components = []
        path = Path(target_path)
        
        if path.is_file() and path.suffix == '.py':
            component = await self._analyze_single_file_component(path)
            if component:
                components.append(component)
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                if not self._should_skip_file(py_file):
                    component = await self._analyze_single_file_component(py_file)
                    if component:
                        components.append(component)
        
        return components
    
    async def _analyze_single_file_component(self, file_path: Path) -> Optional[Component]:
        """Analyze a single file as a component"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract component information
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            imports = self._extract_imports(tree)
            
            # Determine component type
            component_type = self._determine_component_type(file_path, classes, functions)
            
            # Calculate complexity
            complexity_score = self._calculate_component_complexity(tree)
            
            # Extract description from module docstring
            description = self._extract_module_docstring(tree) or f"Component: {file_path.stem}"
            
            # Identify responsibilities
            responsibilities = self._identify_responsibilities(classes, functions, content)
            
            # Identify interfaces (public methods/functions)
            interfaces = self._identify_interfaces(tree)
            
            return Component(
                name=file_path.stem,
                type=component_type,
                description=description,
                responsibilities=responsibilities,
                interfaces=interfaces,
                dependencies=imports,
                location=str(file_path),
                complexity_score=complexity_score,
                lines_of_code=len(content.split('\n'))
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing component {file_path}: {e}")
            return None
    
    async def _analyze_dependencies(self, target_path: str) -> List[Dependency]:
        """Analyze dependencies between components"""
        dependencies = []
        path = Path(target_path)
        
        # Build component map
        component_map = {}
        
        if path.is_dir():
            for py_file in path.rglob('*.py'):
                if not self._should_skip_file(py_file):
                    component_map[py_file.stem] = py_file
        
        # Analyze dependencies
        for component_name, component_path in component_map.items():
            try:
                with open(component_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                # Find imports and calls
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            target_name = alias.name.split('.')[0]
                            if target_name in component_map and target_name != component_name:
                                dependencies.append(Dependency(
                                    source=component_name,
                                    target=target_name,
                                    type='imports',
                                    strength=0.5,
                                    description=f"Imports {alias.name}",
                                    bi_directional=False
                                ))
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            target_name = node.module.split('.')[0]
                            if target_name in component_map and target_name != component_name:
                                dependencies.append(Dependency(
                                    source=component_name,
                                    target=target_name,
                                    type='imports',
                                    strength=0.7,
                                    description=f"Imports from {node.module}",
                                    bi_directional=False
                                ))
                    
                    elif isinstance(node, ast.Call):
                        # Analyze function calls to detect usage dependencies
                        if isinstance(node.func, ast.Attribute):
                            # Method calls
                            if isinstance(node.func.value, ast.Name):
                                target_name = node.func.value.id
                                if target_name in component_map and target_name != component_name:
                                    dependencies.append(Dependency(
                                        source=component_name,
                                        target=target_name,
                                        type='calls',
                                        strength=0.3,
                                        description=f"Calls {target_name}.{node.func.attr}",
                                        bi_directional=False
                                    ))
                    
                    elif isinstance(node, ast.ClassDef):
                        # Inheritance relationships
                        for base in node.bases:
                            if isinstance(base, ast.Name):
                                target_name = base.id
                                if target_name in component_map:
                                    dependencies.append(Dependency(
                                        source=component_name,
                                        target=target_name,
                                        type='inherits',
                                        strength=0.9,
                                        description=f"{node.name} inherits from {target_name}",
                                        bi_directional=False
                                    ))
                            
            except Exception as e:
                self.logger.error(f"Error analyzing dependencies for {component_path}: {e}")
        
        return dependencies
    
    async def _generate_component_diagram(self, components: List[Component], 
                                        dependencies: List[Dependency], 
                                        component_path: str) -> Dict[str, str]:
        """Generate component diagram"""
        component_name = Path(component_path).stem
        diagrams = {}
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        for component in components:
            G.add_node(component.name, 
                      type=component.type,
                      complexity=component.complexity_score,
                      loc=component.lines_of_code,
                      description=component.description)
        
        # Add edges
        for dep in dependencies:
            if dep.source in G.nodes and dep.target in G.nodes:
                G.add_edge(dep.source, dep.target,
                          type=dep.type,
                          strength=dep.strength,
                          description=dep.description)
        
        # Generate different formats
        if 'png' in self.output_formats:
            png_path = await self._generate_matplotlib_component_diagram(G, component_name)
            diagrams['png'] = png_path
        
        if 'svg' in self.output_formats:
            svg_path = await self._generate_graphviz_component_diagram(G, component_name)
            diagrams['svg'] = svg_path
        
        if 'html' in self.output_formats:
            html_path = await self._generate_plotly_component_diagram(G, components, dependencies, component_name)
            diagrams['html'] = html_path
        
        return diagrams
    
    async def _generate_matplotlib_component_diagram(self, G: nx.DiGraph, component_name: str) -> str:
        """Generate component diagram using matplotlib"""
        plt.figure(figsize=(16, 12))
        plt.clf()
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Color nodes by type
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            node_type = node_data.get('type', 'module')
            complexity = node_data.get('complexity', 1)
            
            # Color by type
            if node_type == 'service':
                node_colors.append('#2E86AB')
            elif node_type == 'class':
                node_colors.append('#A23B72')
            elif node_type == 'module':
                node_colors.append('#F18F01')
            else:
                node_colors.append('#6c757d')
            
            # Size by complexity
            node_sizes.append(max(300, complexity * 100))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              alpha=0.8)
        
        # Draw edges with different styles for different types
        edge_types = set([G.edges[edge].get('type', 'default') for edge in G.edges()])
        
        for edge_type in edge_types:
            edges_of_type = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == edge_type]
            
            if edge_type == 'inherits':
                nx.draw_networkx_edges(G, pos, edgelist=edges_of_type,
                                     edge_color='red', width=2, alpha=0.7,
                                     arrowsize=20, arrowstyle='->')
            elif edge_type == 'imports':
                nx.draw_networkx_edges(G, pos, edgelist=edges_of_type,
                                     edge_color='blue', width=1, alpha=0.5,
                                     arrowsize=15, arrowstyle='->')
            else:
                nx.draw_networkx_edges(G, pos, edgelist=edges_of_type,
                                     edge_color='gray', width=0.5, alpha=0.3,
                                     arrowsize=10, arrowstyle='->')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title(f"Component Diagram: {component_name}", fontsize=16, fontweight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', markersize=10, label='Service'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#A23B72', markersize=10, label='Class'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#F18F01', markersize=10, label='Module'),
            plt.Line2D([0], [0], color='red', linewidth=2, label='Inheritance'),
            plt.Line2D([0], [0], color='blue', linewidth=1, label='Import'),
            plt.Line2D([0], [0], color='gray', linewidth=0.5, label='Usage')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save diagram
        output_path = self.diagrams_dir / f"{component_name}_component_diagram.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    async def _generate_graphviz_component_diagram(self, G: nx.DiGraph, component_name: str) -> str:
        """Generate component diagram using Graphviz"""
        dot = graphviz.Digraph(comment=f'Component Diagram: {component_name}')
        dot.attr(rankdir='TB', splines='ortho', nodesep='1', ranksep='1')
        
        # Add nodes
        for node in G.nodes():
            node_data = G.nodes[node]
            node_type = node_data.get('type', 'module')
            complexity = node_data.get('complexity', 1)
            loc = node_data.get('loc', 0)
            
            # Style by type
            if node_type == 'service':
                dot.node(node, node, shape='box', style='filled', fillcolor='lightblue')
            elif node_type == 'class':
                dot.node(node, node, shape='record', style='filled', fillcolor='lightcoral')
            elif node_type == 'module':
                dot.node(node, node, shape='ellipse', style='filled', fillcolor='lightyellow')
            else:
                dot.node(node, node, shape='box', style='filled', fillcolor='lightgray')
        
        # Add edges
        for edge in G.edges(data=True):
            source, target, data = edge
            edge_type = data.get('type', 'default')
            
            if edge_type == 'inherits':
                dot.edge(source, target, color='red', style='bold', arrowhead='empty')
            elif edge_type == 'imports':
                dot.edge(source, target, color='blue', style='solid')
            else:
                dot.edge(source, target, color='gray', style='dashed')
        
        # Save diagram
        output_path = self.diagrams_dir / f"{component_name}_component_diagram"
        dot.render(output_path, format='svg', cleanup=True)
        
        return f"{output_path}.svg"
    
    async def _generate_plotly_component_diagram(self, G: nx.DiGraph, components: List[Component], 
                                               dependencies: List[Dependency], component_name: str) -> str:
        """Generate interactive component diagram using Plotly"""
        # Use networkx layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare node traces
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            marker=dict(
                size=[max(20, G.nodes[node].get('complexity', 1) * 10) for node in G.nodes()],
                color=[hash(G.nodes[node].get('type', 'module')) % 10 for node in G.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Component Type"),
                line=dict(width=2, color='rgb(50,50,50)')
            ),
            text=[node for node in G.nodes()],
            textposition="middle center",
            hovertemplate='<b>%{text}</b><br>' +
                         'Type: %{customdata[0]}<br>' +
                         'Complexity: %{customdata[1]}<br>' +
                         'Lines of Code: %{customdata[2]}<br>' +
                         '<extra></extra>',
            customdata=[[G.nodes[node].get('type', 'module'),
                        G.nodes[node].get('complexity', 1),
                        G.nodes[node].get('loc', 0)] for node in G.nodes()],
            name='Components'
        )
        
        # Prepare edge traces
        edge_traces = []
        
        for edge in G.edges(data=True):
            source, target, data = edge
            x0, y0 = pos[source]
            x1, y1 = pos[target]
            
            edge_type = data.get('type', 'default')
            strength = data.get('strength', 0.5)
            
            # Color and style by type
            if edge_type == 'inherits':
                color = 'red'
                width = 3
            elif edge_type == 'imports':
                color = 'blue'
                width = 2
            else:
                color = 'gray'
                width = 1
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)
        
        fig.update_layout(
            title=f"Interactive Component Diagram: {component_name}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Component relationships and dependencies",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="#999999", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        # Save diagram
        output_path = self.diagrams_dir / f"{component_name}_component_diagram.html"
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    async def _generate_dependency_graph(self, components: List[Component], 
                                       dependencies: List[Dependency], 
                                       component_path: str) -> Dict[str, str]:
        """Generate dependency graph"""
        component_name = Path(component_path).stem
        diagrams = {}
        
        # Create dependency matrix visualization
        if 'html' in self.output_formats:
            html_path = await self._generate_dependency_matrix(components, dependencies, component_name)
            diagrams['html'] = html_path
        
        # Create circular dependency detection diagram
        if 'png' in self.output_formats:
            png_path = await self._generate_circular_dependency_diagram(dependencies, component_name)
            diagrams['png'] = png_path
        
        return diagrams
    
    async def _generate_dependency_matrix(self, components: List[Component], 
                                        dependencies: List[Dependency], 
                                        component_name: str) -> str:
        """Generate dependency matrix heatmap"""
        # Create dependency matrix
        component_names = [comp.name for comp in components]
        n = len(component_names)
        
        # Initialize matrix
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        # Fill matrix with dependency strengths
        name_to_index = {name: i for i, name in enumerate(component_names)}
        
        for dep in dependencies:
            if dep.source in name_to_index and dep.target in name_to_index:
                i = name_to_index[dep.source]
                j = name_to_index[dep.target]
                matrix[i][j] = dep.strength
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            x=component_names,
            y=component_names,
            z=matrix,
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="Dependency Strength")
        ))
        
        fig.update_layout(
            title=f"Dependency Matrix: {component_name}",
            xaxis_title="Target Component",
            yaxis_title="Source Component",
            width=800,
            height=800
        )
        
        # Save diagram
        output_path = self.diagrams_dir / f"{component_name}_dependency_matrix.html"
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    async def _generate_circular_dependency_diagram(self, dependencies: List[Dependency], 
                                                  component_name: str) -> str:
        """Generate circular dependency detection diagram"""
        # Build graph to detect cycles
        G = nx.DiGraph()
        
        for dep in dependencies:
            G.add_edge(dep.source, dep.target)
        
        # Find strongly connected components (cycles)
        cycles = list(nx.strongly_connected_components(G))
        cycles = [cycle for cycle in cycles if len(cycle) > 1]  # Only actual cycles
        
        plt.figure(figsize=(12, 8))
        plt.clf()
        
        # Layout
        pos = nx.circular_layout(G)
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.7)
        
        # Draw all edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3, arrows=True)
        
        # Highlight cycles
        colors = plt.cm.Set3(range(len(cycles)))
        for i, cycle in enumerate(cycles):
            cycle_nodes = list(cycle)
            nx.draw_networkx_nodes(G, pos, nodelist=cycle_nodes, 
                                 node_color=[colors[i]], node_size=700, alpha=0.8)
            
            # Draw cycle edges
            cycle_edges = [(u, v) for u, v in G.edges() if u in cycle and v in cycle]
            nx.draw_networkx_edges(G, pos, edgelist=cycle_edges,
                                 edge_color=colors[i], width=3, alpha=0.8, arrows=True)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        plt.title(f"Circular Dependencies: {component_name}\n{len(cycles)} cycles detected", 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save diagram
        output_path = self.diagrams_dir / f"{component_name}_circular_dependencies.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    async def _generate_sequence_diagram(self, components: List[Component], 
                                       dependencies: List[Dependency], 
                                       component_path: str) -> Dict[str, str]:
        """Generate sequence diagram showing component interactions"""
        component_name = Path(component_path).stem
        diagrams = {}
        
        # Create PlantUML sequence diagram
        if 'png' in self.output_formats:
            png_path = await self._generate_plantuml_sequence_diagram(components, dependencies, component_name)
            diagrams['png'] = png_path
        
        return diagrams
    
    async def _generate_plantuml_sequence_diagram(self, components: List[Component], 
                                                dependencies: List[Dependency], 
                                                component_name: str) -> str:
        """Generate sequence diagram using PlantUML format"""
        # Generate PlantUML code
        plantuml_code = ["@startuml"]
        plantuml_code.append(f"title Sequence Diagram: {component_name}")
        plantuml_code.append("")
        
        # Add participants
        for component in components[:10]:  # Limit to avoid cluttering
            if component.type == 'service':
                plantuml_code.append(f'participant "{component.name}" as {component.name}')
            else:
                plantuml_code.append(f'actor "{component.name}" as {component.name}')
        
        plantuml_code.append("")
        
        # Add interactions based on dependencies
        for dep in dependencies[:20]:  # Limit interactions
            if dep.type in ['calls', 'uses']:
                plantuml_code.append(f'{dep.source} -> {dep.target}: {dep.description}')
        
        plantuml_code.append("@enduml")
        
        # Save PlantUML file
        plantuml_file = self.diagrams_dir / f"{component_name}_sequence.puml"
        with open(plantuml_file, 'w') as f:
            f.write('\n'.join(plantuml_code))
        
        # Try to generate PNG if PlantUML is available
        output_path = self.diagrams_dir / f"{component_name}_sequence_diagram.png"
        try:
            subprocess.run(['plantuml', '-tpng', str(plantuml_file)], 
                          check=True, capture_output=True)
            if (self.diagrams_dir / f"{component_name}_sequence.png").exists():
                (self.diagrams_dir / f"{component_name}_sequence.png").rename(output_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            # PlantUML not available, return PlantUML file path
            self.logger.warning("PlantUML not available, returning .puml file")
            output_path = plantuml_file
        
        return str(output_path)
    
    async def _generate_system_overview(self, components: List[Component], 
                                      dependencies: List[Dependency], 
                                      component_path: str) -> Dict[str, str]:
        """Generate high-level system overview diagram"""
        component_name = Path(component_path).stem
        diagrams = {}
        
        # Create system overview
        if 'html' in self.output_formats:
            html_path = await self._generate_system_overview_plotly(components, dependencies, component_name)
            diagrams['html'] = html_path
        
        return diagrams
    
    async def _generate_system_overview_plotly(self, components: List[Component], 
                                             dependencies: List[Dependency], 
                                             component_name: str) -> str:
        """Generate system overview using Plotly"""
        # Group components by type
        component_groups = {}
        for comp in components:
            comp_type = comp.type
            if comp_type not in component_groups:
                component_groups[comp_type] = []
            component_groups[comp_type].append(comp)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Component Distribution", "Complexity Analysis", 
                          "Dependency Strength", "Lines of Code"),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Component distribution pie chart
        fig.add_trace(
            go.Pie(
                labels=list(component_groups.keys()),
                values=[len(comps) for comps in component_groups.values()],
                name="Component Types"
            ),
            row=1, col=1
        )
        
        # Complexity bar chart
        fig.add_trace(
            go.Bar(
                x=[comp.name for comp in components[:10]],
                y=[comp.complexity_score for comp in components[:10]],
                name="Complexity Scores"
            ),
            row=1, col=2
        )
        
        # Dependency strength scatter
        fig.add_trace(
            go.Scatter(
                x=[dep.strength for dep in dependencies],
                y=[hash(dep.type) % 10 for dep in dependencies],
                mode='markers',
                name="Dependencies",
                text=[f"{dep.source} -> {dep.target}" for dep in dependencies],
                hovertemplate='<b>%{text}</b><br>Strength: %{x}<br><extra></extra>'
            ),
            row=2, col=1
        )
        
        # Lines of code histogram
        fig.add_trace(
            go.Histogram(
                x=[comp.lines_of_code for comp in components],
                name="Lines of Code Distribution"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f"System Overview: {component_name}",
            showlegend=False,
            height=800
        )
        
        # Save diagram
        output_path = self.diagrams_dir / f"{component_name}_system_overview.html"
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    async def _generate_module_hierarchy(self, components: List[Component], 
                                       dependencies: List[Dependency], 
                                       component_path: str) -> Dict[str, str]:
        """Generate module hierarchy diagram"""
        component_name = Path(component_path).stem
        diagrams = {}
        
        # Create hierarchy tree
        if 'html' in self.output_formats:
            html_path = await self._generate_hierarchy_tree(components, dependencies, component_name)
            diagrams['html'] = html_path
        
        return diagrams
    
    async def _generate_hierarchy_tree(self, components: List[Component], 
                                     dependencies: List[Dependency], 
                                     component_name: str) -> str:
        """Generate hierarchy tree visualization"""
        # Build hierarchy based on directory structure and dependencies
        hierarchy = {}
        
        for comp in components:
            path_parts = Path(comp.location).parts
            current_level = hierarchy
            
            for part in path_parts[:-1]:  # Exclude filename
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
            
            # Add the component
            current_level[comp.name] = {
                'type': comp.type,
                'complexity': comp.complexity_score,
                'loc': comp.lines_of_code
            }
        
        # Create tree visualization using Plotly
        # This is a simplified version - could be enhanced with more sophisticated tree layouts
        fig = go.Figure()
        
        # For now, create a simple representation
        component_names = [comp.name for comp in components]
        complexities = [comp.complexity_score for comp in components]
        
        fig.add_trace(go.Scatter(
            x=list(range(len(component_names))),
            y=complexities,
            mode='markers+text',
            text=component_names,
            textposition="top center",
            marker=dict(
                size=[max(10, comp.lines_of_code / 10) for comp in components],
                color=complexities,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Complexity")
            ),
            name="Components"
        ))
        
        fig.update_layout(
            title=f"Module Hierarchy: {component_name}",
            xaxis_title="Component Index",
            yaxis_title="Complexity Score",
            showlegend=False
        )
        
        # Save diagram
        output_path = self.diagrams_dir / f"{component_name}_hierarchy.html"
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    async def _generate_data_flow_diagram(self, components: List[Component], 
                                        dependencies: List[Dependency], 
                                        component_path: str) -> Dict[str, str]:
        """Generate data flow diagram"""
        component_name = Path(component_path).stem
        diagrams = {}
        
        # Create data flow visualization
        if 'png' in self.output_formats:
            png_path = await self._generate_data_flow_matplotlib(components, dependencies, component_name)
            diagrams['png'] = png_path
        
        return diagrams
    
    async def _generate_data_flow_matplotlib(self, components: List[Component], 
                                           dependencies: List[Dependency], 
                                           component_name: str) -> str:
        """Generate data flow diagram using matplotlib"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create directed graph for data flow
        G = nx.DiGraph()
        
        # Add nodes
        for comp in components:
            G.add_node(comp.name, type=comp.type)
        
        # Add edges (data flows)
        for dep in dependencies:
            if dep.type in ['calls', 'uses', 'imports']:
                G.add_edge(dep.source, dep.target, weight=dep.strength)
        
        # Use hierarchical layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes with different shapes for different types
        for node_type, color in [('service', 'lightblue'), ('class', 'lightcoral'), 
                                ('module', 'lightgreen'), ('database', 'yellow')]:
            nodes_of_type = [node for node in G.nodes() if G.nodes[node].get('type') == node_type]
            if nodes_of_type:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type,
                                     node_color=color, node_size=800, alpha=0.8)
        
        # Draw edges with varying thickness based on weight
        edges = G.edges(data=True)
        weights = [edge[2]['weight'] for edge in edges]
        max_weight = max(weights) if weights else 1
        
        for edge in edges:
            source, target, data = edge
            weight = data['weight']
            width = (weight / max_weight) * 5 + 1
            
            nx.draw_networkx_edges(G, pos, edgelist=[(source, target)],
                                 width=width, alpha=0.6, edge_color='gray',
                                 arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        ax.set_title(f"Data Flow Diagram: {component_name}", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save diagram
        output_path = self.diagrams_dir / f"{component_name}_data_flow.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
    
    async def _generate_deployment_diagram(self, components: List[Component], 
                                         dependencies: List[Dependency], 
                                         component_path: str) -> Dict[str, str]:
        """Generate deployment diagram"""
        component_name = Path(component_path).stem
        diagrams = {}
        
        # Create deployment view
        if 'html' in self.output_formats:
            html_path = await self._generate_deployment_view(components, component_name)
            diagrams['html'] = html_path
        
        return diagrams
    
    async def _generate_deployment_view(self, components: List[Component], component_name: str) -> str:
        """Generate deployment view diagram"""
        # Group components by deployment location (simplified)
        deployment_groups = {
            'Frontend': [comp for comp in components if 'ui' in comp.name.lower() or 'view' in comp.name.lower()],
            'Backend': [comp for comp in components if 'service' in comp.type or 'api' in comp.name.lower()],
            'Database': [comp for comp in components if 'model' in comp.name.lower() or 'data' in comp.name.lower()],
            'External': [comp for comp in components if 'client' in comp.name.lower() or 'external' in comp.name.lower()]
        }
        
        # Remove empty groups
        deployment_groups = {k: v for k, v in deployment_groups.items() if v}
        
        # Create deployment diagram
        fig = go.Figure()
        
        # Position groups
        group_positions = {
            'Frontend': (0, 3),
            'Backend': (1, 2),
            'Database': (2, 1),
            'External': (3, 0)
        }
        
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
        
        for i, (group_name, components_in_group) in enumerate(deployment_groups.items()):
            if group_name in group_positions:
                x, y = group_positions[group_name]
                
                # Add group box
                fig.add_shape(
                    type="rect",
                    x0=x-0.4, y0=y-0.4,
                    x1=x+0.4, y1=y+0.4,
                    fillcolor=colors[i % len(colors)],
                    opacity=0.3,
                    line=dict(color="black", width=2)
                )
                
                # Add group label
                fig.add_annotation(
                    x=x, y=y+0.3,
                    text=f"<b>{group_name}</b>",
                    showarrow=False,
                    font=dict(size=14, color="black")
                )
                
                # Add components in group
                for j, comp in enumerate(components_in_group[:5]):  # Limit to 5 per group
                    comp_y = y - 0.2 + (j * 0.1)
                    fig.add_annotation(
                        x=x, y=comp_y,
                        text=comp.name,
                        showarrow=False,
                        font=dict(size=10, color="darkblue")
                    )
        
        fig.update_layout(
            title=f"Deployment Diagram: {component_name}",
            xaxis=dict(range=[-0.5, 3.5], showgrid=False, showticklabels=False),
            yaxis=dict(range=[-0.5, 3.5], showgrid=False, showticklabels=False),
            showlegend=False,
            width=800,
            height=600
        )
        
        # Save diagram
        output_path = self.diagrams_dir / f"{component_name}_deployment.html"
        fig.write_html(str(output_path))
        
        return str(output_path)
    
    async def _generate_class_diagram(self, components: List[Component], 
                                    dependencies: List[Dependency], 
                                    component_path: str) -> Dict[str, str]:
        """Generate UML class diagram"""
        component_name = Path(component_path).stem
        diagrams = {}
        
        # Generate class diagram using PlantUML format
        if 'png' in self.output_formats:
            png_path = await self._generate_plantuml_class_diagram(components, dependencies, component_name)
            diagrams['png'] = png_path
        
        return diagrams
    
    async def _generate_plantuml_class_diagram(self, components: List[Component], 
                                             dependencies: List[Dependency], 
                                             component_name: str) -> str:
        """Generate UML class diagram using PlantUML"""
        plantuml_code = ["@startuml"]
        plantuml_code.append(f"title Class Diagram: {component_name}")
        plantuml_code.append("")
        
        # Add classes
        for comp in components:
            if comp.type == 'class':
                plantuml_code.append(f"class {comp.name} {{")
                
                # Add methods from interfaces
                for interface in comp.interfaces[:5]:  # Limit to avoid cluttering
                    plantuml_code.append(f"  +{interface}()")
                
                plantuml_code.append("}")
                plantuml_code.append("")
        
        # Add relationships
        for dep in dependencies:
            if dep.type == 'inherits':
                plantuml_code.append(f"{dep.target} <|-- {dep.source}")
            elif dep.type == 'uses':
                plantuml_code.append(f"{dep.source} --> {dep.target}")
        
        plantuml_code.append("@enduml")
        
        # Save PlantUML file
        plantuml_file = self.diagrams_dir / f"{component_name}_class.puml"
        with open(plantuml_file, 'w') as f:
            f.write('\n'.join(plantuml_code))
        
        # Try to generate PNG
        output_path = self.diagrams_dir / f"{component_name}_class_diagram.png"
        try:
            subprocess.run(['plantuml', '-tpng', str(plantuml_file)], 
                          check=True, capture_output=True)
            if (self.diagrams_dir / f"{component_name}_class.png").exists():
                (self.diagrams_dir / f"{component_name}_class.png").rename(output_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.warning("PlantUML not available, returning .puml file")
            output_path = plantuml_file
        
        return str(output_path)
    
    async def _generate_interactive_dashboard(self, components: List[Component], 
                                            dependencies: List[Dependency], 
                                            component_path: str) -> str:
        """Generate interactive dashboard with all diagrams"""
        component_name = Path(component_path).stem
        
        # Create comprehensive dashboard HTML
        dashboard_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Architecture Dashboard: {component_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat-box {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .diagram-section {{ margin: 30px 0; }}
        .diagram-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .diagram-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2E86AB; }}
        h2 {{ color: #A23B72; }}
        .metric {{ font-size: 24px; font-weight: bold; color: #F18F01; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Architecture Dashboard: {component_name}</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <h3>Components</h3>
                <div class="metric">{len(components)}</div>
            </div>
            <div class="stat-box">
                <h3>Dependencies</h3>
                <div class="metric">{len(dependencies)}</div>
            </div>
            <div class="stat-box">
                <h3>Avg Complexity</h3>
                <div class="metric">{sum(c.complexity_score for c in components) / len(components) if components else 0:.1f}</div>
            </div>
            <div class="stat-box">
                <h3>Total LOC</h3>
                <div class="metric">{sum(c.lines_of_code for c in components):,}</div>
            </div>
        </div>
        
        <div class="diagram-section">
            <h2>Architecture Diagrams</h2>
            <div class="diagram-grid">
                <div class="diagram-card">
                    <h3>Component Overview</h3>
                    <div id="component-chart"></div>
                </div>
                <div class="diagram-card">
                    <h3>Complexity Distribution</h3>
                    <div id="complexity-chart"></div>
                </div>
                <div class="diagram-card">
                    <h3>Dependency Network</h3>
                    <div id="dependency-chart"></div>
                </div>
                <div class="diagram-card">
                    <h3>Component Types</h3>
                    <div id="types-chart"></div>
                </div>
            </div>
        </div>
        
        <div class="diagram-section">
            <h2>Component Details</h2>
            <div class="diagram-card">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background-color: #f8f9fa;">
                            <th style="padding: 10px; border: 1px solid #dee2e6;">Component</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6;">Type</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6;">Complexity</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6;">LOC</th>
                            <th style="padding: 10px; border: 1px solid #dee2e6;">Dependencies</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add component details
        for comp in components:
            comp_deps = len([d for d in dependencies if d.source == comp.name])
            dashboard_html += f"""
                        <tr>
                            <td style="padding: 10px; border: 1px solid #dee2e6;">{comp.name}</td>
                            <td style="padding: 10px; border: 1px solid #dee2e6;">{comp.type}</td>
                            <td style="padding: 10px; border: 1px solid #dee2e6;">{comp.complexity_score:.1f}</td>
                            <td style="padding: 10px; border: 1px solid #dee2e6;">{comp.lines_of_code}</td>
                            <td style="padding: 10px; border: 1px solid #dee2e6;">{comp_deps}</td>
                        </tr>
            """
        
        # Add JavaScript for charts
        dashboard_html += """
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        // Component overview chart
        var componentData = [{
            values: """ + str([len([c for c in components if c.type == t]) for t in set(c.type for c in components)]) + f""",
            labels: """ + str(list(set(c.type for c in components))) + f""",
            type: 'pie',
            hole: 0.4
        }}];
        
        Plotly.newPlot('component-chart', componentData, {{title: 'Components by Type'}});
        
        // Complexity distribution
        var complexityData = [{{
            x: """ + str([c.name for c in components[:10]]) + f""",
            y: """ + str([c.complexity_score for c in components[:10]]) + f""",
            type: 'bar',
            marker: {{color: '#2E86AB'}}
        }}];
        
        Plotly.newPlot('complexity-chart', complexityData, {{title: 'Complexity Scores'}});
        
        // Dependency strength
        var dependencyData = [{{
            x: """ + str([d.strength for d in dependencies]) + f""",
            type: 'histogram',
            marker: {{color: '#A23B72'}}
        }}];
        
        Plotly.newPlot('dependency-chart', dependencyData, {{title: 'Dependency Strength Distribution'}});
        
        // Component types pie
        var typesData = [{
            values: """ + str([len([c for c in components if c.type == t]) for t in set(c.type for c in components)]) + f""",
            labels: """ + str(list(set(c.type for c in components))) + f""",
            type: 'pie'
        }}];
        
        Plotly.newPlot('types-chart', typesData, {{title: 'Component Distribution'}});
    </script>
</body>
</html>
        """
        
        # Save dashboard
        output_path = self.diagrams_dir / f"{component_name}_dashboard.html"
        with open(output_path, 'w') as f:
            f.write(dashboard_html)
        
        return str(output_path)
    
    async def _generate_diagram_index(self, diagrams_generated: Dict[str, Any], component_path: str) -> str:
        """Generate index of all diagrams"""
        component_name = Path(component_path).stem
        
        index_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Architecture Diagrams Index: {component_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .diagram-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
        .diagram-card {{ border: 1px solid #ddd; padding: 20px; border-radius: 8px; }}
        .diagram-card h3 {{ margin-top: 0; color: #2E86AB; }}
        .diagram-link {{ display: block; margin: 5px 0; color: #A23B72; text-decoration: none; }}
        .diagram-link:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Architecture Diagrams: {component_name}</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="diagram-grid">
        """
        
        diagram_descriptions = {
            'component': 'Component relationships and structure',
            'dependency': 'Dependency analysis and circular dependencies',
            'sequence': 'Sequence of component interactions',
            'overview': 'High-level system overview',
            'hierarchy': 'Module hierarchy and organization',
            'dataflow': 'Data flow between components',
            'deployment': 'Deployment architecture view',
            'class': 'UML class diagram',
            'dashboard': 'Interactive architecture dashboard'
        }
        
        for diagram_type, paths in diagrams_generated.items():
            if isinstance(paths, dict) and 'error' not in paths:
                index_html += f"""
        <div class="diagram-card">
            <h3>{diagram_type.title()} Diagram</h3>
            <p>{diagram_descriptions.get(diagram_type, 'Architectural diagram')}</p>
                """
                
                for format_type, path in paths.items():
                    if isinstance(path, str):
                        filename = Path(path).name
                        index_html += f'<a href="{filename}" class="diagram-link">{format_type.upper()} Version</a>'
                
                index_html += "</div>"
        
        index_html += """
    </div>
</body>
</html>
        """
        
        # Save index
        index_path = self.diagrams_dir / f"{component_name}_diagrams_index.html"
        with open(index_path, 'w') as f:
            f.write(index_html)
        
        return str(index_path)
    
    async def _generate_master_system_diagram(self, all_diagrams: Dict[str, Any]) -> str:
        """Generate master system diagram combining all components"""
        self.logger.info("Generating master system diagram")
        
        # Create master overview
        master_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Master System Architecture</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .component-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px; }}
        .component-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .component-card h3 {{ margin-top: 0; color: #2E86AB; }}
        .diagram-link {{ display: block; margin: 5px 0; color: #A23B72; text-decoration: none; }}
        .diagram-link:hover {{ text-decoration: underline; }}
        h1 {{ color: #2E86AB; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Master System Architecture</h1>
            <p>Complete architectural overview of all system components</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="component-grid">
        """
        
        for component_name, diagrams in all_diagrams.items():
            if component_name != 'master_system':
                master_html += f"""
            <div class="component-card">
                <h3>{component_name.replace('_', ' ').title()}</h3>
                <p>Architectural diagrams and analysis</p>
                """
                
                if isinstance(diagrams, dict):
                    for diagram_type, paths in diagrams.items():
                        if isinstance(paths, dict) and 'error' not in paths:
                            for format_type, path in paths.items():
                                if isinstance(path, str):
                                    filename = Path(path).name
                                    master_html += f'<a href="{filename}" class="diagram-link">{diagram_type.title()} ({format_type.upper()})</a>'
                
                master_html += "</div>"
        
        master_html += """
        </div>
    </div>
</body>
</html>
        """
        
        # Save master diagram
        master_path = self.diagrams_dir / "master_system_architecture.html"
        with open(master_path, 'w') as f:
            f.write(master_html)
        
        return str(master_path)
    
    # Helper methods
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract imports from AST"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    
    def _determine_component_type(self, file_path: Path, classes: List[str], functions: List[str]) -> str:
        """Determine component type based on analysis"""
        filename = file_path.name.lower()
        
        if 'service' in filename or 'api' in filename:
            return 'service'
        elif 'model' in filename or 'schema' in filename:
            return 'model'
        elif 'view' in filename or 'ui' in filename:
            return 'view'
        elif 'test' in filename:
            return 'test'
        elif classes:
            return 'class'
        elif functions:
            return 'module'
        else:
            return 'utility'
    
    def _extract_module_docstring(self, tree: ast.AST) -> str:
        """Extract module docstring"""
        if (isinstance(tree, ast.Module) and tree.body and 
            isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, (ast.Str, ast.Constant))):
            if isinstance(tree.body[0].value, ast.Str):
                return tree.body[0].value.s.strip()
            elif isinstance(tree.body[0].value, ast.Constant) and isinstance(tree.body[0].value.value, str):
                return tree.body[0].value.value.strip()
        return ""
    
    def _identify_responsibilities(self, classes: List[str], functions: List[str], content: str) -> List[str]:
        """Identify component responsibilities"""
        responsibilities = []
        content_lower = content.lower()
        
        # Analyze based on keywords and patterns
        if 'database' in content_lower or 'db' in content_lower:
            responsibilities.append('Data Management')
        
        if 'api' in content_lower or 'endpoint' in content_lower:
            responsibilities.append('API Services')
        
        if 'validate' in content_lower or 'check' in content_lower:
            responsibilities.append('Validation')
        
        if 'process' in content_lower or 'transform' in content_lower:
            responsibilities.append('Data Processing')
        
        if 'log' in content_lower or 'audit' in content_lower:
            responsibilities.append('Logging')
        
        if not responsibilities:
            responsibilities.append('General Processing')
        
        return responsibilities
    
    def _identify_interfaces(self, tree: ast.AST) -> List[str]:
        """Identify public interfaces (methods/functions)"""
        interfaces = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):  # Public functions
                    interfaces.append(node.name)
            elif isinstance(node, ast.ClassDef):
                for class_node in node.body:
                    if isinstance(class_node, ast.FunctionDef) and not class_node.name.startswith('_'):
                        interfaces.append(f"{node.name}.{class_node.name}")
        
        return interfaces
    
    def _calculate_component_complexity(self, tree: ast.AST) -> float:
        """Calculate component complexity score"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 0.5
            elif isinstance(node, ast.ClassDef):
                complexity += 1
        
        return min(10.0, complexity / 10)
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            '__pycache__',
            '.git',
            'test_',
            '_test.py',
            'tests/',
            'venv/',
            '.venv/',
        ]
        
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    def _find_components_to_update(self) -> List[str]:
        """Find components that need diagram updates"""
        # This would normally check for recent changes
        # For now, return common component directories
        components = []
        
        base_paths = [
            'agents',
            'core',
            'backtest',
            'examples'
        ]
        
        for base_path in base_paths:
            path = Path(base_path)
            if path.exists():
                components.append(str(path))
        
        return components