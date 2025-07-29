"""
Architecture Knowledge Graph

Simple in-memory knowledge graph for architectural patterns and relationships.
This is a lightweight alternative to Neo4j for basic operations.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict


@dataclass
class Node:
    """Represents a node in the knowledge graph"""
    id: str
    type: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Edge:
    """Represents an edge/relationship in the knowledge graph"""
    id: str
    type: str
    from_node: str
    to_node: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class ArchitectureKnowledgeGraph:
    """
    Lightweight in-memory knowledge graph for architectural concepts.
    
    For production use, this would be replaced with Neo4j integration,
    but provides a simple alternative for testing and basic operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.node_index: Dict[str, Set[str]] = defaultdict(set)  # type -> node_ids
        self.edge_index: Dict[str, Set[str]] = defaultdict(set)  # type -> edge_ids
        self._id_counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique ID"""
        self._id_counter += 1
        return f"id_{self._id_counter}"
    
    def add_node(self, node_type: str, name: str, properties: Dict[str, Any] = None) -> str:
        """Add a node to the graph"""
        node_id = self._generate_id()
        node = Node(
            id=node_id,
            type=node_type,
            name=name,
            properties=properties or {}
        )
        
        self.nodes[node_id] = node
        self.node_index[node_type].add(node_id)
        
        self.logger.debug(f"Added node: {node_type} - {name} (ID: {node_id})")
        return node_id
    
    def add_edge(
        self,
        edge_type: str,
        from_node_id: str,
        to_node_id: str,
        properties: Dict[str, Any] = None
    ) -> str:
        """Add an edge between two nodes"""
        
        # Validate nodes exist
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("Both nodes must exist before creating edge")
        
        edge_id = self._generate_id()
        edge = Edge(
            id=edge_id,
            type=edge_type,
            from_node=from_node_id,
            to_node=to_node_id,
            properties=properties or {}
        )
        
        self.edges[edge_id] = edge
        self.edge_index[edge_type].add(edge_id)
        
        self.logger.debug(f"Added edge: {edge_type} from {from_node_id} to {to_node_id}")
        return edge_id
    
    def find_nodes_by_type(self, node_type: str) -> List[Node]:
        """Find all nodes of a specific type"""
        node_ids = self.node_index.get(node_type, set())
        return [self.nodes[node_id] for node_id in node_ids]
    
    def find_node_by_name(self, name: str, node_type: Optional[str] = None) -> Optional[Node]:
        """Find a node by name, optionally filtered by type"""
        for node in self.nodes.values():
            if node.name == name:
                if node_type is None or node.type == node_type:
                    return node
        return None
    
    def get_relationships(self, node_id: str, edge_type: Optional[str] = None) -> List[Tuple[Edge, Node]]:
        """Get all relationships for a node"""
        relationships = []
        
        for edge in self.edges.values():
            if edge.from_node == node_id:
                if edge_type is None or edge.type == edge_type:
                    target_node = self.nodes.get(edge.to_node)
                    if target_node:
                        relationships.append((edge, target_node))
        
        return relationships
    
    def get_incoming_relationships(
        self,
        node_id: str,
        edge_type: Optional[str] = None
    ) -> List[Tuple[Edge, Node]]:
        """Get all incoming relationships for a node"""
        relationships = []
        
        for edge in self.edges.values():
            if edge.to_node == node_id:
                if edge_type is None or edge.type == edge_type:
                    source_node = self.nodes.get(edge.from_node)
                    if source_node:
                        relationships.append((edge, source_node))
        
        return relationships
    
    def find_path(self, from_node_id: str, to_node_id: str, max_depth: int = 5) -> Optional[List[str]]:
        """Find a path between two nodes (simple BFS)"""
        if from_node_id == to_node_id:
            return [from_node_id]
        
        visited = set()
        queue = [(from_node_id, [from_node_id])]
        
        while queue and len(queue[0][1]) <= max_depth:
            current_id, path = queue.pop(0)
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            # Get all outgoing edges
            for edge in self.edges.values():
                if edge.from_node == current_id:
                    next_id = edge.to_node
                    
                    if next_id == to_node_id:
                        return path + [next_id]
                    
                    if next_id not in visited:
                        queue.append((next_id, path + [next_id]))
        
        return None
    
    def add_pattern(self, pattern: Dict[str, Any]) -> str:
        """Add an architectural pattern to the graph"""
        pattern_id = self.add_node(
            node_type="Pattern",
            name=pattern['name'],
            properties={
                'category': pattern.get('category', 'Unknown'),
                'description': pattern.get('description', ''),
                'benefits': pattern.get('benefits', []),
                'drawbacks': pattern.get('drawbacks', []),
                'context': pattern.get('context', '')
            }
        )
        
        # Add relationships to frameworks if specified
        if 'frameworks' in pattern:
            for framework in pattern['frameworks']:
                framework_node = self.find_node_by_name(framework, "Framework")
                if framework_node:
                    self.add_edge("IMPLEMENTED_IN", pattern_id, framework_node.id)
        
        return pattern_id
    
    def add_principle(self, principle: Dict[str, Any]) -> str:
        """Add a design principle to the graph"""
        principle_id = self.add_node(
            node_type="Principle",
            name=principle['name'],
            properties={
                'description': principle.get('description', ''),
                'rationale': principle.get('rationale', ''),
                'category': principle.get('category', 'General')
            }
        )
        
        return principle_id
    
    def link_pattern_to_principle(self, pattern_id: str, principle_id: str, relationship: str = "IMPLEMENTS"):
        """Link a pattern to a principle it implements"""
        self.add_edge(relationship, pattern_id, principle_id)
    
    def get_pattern_relationships(self, pattern_name: str) -> Dict[str, List[str]]:
        """Get all relationships for a pattern"""
        pattern = self.find_node_by_name(pattern_name, "Pattern")
        if not pattern:
            return {}
        
        relationships = {
            'implements': [],
            'conflicts_with': [],
            'complements': [],
            'frameworks': []
        }
        
        # Outgoing relationships
        for edge, target in self.get_relationships(pattern.id):
            if edge.type == "IMPLEMENTS" and target.type == "Principle":
                relationships['implements'].append(target.name)
            elif edge.type == "CONFLICTS_WITH" and target.type == "Pattern":
                relationships['conflicts_with'].append(target.name)
            elif edge.type == "COMPLEMENTS" and target.type == "Pattern":
                relationships['complements'].append(target.name)
            elif edge.type == "IMPLEMENTED_IN" and target.type == "Framework":
                relationships['frameworks'].append(target.name)
        
        return relationships
    
    def export_graph(self) -> Dict[str, Any]:
        """Export the graph as a dictionary"""
        return {
            'nodes': [
                {
                    'id': node.id,
                    'type': node.type,
                    'name': node.name,
                    'properties': node.properties
                }
                for node in self.nodes.values()
            ],
            'edges': [
                {
                    'id': edge.id,
                    'type': edge.type,
                    'from': edge.from_node,
                    'to': edge.to_node,
                    'properties': edge.properties
                }
                for edge in self.edges.values()
            ]
        }
    
    def import_graph(self, data: Dict[str, Any]):
        """Import a graph from a dictionary"""
        # Clear existing data
        self.nodes.clear()
        self.edges.clear()
        self.node_index.clear()
        self.edge_index.clear()
        
        # Import nodes
        for node_data in data.get('nodes', []):
            node = Node(
                id=node_data['id'],
                type=node_data['type'],
                name=node_data['name'],
                properties=node_data.get('properties', {})
            )
            self.nodes[node.id] = node
            self.node_index[node.type].add(node.id)
        
        # Import edges
        for edge_data in data.get('edges', []):
            edge = Edge(
                id=edge_data['id'],
                type=edge_data['type'],
                from_node=edge_data['from'],
                to_node=edge_data['to'],
                properties=edge_data.get('properties', {})
            )
            self.edges[edge.id] = edge
            self.edge_index[edge.type].add(edge.id)