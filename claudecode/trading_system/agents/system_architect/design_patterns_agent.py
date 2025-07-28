"""
Design Patterns Refactoring Agent

This agent identifies opportunities for design pattern application,
suggests appropriate patterns, and performs automated refactoring
to implement common design patterns.
"""

import ast
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

from ..base.agent import BaseAgent


@dataclass
class PatternOpportunity:
    """Represents an opportunity to apply a design pattern"""
    pattern_name: str
    file_path: str
    line_number: int
    class_or_function: str
    description: str
    confidence: float  # 0.0 to 1.0
    refactoring_benefit: str
    implementation_effort: int  # 1-10 scale
    suggested_implementation: str


class DesignPatternsAgent(BaseAgent):
    """
    Design Patterns Analysis and Refactoring Agent
    
    Identifies opportunities for applying design patterns including:
    - Creational: Factory, Builder, Singleton, Abstract Factory
    - Structural: Adapter, Decorator, Facade, Proxy
    - Behavioral: Strategy, Observer, Command, Template Method, State
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("DesignPatterns", config)
        self.logger = logging.getLogger(__name__)
        
        # Pattern detection configurations
        self.min_confidence = config.get('min_confidence', 0.6)
        self.factory_threshold = config.get('factory_threshold', 3)  # Min classes for factory
        self.strategy_threshold = config.get('strategy_threshold', 3)  # Min if/elif for strategy
    
    async def analyze_pattern_opportunities(self, target_path: str) -> Dict[str, Any]:
        """
        Analyze code for design pattern application opportunities
        
        Args:
            target_path: Path to analyze
            
        Returns:
            Pattern opportunities and recommendations
        """
        self.logger.info(f"Analyzing design pattern opportunities in {target_path}")
        
        opportunities = []
        path = Path(target_path)
        
        if path.is_file() and path.suffix == '.py':
            opportunities.extend(await self._analyze_file(path))
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                opportunities.extend(await self._analyze_file(py_file))
        
        # Filter by confidence threshold
        high_confidence = [op for op in opportunities if op.confidence >= self.min_confidence]
        
        # Prioritize opportunities
        prioritized = self._prioritize_opportunities(high_confidence)
        
        # Generate pattern usage documentation
        current_patterns = self._document_existing_patterns(opportunities)
        
        return {
            'total_opportunities': len(opportunities),
            'high_confidence_opportunities': len(high_confidence),
            'opportunities': [self._opportunity_to_dict(op) for op in prioritized],
            'existing_patterns': current_patterns,
            'recommendations': self._generate_pattern_recommendations(prioritized),
            'refactoring_priorities': self._create_refactoring_plan(prioritized)
        }
    
    async def _analyze_file(self, file_path: Path) -> List[PatternOpportunity]:
        """Analyze a single file for pattern opportunities"""
        opportunities = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Analyze for different pattern opportunities
            opportunities.extend(self._detect_factory_opportunities(tree, file_path))
            opportunities.extend(self._detect_strategy_opportunities(tree, file_path))
            opportunities.extend(self._detect_observer_opportunities(tree, file_path))
            opportunities.extend(self._detect_decorator_opportunities(tree, file_path))
            opportunities.extend(self._detect_singleton_opportunities(tree, file_path))
            opportunities.extend(self._detect_builder_opportunities(tree, file_path))
            opportunities.extend(self._detect_adapter_opportunities(tree, file_path))
            opportunities.extend(self._detect_command_opportunities(tree, file_path))
            opportunities.extend(self._detect_template_method_opportunities(tree, file_path))
            opportunities.extend(self._detect_facade_opportunities(tree, file_path))
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
        
        return opportunities
    
    def _detect_factory_opportunities(self, tree: ast.AST, file_path: Path) -> List[PatternOpportunity]:
        """Detect opportunities for Factory pattern"""
        opportunities = []
        
        # Look for classes with many similar instantiation patterns
        class_instantiations = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                class_name = node.func.id
                if class_name[0].isupper():  # Likely a class
                    if class_name not in class_instantiations:
                        class_instantiations[class_name] = []
                    class_instantiations[class_name].append(node)
        
        # Find patterns with conditional instantiation
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                instantiations_in_branches = self._count_instantiations_in_branches(node)
                if instantiations_in_branches >= self.factory_threshold:
                    opportunities.append(PatternOpportunity(
                        pattern_name="Factory Method",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        class_or_function=self._get_containing_function_or_class(node, tree),
                        description=f"Conditional object creation with {instantiations_in_branches} different types",
                        confidence=0.8,
                        refactoring_benefit="Eliminates conditional object creation, improves extensibility",
                        implementation_effort=5,
                        suggested_implementation="Create factory method or abstract factory to handle object creation"
                    ))
        
        return opportunities
    
    def _detect_strategy_opportunities(self, tree: ast.AST, file_path: Path) -> List[PatternOpportunity]:
        """Detect opportunities for Strategy pattern"""
        opportunities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                chain_length = self._count_if_elif_chain(node)
                if chain_length >= self.strategy_threshold:
                    # Check if each branch performs similar operations
                    if self._branches_have_similar_structure(node):
                        opportunities.append(PatternOpportunity(
                            pattern_name="Strategy",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            class_or_function=self._get_containing_function_or_class(node, tree),
                            description=f"Long if/elif chain with {chain_length} conditions performing similar operations",
                            confidence=0.85,
                            refactoring_benefit="Eliminates conditional logic, enables runtime algorithm selection",
                            implementation_effort=6,
                            suggested_implementation="Extract each branch into separate strategy class with common interface"
                        ))
        
        return opportunities
    
    def _detect_observer_opportunities(self, tree: ast.AST, file_path: Path) -> List[PatternOpportunity]:
        """Detect opportunities for Observer pattern"""
        opportunities = []
        
        # Look for manual notification patterns
        for class_node in ast.walk(tree):
            if isinstance(class_node, ast.ClassDef):
                notification_methods = []
                callback_lists = []
                
                for method in class_node.body:
                    if isinstance(method, ast.FunctionDef):
                        method_name = method.name.lower()
                        
                        # Look for notification methods
                        if any(term in method_name for term in ['notify', 'update', 'broadcast', 'trigger']):
                            notification_methods.append(method.name)
                        
                        # Look for callback list management
                        for node in ast.walk(method):
                            if isinstance(node, ast.Call) and hasattr(node.func, 'attr'):
                                if node.func.attr in ['append', 'remove'] and 'callback' in str(node):
                                    callback_lists.append(method.name)
                
                if len(notification_methods) >= 2 or callback_lists:
                    opportunities.append(PatternOpportunity(
                        pattern_name="Observer",
                        file_path=str(file_path),
                        line_number=class_node.lineno,
                        class_or_function=class_node.name,
                        description=f"Manual notification system with {len(notification_methods)} notification methods",
                        confidence=0.75,
                        refactoring_benefit="Decouples subject from observers, improves maintainability",
                        implementation_effort=7,
                        suggested_implementation="Implement Observer pattern with Subject and Observer interfaces"
                    ))
        
        return opportunities
    
    def _detect_decorator_opportunities(self, tree: ast.AST, file_path: Path) -> List[PatternOpportunity]:
        """Detect opportunities for Decorator pattern"""
        opportunities = []
        
        # Look for wrapper-like patterns
        for class_node in ast.walk(tree):
            if isinstance(class_node, ast.ClassDef):
                # Check if class wraps another object
                has_wrapped_object = False
                delegates_methods = 0
                
                for method in class_node.body:
                    if isinstance(method, ast.FunctionDef):
                        # Look for delegation patterns
                        for call_node in ast.walk(method):
                            if (isinstance(call_node, ast.Call) and 
                                isinstance(call_node.func, ast.Attribute) and
                                isinstance(call_node.func.value, ast.Attribute)):
                                delegates_methods += 1
                        
                        # Look for wrapped object in __init__
                        if method.name == '__init__':
                            for assign in ast.walk(method):
                                if isinstance(assign, ast.Assign):
                                    has_wrapped_object = True
                
                if has_wrapped_object and delegates_methods >= 3:
                    opportunities.append(PatternOpportunity(
                        pattern_name="Decorator",
                        file_path=str(file_path),
                        line_number=class_node.lineno,
                        class_or_function=class_node.name,
                        description=f"Class wraps object and delegates {delegates_methods} methods",
                        confidence=0.7,
                        refactoring_benefit="Allows dynamic behavior addition without subclassing",
                        implementation_effort=4,
                        suggested_implementation="Formalize decorator pattern with component interface"
                    ))
        
        return opportunities
    
    def _detect_singleton_opportunities(self, tree: ast.AST, file_path: Path) -> List[PatternOpportunity]:
        """Detect opportunities for Singleton pattern"""
        opportunities = []
        
        for class_node in ast.walk(tree):
            if isinstance(class_node, ast.ClassDef):
                # Look for global state or configuration classes
                is_config_like = any(term in class_node.name.lower() 
                                   for term in ['config', 'settings', 'manager', 'registry'])
                
                has_class_variables = any(isinstance(node, ast.Assign) 
                                        for node in class_node.body)
                
                # Look for manual singleton implementation
                has_instance_check = False
                for method in class_node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == '__new__':
                        has_instance_check = True
                
                if (is_config_like and has_class_variables) or has_instance_check:
                    confidence = 0.8 if has_instance_check else 0.6
                    
                    opportunities.append(PatternOpportunity(
                        pattern_name="Singleton",
                        file_path=str(file_path),
                        line_number=class_node.lineno,
                        class_or_function=class_node.name,
                        description=f"Class '{class_node.name}' appears to manage global state",
                        confidence=confidence,
                        refactoring_benefit="Ensures single instance and global access point",
                        implementation_effort=3,
                        suggested_implementation="Implement proper Singleton pattern with thread safety"
                    ))
        
        return opportunities
    
    def _detect_builder_opportunities(self, tree: ast.AST, file_path: Path) -> List[PatternOpportunity]:
        """Detect opportunities for Builder pattern"""
        opportunities = []
        
        for class_node in ast.walk(tree):
            if isinstance(class_node, ast.ClassDef):
                init_method = None
                for method in class_node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == '__init__':
                        init_method = method
                        break
                
                if init_method and len(init_method.args.args) > 6:  # Many constructor parameters
                    opportunities.append(PatternOpportunity(
                        pattern_name="Builder",
                        file_path=str(file_path),
                        line_number=class_node.lineno,
                        class_or_function=class_node.name,
                        description=f"Class constructor has {len(init_method.args.args)-1} parameters",
                        confidence=0.75,
                        refactoring_benefit="Simplifies object construction, improves readability",
                        implementation_effort=5,
                        suggested_implementation="Create Builder class to construct complex objects step by step"
                    ))
        
        return opportunities
    
    def _detect_adapter_opportunities(self, tree: ast.AST, file_path: Path) -> List[PatternOpportunity]:
        """Detect opportunities for Adapter pattern"""
        opportunities = []
        
        # Look for classes that seem to translate between interfaces
        for class_node in ast.walk(tree):
            if isinstance(class_node, ast.ClassDef):
                if any(term in class_node.name.lower() 
                      for term in ['adapter', 'wrapper', 'bridge', 'translator']):
                    
                    opportunities.append(PatternOpportunity(
                        pattern_name="Adapter",
                        file_path=str(file_path),
                        line_number=class_node.lineno,
                        class_or_function=class_node.name,
                        description=f"Class name suggests adapter pattern: {class_node.name}",
                        confidence=0.65,
                        refactoring_benefit="Enables interface compatibility between incompatible classes",
                        implementation_effort=4,
                        suggested_implementation="Formalize adapter pattern with clear target and adaptee interfaces"
                    ))
        
        return opportunities
    
    def _detect_command_opportunities(self, tree: ast.AST, file_path: Path) -> List[PatternOpportunity]:
        """Detect opportunities for Command pattern"""
        opportunities = []
        
        # Look for action/command-like classes or undo/redo functionality
        for class_node in ast.walk(tree):
            if isinstance(class_node, ast.ClassDef):
                is_action_like = any(term in class_node.name.lower() 
                                   for term in ['action', 'command', 'operation', 'task'])
                
                has_execute = any(method.name == 'execute' 
                                for method in class_node.body 
                                if isinstance(method, ast.FunctionDef))
                
                has_undo = any(method.name in ['undo', 'revert'] 
                             for method in class_node.body 
                             if isinstance(method, ast.FunctionDef))
                
                if is_action_like or (has_execute and has_undo):
                    opportunities.append(PatternOpportunity(
                        pattern_name="Command",
                        file_path=str(file_path),
                        line_number=class_node.lineno,
                        class_or_function=class_node.name,
                        description=f"Class implements command-like behavior",
                        confidence=0.7,
                        refactoring_benefit="Enables undo/redo, queuing, and logging of operations",
                        implementation_effort=5,
                        suggested_implementation="Implement Command interface with execute() and undo() methods"
                    ))
        
        return opportunities
    
    def _detect_template_method_opportunities(self, tree: ast.AST, file_path: Path) -> List[PatternOpportunity]:
        """Detect opportunities for Template Method pattern"""
        opportunities = []
        
        # Look for classes with similar method structures
        class_hierarchies = self._find_class_hierarchies(tree)
        
        for base_class, derived_classes in class_hierarchies.items():
            if len(derived_classes) >= 2:
                # Check for similar method names across derived classes
                common_methods = self._find_common_methods(derived_classes)
                if len(common_methods) >= 3:
                    opportunities.append(PatternOpportunity(
                        pattern_name="Template Method",
                        file_path=str(file_path),
                        line_number=base_class.lineno if base_class else 1,
                        class_or_function=base_class.name if base_class else "Multiple classes",
                        description=f"Class hierarchy with {len(common_methods)} common methods across {len(derived_classes)} subclasses",
                        confidence=0.8,
                        refactoring_benefit="Defines algorithm skeleton, allows subclass customization",
                        implementation_effort=6,
                        suggested_implementation="Move common algorithm to base class with hook methods for customization"
                    ))
        
        return opportunities
    
    def _detect_facade_opportunities(self, tree: ast.AST, file_path: Path) -> List[PatternOpportunity]:
        """Detect opportunities for Facade pattern"""
        opportunities = []
        
        # Look for classes that coordinate many other classes
        for class_node in ast.walk(tree):
            if isinstance(class_node, ast.ClassDef):
                imported_classes = set()
                
                for method in class_node.body:
                    if isinstance(method, ast.FunctionDef):
                        for call_node in ast.walk(method):
                            if isinstance(call_node, ast.Call) and isinstance(call_node.func, ast.Name):
                                if call_node.func.id[0].isupper():
                                    imported_classes.add(call_node.func.id)
                
                if len(imported_classes) >= 5:
                    opportunities.append(PatternOpportunity(
                        pattern_name="Facade",
                        file_path=str(file_path),
                        line_number=class_node.lineno,
                        class_or_function=class_node.name,
                        description=f"Class coordinates {len(imported_classes)} different classes",
                        confidence=0.7,
                        refactoring_benefit="Simplifies complex subsystem interface",
                        implementation_effort=4,
                        suggested_implementation="Formalize facade with simplified interface to complex subsystem"
                    ))
        
        return opportunities
    
    def _count_if_elif_chain(self, if_node: ast.If) -> int:
        """Count the length of an if/elif chain"""
        count = 1
        current = if_node
        
        while (hasattr(current, 'orelse') and 
               len(current.orelse) == 1 and 
               isinstance(current.orelse[0], ast.If)):
            count += 1
            current = current.orelse[0]
        
        return count
    
    def _count_instantiations_in_branches(self, if_node: ast.If) -> int:
        """Count different class instantiations across if/elif branches"""
        instantiations = set()
        
        def collect_instantiations(node):
            for n in ast.walk(node):
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    if n.func.id[0].isupper():
                        instantiations.add(n.func.id)
        
        # Check main if body
        for stmt in if_node.body:
            collect_instantiations(stmt)
        
        # Check elif/else branches
        current = if_node
        while hasattr(current, 'orelse') and current.orelse:
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                for stmt in current.body:
                    collect_instantiations(stmt)
            else:
                for stmt in current.orelse:
                    collect_instantiations(stmt)
                break
        
        return len(instantiations)
    
    def _branches_have_similar_structure(self, if_node: ast.If) -> bool:
        """Check if if/elif branches have similar structure"""
        branch_structures = []
        
        def get_structure(body):
            return [type(stmt).__name__ for stmt in body]
        
        branch_structures.append(get_structure(if_node.body))
        
        current = if_node
        while hasattr(current, 'orelse') and current.orelse:
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                branch_structures.append(get_structure(current.body))
            else:
                branch_structures.append(get_structure(current.orelse))
                break
        
        # Check if structures are similar (at least 70% overlap)
        if len(branch_structures) < 2:
            return False
        
        first_structure = set(branch_structures[0])
        similarity_count = 0
        
        for structure in branch_structures[1:]:
            overlap = len(first_structure.intersection(set(structure)))
            if overlap / max(len(first_structure), len(structure)) >= 0.7:
                similarity_count += 1
        
        return similarity_count >= len(branch_structures) - 1
    
    def _get_containing_function_or_class(self, node: ast.AST, tree: ast.AST) -> str:
        """Get the name of the function or class containing a node"""
        # This is a simplified implementation
        for parent in ast.walk(tree):
            if isinstance(parent, (ast.FunctionDef, ast.ClassDef)):
                for child in ast.walk(parent):
                    if child is node:
                        return parent.name
        return "Unknown"
    
    def _find_class_hierarchies(self, tree: ast.AST) -> Dict[ast.ClassDef, List[ast.ClassDef]]:
        """Find class inheritance hierarchies"""
        hierarchies = {}
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        for class_node in classes:
            if class_node.bases:
                for base in class_node.bases:
                    if isinstance(base, ast.Name):
                        base_class = next((c for c in classes if c.name == base.id), None)
                        if base_class:
                            if base_class not in hierarchies:
                                hierarchies[base_class] = []
                            hierarchies[base_class].append(class_node)
        
        return hierarchies
    
    def _find_common_methods(self, classes: List[ast.ClassDef]) -> List[str]:
        """Find common method names across classes"""
        if not classes:
            return []
        
        method_sets = []
        for class_node in classes:
            methods = {method.name for method in class_node.body 
                      if isinstance(method, ast.FunctionDef) and not method.name.startswith('_')}
            method_sets.append(methods)
        
        common_methods = method_sets[0]
        for method_set in method_sets[1:]:
            common_methods = common_methods.intersection(method_set)
        
        return list(common_methods)
    
    def _prioritize_opportunities(self, opportunities: List[PatternOpportunity]) -> List[PatternOpportunity]:
        """Prioritize pattern opportunities by benefit and confidence"""
        return sorted(opportunities, 
                     key=lambda op: (op.confidence, -op.implementation_effort), 
                     reverse=True)
    
    def _generate_pattern_recommendations(self, opportunities: List[PatternOpportunity]) -> List[str]:
        """Generate high-level pattern recommendations"""
        recommendations = []
        
        pattern_counts = {}
        for op in opportunities:
            pattern_counts[op.pattern_name] = pattern_counts.get(op.pattern_name, 0) + 1
        
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            recommendations.append(f"Apply {pattern} pattern in {count} locations for improved design")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _create_refactoring_plan(self, opportunities: List[PatternOpportunity]) -> List[Dict[str, Any]]:
        """Create prioritized refactoring plan"""
        plan = []
        
        for op in opportunities[:5]:  # Top 5 opportunities
            plan.append({
                'type': 'pattern_application',
                'target': f"{op.file_path}:{op.line_number}",
                'pattern': op.pattern_name,
                'description': op.description,
                'impact': int(op.confidence * 10),
                'effort': op.implementation_effort,
                'implementation': op.suggested_implementation
            })
        
        return plan
    
    def _document_existing_patterns(self, opportunities: List[PatternOpportunity]) -> Dict[str, List[str]]:
        """Document patterns already in use"""
        # This would analyze existing code for implemented patterns
        return {
            'identified_patterns': [],
            'pattern_usage': {}
        }
    
    def _opportunity_to_dict(self, opportunity: PatternOpportunity) -> Dict[str, Any]:
        """Convert opportunity to dictionary"""
        return {
            'pattern_name': opportunity.pattern_name,
            'file_path': opportunity.file_path,
            'line_number': opportunity.line_number,
            'class_or_function': opportunity.class_or_function,
            'description': opportunity.description,
            'confidence': opportunity.confidence,
            'refactoring_benefit': opportunity.refactoring_benefit,
            'implementation_effort': opportunity.implementation_effort,
            'suggested_implementation': opportunity.suggested_implementation
        }
    
    async def apply_design_pattern(self, target: str, pattern_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a design pattern to the specified code location
        
        Args:
            target: File path and line number
            pattern_info: Information about the pattern to apply
            
        Returns:
            Result of pattern application
        """
        self.logger.info(f"Applying {pattern_info.get('pattern')} pattern to {target}")
        
        # Placeholder for actual pattern implementation
        return {
            'success': False,
            'message': "Automatic pattern application is not yet implemented",
            'manual_steps': pattern_info.get('implementation', 'No implementation details available')
        }
    
    async def document_patterns_used(self, component: str) -> Dict[str, Any]:
        """
        Document design patterns used in a component
        
        Args:
            component: Component to document
            
        Returns:
            Pattern documentation
        """
        return {
            'identified_patterns': [],
            'pattern_descriptions': {},
            'usage_examples': {},
            'benefits_realized': []
        }