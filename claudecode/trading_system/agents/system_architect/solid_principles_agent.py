"""
SOLID Principles Enforcement Agent

This agent analyzes code for SOLID principles violations and provides
automated refactoring suggestions and implementations.
"""

import ast
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re

from ..base.agent import BaseAgent


@dataclass
class SOLIDViolation:
    """Represents a SOLID principle violation"""
    principle: str  # S, O, L, I, or D
    file_path: str
    line_number: int
    function_or_class: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    suggested_fix: str
    refactoring_effort: int  # 1-10 scale


class SOLIDPrinciplesAgent(BaseAgent):
    """
    SOLID Principles Enforcement Agent
    
    Analyzes code for violations of SOLID principles:
    - Single Responsibility Principle (SRP)
    - Open/Closed Principle (OCP)
    - Liskov Substitution Principle (LSP)
    - Interface Segregation Principle (ISP)
    - Dependency Inversion Principle (DIP)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SOLIDPrinciples", config)
        self.logger = logging.getLogger(__name__)
        
        # Thresholds for violation detection
        self.complexity_threshold = config.get('complexity_threshold', 10)
        self.method_count_threshold = config.get('method_count_threshold', 15)
        self.line_count_threshold = config.get('line_count_threshold', 300)
    
    async def analyze_solid_compliance(self, target_path: str) -> Dict[str, Any]:
        """
        Analyze code for SOLID principles compliance
        
        Args:
            target_path: Path to analyze
            
        Returns:
            Comprehensive SOLID analysis results
        """
        self.logger.info(f"Analyzing SOLID compliance for {target_path}")
        
        violations = []
        path = Path(target_path)
        
        if path.is_file() and path.suffix == '.py':
            violations.extend(await self._analyze_file(path))
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                violations.extend(await self._analyze_file(py_file))
        
        # Calculate scores and generate report
        principle_scores = self._calculate_principle_scores(violations)
        overall_score = sum(principle_scores.values()) / len(principle_scores)
        
        recommendations = self._generate_recommendations(violations)
        critical_issues = [v for v in violations if v.severity == 'critical']
        
        return {
            'overall_score': overall_score,
            'principle_scores': principle_scores,
            'violations': [self._violation_to_dict(v) for v in violations],
            'recommendations': recommendations,
            'critical_issues': [self._violation_to_dict(v) for v in critical_issues],
            'refactoring_priorities': self._prioritize_refactoring(violations)
        }
    
    async def _analyze_file(self, file_path: Path) -> List[SOLIDViolation]:
        """Analyze a single Python file for SOLID violations"""
        violations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Analyze each class and function
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    violations.extend(self._analyze_class(node, file_path, content))
                elif isinstance(node, ast.FunctionDef):
                    violations.extend(self._analyze_function(node, file_path, content))
                    
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
        
        return violations
    
    def _analyze_class(self, class_node: ast.ClassDef, file_path: Path, content: str) -> List[SOLIDViolation]:
        """Analyze a class for SOLID violations"""
        violations = []
        
        # Single Responsibility Principle violations
        srp_violations = self._check_srp_violations(class_node, file_path, content)
        violations.extend(srp_violations)
        
        # Open/Closed Principle violations
        ocp_violations = self._check_ocp_violations(class_node, file_path, content)
        violations.extend(ocp_violations)
        
        # Liskov Substitution Principle violations
        lsp_violations = self._check_lsp_violations(class_node, file_path, content)
        violations.extend(lsp_violations)
        
        # Interface Segregation Principle violations
        isp_violations = self._check_isp_violations(class_node, file_path, content)
        violations.extend(isp_violations)
        
        # Dependency Inversion Principle violations
        dip_violations = self._check_dip_violations(class_node, file_path, content)
        violations.extend(dip_violations)
        
        return violations
    
    def _analyze_function(self, func_node: ast.FunctionDef, file_path: Path, content: str) -> List[SOLIDViolation]:
        """Analyze a function for SOLID violations"""
        violations = []
        
        # Single Responsibility Principle for functions
        if self._function_has_multiple_responsibilities(func_node):
            violations.append(SOLIDViolation(
                principle='S',
                file_path=str(file_path),
                line_number=func_node.lineno,
                function_or_class=func_node.name,
                description=f"Function '{func_node.name}' appears to have multiple responsibilities",
                severity=self._calculate_severity(func_node),
                suggested_fix="Consider splitting this function into smaller, single-purpose functions",
                refactoring_effort=5
            ))
        
        return violations
    
    def _check_srp_violations(self, class_node: ast.ClassDef, file_path: Path, content: str) -> List[SOLIDViolation]:
        """Check for Single Responsibility Principle violations"""
        violations = []
        
        methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        # Too many methods suggests multiple responsibilities
        if len(methods) > self.method_count_threshold:
            violations.append(SOLIDViolation(
                principle='S',
                file_path=str(file_path),
                line_number=class_node.lineno,
                function_or_class=class_node.name,
                description=f"Class '{class_node.name}' has {len(methods)} methods, suggesting multiple responsibilities",
                severity='high',
                suggested_fix="Consider splitting this class based on distinct responsibilities",
                refactoring_effort=8
            ))
        
        # Analyze method responsibilities
        responsibilities = self._identify_class_responsibilities(class_node)
        if len(responsibilities) > 3:
            violations.append(SOLIDViolation(
                principle='S',
                file_path=str(file_path),
                line_number=class_node.lineno,
                function_or_class=class_node.name,
                description=f"Class '{class_node.name}' handles {len(responsibilities)} different responsibilities: {', '.join(responsibilities)}",
                severity='medium',
                suggested_fix="Separate each responsibility into its own class",
                refactoring_effort=6
            ))
        
        return violations
    
    def _check_ocp_violations(self, class_node: ast.ClassDef, file_path: Path, content: str) -> List[SOLIDViolation]:
        """Check for Open/Closed Principle violations"""
        violations = []
        
        # Look for large if/elif chains that suggest need for extension
        for method in class_node.body:
            if isinstance(method, ast.FunctionDef):
                if_chain_length = self._count_if_elif_chain(method)
                if if_chain_length > 5:
                    violations.append(SOLIDViolation(
                        principle='O',
                        file_path=str(file_path),
                        line_number=method.lineno,
                        function_or_class=f"{class_node.name}.{method.name}",
                        description=f"Method '{method.name}' has long if/elif chain ({if_chain_length} conditions), violating OCP",
                        severity='medium',
                        suggested_fix="Consider using Strategy pattern or polymorphism to eliminate conditional logic",
                        refactoring_effort=7
                    ))
        
        return violations
    
    def _check_lsp_violations(self, class_node: ast.ClassDef, file_path: Path, content: str) -> List[SOLIDViolation]:
        """Check for Liskov Substitution Principle violations"""
        violations = []
        
        # Look for method overrides that strengthen preconditions or weaken postconditions
        for method in class_node.body:
            if isinstance(method, ast.FunctionDef) and self._is_override(method):
                # Check for type checking in override
                if self._has_type_checking_in_override(method):
                    violations.append(SOLIDViolation(
                        principle='L',
                        file_path=str(file_path),
                        line_number=method.lineno,
                        function_or_class=f"{class_node.name}.{method.name}",
                        description=f"Override method '{method.name}' performs type checking, potentially violating LSP",
                        severity='medium',
                        suggested_fix="Remove type checking and ensure override can handle same inputs as base method",
                        refactoring_effort=4
                    ))
        
        return violations
    
    def _check_isp_violations(self, class_node: ast.ClassDef, file_path: Path, content: str) -> List[SOLIDViolation]:
        """Check for Interface Segregation Principle violations"""
        violations = []
        
        # Look for large interfaces (many abstract methods)
        abstract_methods = self._count_abstract_methods(class_node)
        if abstract_methods > 8:
            violations.append(SOLIDViolation(
                principle='I',
                file_path=str(file_path),
                line_number=class_node.lineno,
                function_or_class=class_node.name,
                description=f"Interface '{class_node.name}' has {abstract_methods} abstract methods, violating ISP",
                severity='medium',
                suggested_fix="Split interface into smaller, more focused interfaces",
                refactoring_effort=6
            ))
        
        return violations
    
    def _check_dip_violations(self, class_node: ast.ClassDef, file_path: Path, content: str) -> List[SOLIDViolation]:
        """Check for Dependency Inversion Principle violations"""
        violations = []
        
        # Look for direct instantiation of concrete classes
        concrete_dependencies = self._find_concrete_dependencies(class_node)
        if concrete_dependencies:
            violations.append(SOLIDViolation(
                principle='D',
                file_path=str(file_path),
                line_number=class_node.lineno,
                function_or_class=class_node.name,
                description=f"Class '{class_node.name}' directly instantiates concrete classes: {', '.join(concrete_dependencies)}",
                severity='medium',
                suggested_fix="Inject dependencies through constructor or use dependency injection framework",
                refactoring_effort=5
            ))
        
        return violations
    
    def _function_has_multiple_responsibilities(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has multiple responsibilities"""
        # Count different types of operations
        operations = {
            'database': 0,
            'computation': 0,
            'validation': 0,
            'formatting': 0,
            'io': 0
        }
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                func_name = getattr(node.func, 'id', '') or getattr(node.func, 'attr', '')
                
                if any(db_term in func_name.lower() for db_term in ['query', 'select', 'insert', 'update', 'delete']):
                    operations['database'] += 1
                elif any(math_term in func_name.lower() for math_term in ['calculate', 'compute', 'sum', 'avg']):
                    operations['computation'] += 1
                elif any(val_term in func_name.lower() for val_term in ['validate', 'check', 'verify']):
                    operations['validation'] += 1
                elif any(fmt_term in func_name.lower() for fmt_term in ['format', 'serialize', 'parse']):
                    operations['formatting'] += 1
                elif any(io_term in func_name.lower() for io_term in ['read', 'write', 'open', 'save']):
                    operations['io'] += 1
        
        # If function performs more than 2 types of operations, it likely has multiple responsibilities
        active_operations = sum(1 for count in operations.values() if count > 0)
        return active_operations > 2
    
    def _identify_class_responsibilities(self, class_node: ast.ClassDef) -> List[str]:
        """Identify different responsibilities of a class based on method names"""
        responsibilities = set()
        
        for method in class_node.body:
            if isinstance(method, ast.FunctionDef) and not method.name.startswith('_'):
                method_name = method.name.lower()
                
                if any(term in method_name for term in ['save', 'load', 'read', 'write', 'persist']):
                    responsibilities.add('data_persistence')
                elif any(term in method_name for term in ['validate', 'check', 'verify']):
                    responsibilities.add('validation')
                elif any(term in method_name for term in ['calculate', 'compute', 'process']):
                    responsibilities.add('computation')
                elif any(term in method_name for term in ['format', 'render', 'display']):
                    responsibilities.add('presentation')
                elif any(term in method_name for term in ['send', 'receive', 'connect']):
                    responsibilities.add('communication')
                elif any(term in method_name for term in ['log', 'trace', 'debug']):
                    responsibilities.add('logging')
        
        return list(responsibilities)
    
    def _count_if_elif_chain(self, method: ast.FunctionDef) -> int:
        """Count the length of if/elif chains in a method"""
        max_chain = 0
        
        for node in ast.walk(method):
            if isinstance(node, ast.If):
                chain_length = 1
                current = node
                while hasattr(current, 'orelse') and len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                    chain_length += 1
                    current = current.orelse[0]
                max_chain = max(max_chain, chain_length)
        
        return max_chain
    
    def _is_override(self, method: ast.FunctionDef) -> bool:
        """Check if method is likely an override"""
        # Simple heuristic: check for @override decorator or common override methods
        for decorator in method.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'override':
                return True
        
        common_overrides = ['__init__', '__str__', '__repr__', '__eq__', '__hash__']
        return method.name in common_overrides
    
    def _has_type_checking_in_override(self, method: ast.FunctionDef) -> bool:
        """Check if override method performs type checking"""
        for node in ast.walk(method):
            if isinstance(node, ast.Call):
                func_name = getattr(node.func, 'id', '')
                if func_name in ['isinstance', 'type']:
                    return True
        return False
    
    def _count_abstract_methods(self, class_node: ast.ClassDef) -> int:
        """Count abstract methods in a class"""
        abstract_count = 0
        
        for method in class_node.body:
            if isinstance(method, ast.FunctionDef):
                for decorator in method.decorator_list:
                    if isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                        abstract_count += 1
                        break
        
        return abstract_count
    
    def _find_concrete_dependencies(self, class_node: ast.ClassDef) -> List[str]:
        """Find direct instantiations of concrete classes"""
        concrete_deps = []
        
        for node in ast.walk(class_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                # Look for class instantiation (capitalized names)
                if node.func.id[0].isupper() and node.func.id not in ['True', 'False', 'None']:
                    concrete_deps.append(node.func.id)
        
        return list(set(concrete_deps))
    
    def _calculate_severity(self, node: ast.AST) -> str:
        """Calculate severity of a violation"""
        if isinstance(node, ast.FunctionDef):
            lines = node.end_lineno - node.lineno if node.end_lineno else 20
            if lines > 100:
                return 'critical'
            elif lines > 50:
                return 'high'
            elif lines > 20:
                return 'medium'
            else:
                return 'low'
        return 'medium'
    
    def _calculate_principle_scores(self, violations: List[SOLIDViolation]) -> Dict[str, float]:
        """Calculate scores for each SOLID principle"""
        principle_counts = {'S': 0, 'O': 0, 'L': 0, 'I': 0, 'D': 0}
        
        for violation in violations:
            principle_counts[violation.principle] += 1
        
        # Convert counts to scores (10 - violations, minimum 0)
        scores = {}
        for principle, count in principle_counts.items():
            scores[principle] = max(0, 10 - count)
        
        return scores
    
    def _generate_recommendations(self, violations: List[SOLIDViolation]) -> List[str]:
        """Generate refactoring recommendations"""
        recommendations = []
        
        # Group violations by principle
        by_principle = {}
        for violation in violations:
            if violation.principle not in by_principle:
                by_principle[violation.principle] = []
            by_principle[violation.principle].append(violation)
        
        for principle, viols in by_principle.items():
            if principle == 'S':
                recommendations.append(f"Split {len(viols)} classes/functions with multiple responsibilities")
            elif principle == 'O':
                recommendations.append(f"Replace {len(viols)} conditional logic blocks with polymorphism")
            elif principle == 'L':
                recommendations.append(f"Fix {len(viols)} inheritance violations")
            elif principle == 'I':
                recommendations.append(f"Segregate {len(viols)} large interfaces")
            elif principle == 'D':
                recommendations.append(f"Inject dependencies for {len(viols)} tightly coupled classes")
        
        return recommendations
    
    def _prioritize_refactoring(self, violations: List[SOLIDViolation]) -> List[Dict[str, Any]]:
        """Prioritize refactoring tasks"""
        priorities = []
        
        for violation in violations:
            impact = {'critical': 10, 'high': 8, 'medium': 5, 'low': 2}[violation.severity]
            
            priorities.append({
                'type': 'solid_violation',
                'target': f"{violation.file_path}:{violation.line_number}",
                'description': violation.description,
                'principle': violation.principle,
                'impact': impact,
                'effort': violation.refactoring_effort,
                'suggested_fix': violation.suggested_fix
            })
        
        return sorted(priorities, key=lambda x: (x['impact'], -x['effort']), reverse=True)
    
    def _violation_to_dict(self, violation: SOLIDViolation) -> Dict[str, Any]:
        """Convert violation to dictionary"""
        return {
            'principle': violation.principle,
            'file_path': violation.file_path,
            'line_number': violation.line_number,
            'function_or_class': violation.function_or_class,
            'description': violation.description,
            'severity': violation.severity,
            'suggested_fix': violation.suggested_fix,
            'refactoring_effort': violation.refactoring_effort
        }
    
    async def fix_solid_violation(self, target: str, violation_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to fix a SOLID violation automatically
        
        Args:
            target: File path and line number (file:line)
            violation_info: Information about the violation
            
        Returns:
            Result of the fix attempt
        """
        self.logger.info(f"Attempting to fix SOLID violation: {violation_info.get('description')}")
        
        # This is a placeholder for actual refactoring logic
        # In a real implementation, this would perform AST transformations
        
        return {
            'success': False,
            'message': "Automatic SOLID violation fixing is not yet implemented",
            'manual_steps': violation_info.get('suggested_fix', 'No suggestions available')
        }