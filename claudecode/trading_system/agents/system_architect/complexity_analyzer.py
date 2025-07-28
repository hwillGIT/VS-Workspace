"""
Cyclomatic Complexity Analyzer and Optimizer

This agent analyzes code complexity using multiple metrics and provides
automated refactoring suggestions to reduce complexity.
"""

import ast
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from pathlib import Path
import math

from ..base.agent import BaseAgent


@dataclass
class ComplexityMetrics:
    """Comprehensive complexity metrics for a code element"""
    cyclomatic_complexity: int
    cognitive_complexity: int
    nesting_depth: int
    line_count: int
    parameter_count: int
    branch_count: int
    loop_count: int
    overall_score: float


@dataclass
class ComplexityIssue:
    """Represents a complexity issue in code"""
    file_path: str
    line_number: int
    function_or_class: str
    issue_type: str
    severity: str
    current_complexity: int
    recommended_complexity: int
    description: str
    refactoring_suggestion: str
    estimated_effort: int


class ComplexityAnalyzer(BaseAgent):
    """
    Cyclomatic Complexity Analyzer and Optimizer
    
    Analyzes multiple complexity metrics:
    - Cyclomatic Complexity (McCabe)
    - Cognitive Complexity
    - Nesting Depth
    - Function/Class Size
    - Parameter Count
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("ComplexityAnalyzer", config)
        self.logger = logging.getLogger(__name__)
        
        # Complexity thresholds
        self.cyclomatic_threshold = config.get('cyclomatic_threshold', 10)
        self.cognitive_threshold = config.get('cognitive_threshold', 15)
        self.nesting_threshold = config.get('nesting_threshold', 4)
        self.line_threshold = config.get('line_threshold', 50)
        self.parameter_threshold = config.get('parameter_threshold', 5)
        
        # Scoring weights
        self.complexity_weights = {
            'cyclomatic': 0.3,
            'cognitive': 0.25,
            'nesting': 0.2,
            'lines': 0.15,
            'parameters': 0.1
        }
    
    async def analyze_complexity(self, target_path: str) -> Dict[str, Any]:
        """
        Analyze complexity metrics for code
        
        Args:
            target_path: Path to analyze
            
        Returns:
            Comprehensive complexity analysis results
        """
        self.logger.info(f"Analyzing complexity for {target_path}")
        
        results = {
            'file_metrics': {},
            'function_metrics': {},
            'class_metrics': {},
            'complexity_issues': [],
            'overall_statistics': {},
            'recommendations': [],
            'refactoring_priorities': []
        }
        
        path = Path(target_path)
        
        if path.is_file() and path.suffix == '.py':
            file_results = await self._analyze_file(path)
            results['file_metrics'][str(path)] = file_results
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                file_results = await self._analyze_file(py_file)
                results['file_metrics'][str(py_file)] = file_results
        
        # Aggregate results
        self._aggregate_results(results)
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        results['refactoring_priorities'] = self._prioritize_refactoring(results)
        
        # Calculate overall score
        results['overall_score'] = self._calculate_overall_score(results)
        
        return results
    
    async def _analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze complexity metrics for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            file_results = {
                'functions': {},
                'classes': {},
                'complexity_issues': [],
                'file_metrics': self._calculate_file_metrics(tree, content)
            }
            
            # Analyze functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_metrics = self._analyze_function_complexity(node, content)
                    file_results['functions'][node.name] = func_metrics
                    
                    # Check for complexity issues
                    issues = self._identify_complexity_issues(node, func_metrics, file_path)
                    file_results['complexity_issues'].extend(issues)
                
                elif isinstance(node, ast.ClassDef):
                    class_metrics = self._analyze_class_complexity(node, content)
                    file_results['classes'][node.name] = class_metrics
                    
                    # Check for class-level complexity issues
                    issues = self._identify_class_complexity_issues(node, class_metrics, file_path)
                    file_results['complexity_issues'].extend(issues)
            
            return file_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            return {'error': str(e)}
    
    def _calculate_file_metrics(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Calculate file-level metrics"""
        lines = content.split('\n')
        
        return {
            'total_lines': len(lines),
            'code_lines': len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'blank_lines': len([line for line in lines if not line.strip()]),
            'function_count': len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
            'class_count': len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        }
    
    def _analyze_function_complexity(self, func_node: ast.FunctionDef, content: str) -> ComplexityMetrics:
        """Analyze complexity metrics for a function"""
        cyclomatic = self._calculate_cyclomatic_complexity(func_node)
        cognitive = self._calculate_cognitive_complexity(func_node)
        nesting = self._calculate_max_nesting_depth(func_node)
        line_count = (func_node.end_lineno or func_node.lineno) - func_node.lineno + 1
        param_count = len(func_node.args.args)
        branches = self._count_branches(func_node)
        loops = self._count_loops(func_node)
        
        # Calculate overall complexity score
        scores = {
            'cyclomatic': min(10, cyclomatic) / 10,
            'cognitive': min(10, cognitive) / 15,
            'nesting': min(10, nesting) / 4,
            'lines': min(10, line_count) / 50,
            'parameters': min(10, param_count) / 5
        }
        
        overall_score = sum(score * self.complexity_weights[metric] 
                          for metric, score in scores.items()) * 10
        
        return ComplexityMetrics(
            cyclomatic_complexity=cyclomatic,
            cognitive_complexity=cognitive,
            nesting_depth=nesting,
            line_count=line_count,
            parameter_count=param_count,
            branch_count=branches,
            loop_count=loops,
            overall_score=overall_score
        )
    
    def _analyze_class_complexity(self, class_node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Analyze complexity metrics for a class"""
        methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
        
        method_complexities = []
        total_complexity = 0
        
        for method in methods:
            method_metrics = self._analyze_function_complexity(method, content)
            method_complexities.append(method_metrics.overall_score)
            total_complexity += method_metrics.cyclomatic_complexity
        
        line_count = (class_node.end_lineno or class_node.lineno) - class_node.lineno + 1
        
        return {
            'method_count': len(methods),
            'total_complexity': total_complexity,
            'average_method_complexity': sum(method_complexities) / len(method_complexities) if method_complexities else 0,
            'max_method_complexity': max(method_complexities) if method_complexities else 0,
            'line_count': line_count,
            'complexity_distribution': method_complexities
        }
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate McCabe cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # Add complexity for each boolean operator
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ExceptHandler, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _calculate_cognitive_complexity(self, node: ast.AST) -> int:
        """Calculate cognitive complexity (more human-oriented than cyclomatic)"""
        complexity = 0
        nesting_level = 0
        
        def calculate_recursive(node, level):
            nonlocal complexity
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1 + level
                level += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                complexity += 1 + level
                level += 1
            elif isinstance(node, (ast.Break, ast.Continue)):
                complexity += 1 + level
            
            for child in ast.iter_child_nodes(node):
                calculate_recursive(child, level)
        
        calculate_recursive(node, 0)
        return complexity
    
    def _calculate_max_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        
        def calculate_depth(node, current_depth):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try)):
                current_depth += 1
            
            for child in ast.iter_child_nodes(node):
                calculate_depth(child, current_depth)
        
        calculate_depth(node, 0)
        return max_depth
    
    def _count_branches(self, node: ast.AST) -> int:
        """Count branching statements"""
        return len([n for n in ast.walk(node) if isinstance(n, (ast.If, ast.IfExp))])
    
    def _count_loops(self, node: ast.AST) -> int:
        """Count loop statements"""
        return len([n for n in ast.walk(node) if isinstance(n, (ast.While, ast.For, ast.AsyncFor))])
    
    def _identify_complexity_issues(self, func_node: ast.FunctionDef, 
                                  metrics: ComplexityMetrics, 
                                  file_path: Path) -> List[ComplexityIssue]:
        """Identify complexity issues in a function"""
        issues = []
        
        if metrics.cyclomatic_complexity > self.cyclomatic_threshold:
            issues.append(ComplexityIssue(
                file_path=str(file_path),
                line_number=func_node.lineno,
                function_or_class=func_node.name,
                issue_type='cyclomatic_complexity',
                severity=self._calculate_severity(metrics.cyclomatic_complexity, self.cyclomatic_threshold),
                current_complexity=metrics.cyclomatic_complexity,
                recommended_complexity=self.cyclomatic_threshold,
                description=f"Function has cyclomatic complexity of {metrics.cyclomatic_complexity}",
                refactoring_suggestion="Split function into smaller functions, reduce branching logic",
                estimated_effort=min(10, metrics.cyclomatic_complexity - self.cyclomatic_threshold + 3)
            ))
        
        if metrics.cognitive_complexity > self.cognitive_threshold:
            issues.append(ComplexityIssue(
                file_path=str(file_path),
                line_number=func_node.lineno,
                function_or_class=func_node.name,
                issue_type='cognitive_complexity',
                severity=self._calculate_severity(metrics.cognitive_complexity, self.cognitive_threshold),
                current_complexity=metrics.cognitive_complexity,
                recommended_complexity=self.cognitive_threshold,
                description=f"Function has cognitive complexity of {metrics.cognitive_complexity}",
                refactoring_suggestion="Reduce nesting, extract complex logic to separate functions",
                estimated_effort=min(10, metrics.cognitive_complexity - self.cognitive_threshold + 3)
            ))
        
        if metrics.nesting_depth > self.nesting_threshold:
            issues.append(ComplexityIssue(
                file_path=str(file_path),
                line_number=func_node.lineno,
                function_or_class=func_node.name,
                issue_type='nesting_depth',
                severity=self._calculate_severity(metrics.nesting_depth, self.nesting_threshold),
                current_complexity=metrics.nesting_depth,
                recommended_complexity=self.nesting_threshold,
                description=f"Function has nesting depth of {metrics.nesting_depth}",
                refactoring_suggestion="Use early returns, extract nested logic, apply guard clauses",
                estimated_effort=min(8, metrics.nesting_depth - self.nesting_threshold + 2)
            ))
        
        if metrics.line_count > self.line_threshold:
            issues.append(ComplexityIssue(
                file_path=str(file_path),
                line_number=func_node.lineno,
                function_or_class=func_node.name,
                issue_type='function_size',
                severity=self._calculate_severity(metrics.line_count, self.line_threshold),
                current_complexity=metrics.line_count,
                recommended_complexity=self.line_threshold,
                description=f"Function has {metrics.line_count} lines",
                refactoring_suggestion="Split into smaller, focused functions",
                estimated_effort=min(8, (metrics.line_count - self.line_threshold) // 10 + 2)
            ))
        
        if metrics.parameter_count > self.parameter_threshold:
            issues.append(ComplexityIssue(
                file_path=str(file_path),
                line_number=func_node.lineno,
                function_or_class=func_node.name,
                issue_type='parameter_count',
                severity=self._calculate_severity(metrics.parameter_count, self.parameter_threshold),
                current_complexity=metrics.parameter_count,
                recommended_complexity=self.parameter_threshold,
                description=f"Function has {metrics.parameter_count} parameters",
                refactoring_suggestion="Group related parameters into objects, use builder pattern",
                estimated_effort=min(6, metrics.parameter_count - self.parameter_threshold + 2)
            ))
        
        return issues
    
    def _identify_class_complexity_issues(self, class_node: ast.ClassDef, 
                                        metrics: Dict[str, Any], 
                                        file_path: Path) -> List[ComplexityIssue]:
        """Identify complexity issues at class level"""
        issues = []
        
        if metrics['method_count'] > 20:
            issues.append(ComplexityIssue(
                file_path=str(file_path),
                line_number=class_node.lineno,
                function_or_class=class_node.name,
                issue_type='class_size',
                severity='high' if metrics['method_count'] > 30 else 'medium',
                current_complexity=metrics['method_count'],
                recommended_complexity=20,
                description=f"Class has {metrics['method_count']} methods",
                refactoring_suggestion="Split class based on responsibilities (SRP)",
                estimated_effort=min(10, metrics['method_count'] // 5)
            ))
        
        if metrics['total_complexity'] > 50:
            issues.append(ComplexityIssue(
                file_path=str(file_path),
                line_number=class_node.lineno,
                function_or_class=class_node.name,
                issue_type='class_complexity',
                severity='high' if metrics['total_complexity'] > 80 else 'medium',
                current_complexity=metrics['total_complexity'],
                recommended_complexity=50,
                description=f"Class has total complexity of {metrics['total_complexity']}",
                refactoring_suggestion="Refactor complex methods, split class responsibilities",
                estimated_effort=min(10, metrics['total_complexity'] // 10)
            ))
        
        return issues
    
    def _calculate_severity(self, current: int, threshold: int) -> str:
        """Calculate severity based on how much threshold is exceeded"""
        ratio = current / threshold
        if ratio >= 2.5:
            return 'critical'
        elif ratio >= 2.0:
            return 'high'
        elif ratio >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _aggregate_results(self, results: Dict[str, Any]) -> None:
        """Aggregate complexity results across all files"""
        all_issues = []
        total_functions = 0
        total_classes = 0
        complexity_sum = 0
        
        for file_path, file_results in results['file_metrics'].items():
            if 'error' in file_results:
                continue
                
            all_issues.extend(file_results['complexity_issues'])
            
            # Aggregate function metrics
            for func_name, func_metrics in file_results['functions'].items():
                total_functions += 1
                complexity_sum += func_metrics.overall_score
                results['function_metrics'][f"{file_path}:{func_name}"] = func_metrics
            
            # Aggregate class metrics
            for class_name, class_metrics in file_results['classes'].items():
                total_classes += 1
                results['class_metrics'][f"{file_path}:{class_name}"] = class_metrics
        
        results['complexity_issues'] = all_issues
        results['overall_statistics'] = {
            'total_functions': total_functions,
            'total_classes': total_classes,
            'average_function_complexity': complexity_sum / total_functions if total_functions > 0 else 0,
            'total_issues': len(all_issues),
            'critical_issues': len([i for i in all_issues if i.severity == 'critical']),
            'high_issues': len([i for i in all_issues if i.severity == 'high']),
            'medium_issues': len([i for i in all_issues if i.severity == 'medium']),
            'low_issues': len([i for i in all_issues if i.severity == 'low'])
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate complexity reduction recommendations"""
        recommendations = []
        stats = results['overall_statistics']
        
        if stats['critical_issues'] > 0:
            recommendations.append(f"Urgently refactor {stats['critical_issues']} functions with critical complexity")
        
        if stats['high_issues'] > 0:
            recommendations.append(f"Prioritize refactoring {stats['high_issues']} functions with high complexity")
        
        if stats['average_function_complexity'] > 7:
            recommendations.append("Overall codebase complexity is high - consider architectural refactoring")
        
        # Issue-specific recommendations
        issue_types = {}
        for issue in results['complexity_issues']:
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
        
        for issue_type, count in issue_types.items():
            if issue_type == 'cyclomatic_complexity':
                recommendations.append(f"Reduce branching logic in {count} functions")
            elif issue_type == 'nesting_depth':
                recommendations.append(f"Flatten nested structures in {count} functions")
            elif issue_type == 'function_size':
                recommendations.append(f"Split {count} large functions into smaller ones")
            elif issue_type == 'parameter_count':
                recommendations.append(f"Reduce parameter counts in {count} functions")
        
        return recommendations[:8]  # Top 8 recommendations
    
    def _prioritize_refactoring(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create prioritized refactoring plan"""
        priorities = []
        
        for issue in results['complexity_issues']:
            impact = {'critical': 10, 'high': 8, 'medium': 5, 'low': 2}[issue.severity]
            
            priorities.append({
                'type': 'complexity_reduction',
                'target': f"{issue.file_path}:{issue.line_number}",
                'issue_type': issue.issue_type,
                'description': issue.description,
                'impact': impact,
                'effort': issue.estimated_effort,
                'current_complexity': issue.current_complexity,
                'target_complexity': issue.recommended_complexity,
                'suggested_fix': issue.refactoring_suggestion
            })
        
        return sorted(priorities, key=lambda x: (x['impact'], -x['effort']), reverse=True)[:10]
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall complexity health score"""
        stats = results['overall_statistics']
        
        if stats['total_functions'] == 0:
            return 10.0
        
        # Base score from average complexity
        base_score = max(0, 10 - stats['average_function_complexity'])
        
        # Penalties for issues
        issue_penalty = (
            stats['critical_issues'] * 2 +
            stats['high_issues'] * 1 +
            stats['medium_issues'] * 0.5 +
            stats['low_issues'] * 0.1
        ) / stats['total_functions']
        
        overall_score = max(0, base_score - issue_penalty)
        return round(overall_score, 2)
    
    async def reduce_complexity(self, target: str, complexity_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to reduce complexity automatically
        
        Args:
            target: File path and line number
            complexity_info: Information about the complexity issue
            
        Returns:
            Result of complexity reduction attempt
        """
        self.logger.info(f"Attempting to reduce complexity: {complexity_info.get('description')}")
        
        # Placeholder for actual refactoring logic
        return {
            'success': False,
            'message': "Automatic complexity reduction is not yet implemented",
            'manual_steps': complexity_info.get('suggested_fix', 'No suggestions available')
        }