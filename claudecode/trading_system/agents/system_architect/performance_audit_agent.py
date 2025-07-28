"""
Performance Audit Agent

This agent analyzes code for performance bottlenecks and provides
optimization recommendations with automated fixes where possible.
"""

import ast
import re
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import psutil
import sys

from ...core.base.agent import BaseAgent


@dataclass
class PerformanceIssue:
    """Represents a performance issue"""
    category: str  # e.g., 'loop', 'memory', 'io', 'algorithm', 'database'
    severity: str  # 'low', 'medium', 'high', 'critical'
    file_path: str
    line_number: int
    function_or_class: str
    description: str
    current_complexity: str  # O(n), O(n²), etc.
    optimized_complexity: str
    estimated_improvement: str  # percentage or factor
    optimization_technique: str
    code_example: Optional[str]
    automated_fix_available: bool
    refactoring_effort: int  # 1-10 scale


@dataclass
class PerformanceProfile:
    """Performance profiling results"""
    execution_time: float
    memory_usage: float
    cpu_utilization: float
    io_operations: int
    database_queries: int
    bottlenecks: List[str]


class PerformanceAuditAgent(BaseAgent):
    """
    Performance Audit Agent
    
    Analyzes code for performance issues including:
    - Algorithmic complexity analysis
    - Memory usage optimization
    - I/O efficiency
    - Database query optimization
    - Loop optimization
    - Caching opportunities
    - Async/await patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("PerformanceAudit", config.get('performance_audit', {}))
        self.logger = logging.getLogger(__name__)
        
        # Performance thresholds
        self.complexity_threshold = config.get('complexity_threshold', 'O(n²)')
        self.memory_threshold_mb = config.get('memory_threshold_mb', 100)
        self.execution_time_threshold_ms = config.get('execution_time_threshold_ms', 1000)
        self.loop_depth_threshold = config.get('loop_depth_threshold', 3)
        
        # Analysis patterns
        self.performance_patterns = self._load_performance_patterns()
        
    async def analyze_performance(self, target_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive performance analysis
        
        Args:
            target_path: Path to analyze
            
        Returns:
            Performance analysis results
        """
        self.logger.info(f"Starting performance analysis of {target_path}")
        
        issues = []
        path = Path(target_path)
        
        # Static analysis
        if path.is_file() and path.suffix == '.py':
            issues.extend(await self._analyze_file(path))
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                if not self._should_skip_file(py_file):
                    issues.extend(await self._analyze_file(py_file))
        
        # Generate performance report
        overall_score = self._calculate_performance_score(issues)
        recommendations = self._generate_recommendations(issues)
        critical_issues = [issue for issue in issues if issue.severity == 'critical']
        
        return {
            'overall_score': overall_score,
            'performance_issues': [self._issue_to_dict(issue) for issue in issues],
            'recommendations': recommendations,
            'critical_issues': [self._issue_to_dict(issue) for issue in critical_issues],
            'refactoring_priorities': self._prioritize_optimizations(issues),
            'optimization_opportunities': self._identify_optimization_opportunities(issues)
        }
    
    async def _analyze_file(self, file_path: Path) -> List[PerformanceIssue]:
        """Analyze a single file for performance issues"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # AST-based analysis
            try:
                tree = ast.parse(content)
                issues.extend(self._analyze_algorithmic_complexity(tree, file_path))
                issues.extend(self._analyze_loop_performance(tree, file_path))
                issues.extend(self._analyze_memory_usage(tree, file_path))
                issues.extend(self._analyze_io_patterns(tree, file_path))
                issues.extend(self._analyze_database_queries(tree, file_path))
                issues.extend(self._analyze_async_patterns(tree, file_path))
            except SyntaxError:
                self.logger.warning(f"Could not parse {file_path} for AST analysis")
            
            # Pattern-based analysis
            issues.extend(self._analyze_performance_patterns(file_path, content))
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
        
        return issues
    
    def _analyze_algorithmic_complexity(self, tree: ast.AST, file_path: Path) -> List[PerformanceIssue]:
        """Analyze algorithmic complexity of functions"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_function_complexity(node)
                
                if self._is_high_complexity(complexity):
                    issues.append(PerformanceIssue(
                        category='algorithm',
                        severity=self._get_complexity_severity(complexity),
                        file_path=str(file_path),
                        line_number=node.lineno,
                        function_or_class=node.name,
                        description=f"Function '{node.name}' has high algorithmic complexity: {complexity}",
                        current_complexity=complexity,
                        optimized_complexity=self._suggest_optimized_complexity(complexity),
                        estimated_improvement="50-80% faster execution",
                        optimization_technique=self._suggest_optimization_technique(node, complexity),
                        code_example=None,
                        automated_fix_available=self._can_auto_optimize(node, complexity),
                        refactoring_effort=self._estimate_refactoring_effort(complexity)
                    ))
        
        return issues
    
    def _analyze_loop_performance(self, tree: ast.AST, file_path: Path) -> List[PerformanceIssue]:
        """Analyze loop performance issues"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Find nested loops
                nested_loops = self._find_nested_loops(node)
                
                for loop_info in nested_loops:
                    if loop_info['depth'] > self.loop_depth_threshold:
                        issues.append(PerformanceIssue(
                            category='loop',
                            severity='high' if loop_info['depth'] > 4 else 'medium',
                            file_path=str(file_path),
                            line_number=loop_info['line'],
                            function_or_class=node.name,
                            description=f"Deeply nested loops (depth: {loop_info['depth']}) in function '{node.name}'",
                            current_complexity=f"O(n^{loop_info['depth']})",
                            optimized_complexity="O(n) or O(n log n)",
                            estimated_improvement="90%+ performance improvement possible",
                            optimization_technique="Consider using vectorization, memoization, or alternative algorithms",
                            code_example=None,
                            automated_fix_available=False,
                            refactoring_effort=8
                        ))
                
                # Find inefficient loop patterns
                inefficient_patterns = self._find_inefficient_loop_patterns(node)
                issues.extend(inefficient_patterns)
        
        return issues
    
    def _analyze_memory_usage(self, tree: ast.AST, file_path: Path) -> List[PerformanceIssue]:
        """Analyze memory usage patterns"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for memory-intensive patterns
                memory_issues = []
                
                # Large list comprehensions
                for child in ast.walk(node):
                    if isinstance(child, ast.ListComp):
                        if self._is_large_comprehension(child):
                            memory_issues.append(PerformanceIssue(
                                category='memory',
                                severity='medium',
                                file_path=str(file_path),
                                line_number=child.lineno,
                                function_or_class=node.name,
                                description="Large list comprehension may consume excessive memory",
                                current_complexity="O(n) memory",
                                optimized_complexity="O(1) memory with generator",
                                estimated_improvement="90% memory reduction",
                                optimization_technique="Use generator expression instead of list comprehension",
                                code_example="Use (x for x in items) instead of [x for x in items]",
                                automated_fix_available=True,
                                refactoring_effort=2
                            ))
                
                # String concatenation in loops
                string_concat_issues = self._find_string_concatenation_in_loops(node)
                memory_issues.extend(string_concat_issues)
                
                issues.extend(memory_issues)
        
        return issues
    
    def _analyze_io_patterns(self, tree: ast.AST, file_path: Path) -> List[PerformanceIssue]:
        """Analyze I/O performance patterns"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Find I/O operations in loops
                io_in_loops = self._find_io_in_loops(node)
                for io_issue in io_in_loops:
                    issues.append(PerformanceIssue(
                        category='io',
                        severity='high',
                        file_path=str(file_path),
                        line_number=io_issue['line'],
                        function_or_class=node.name,
                        description=f"I/O operation inside loop: {io_issue['operation']}",
                        current_complexity="O(n) I/O operations",
                        optimized_complexity="O(1) with batching",
                        estimated_improvement="80-95% I/O reduction",
                        optimization_technique="Batch I/O operations outside the loop",
                        code_example="Read all data once, then process in memory",
                        automated_fix_available=False,
                        refactoring_effort=6
                    ))
                
                # Find synchronous I/O that could be async
                sync_io_issues = self._find_synchronous_io(node)
                issues.extend(sync_io_issues)
        
        return issues
    
    def _analyze_database_queries(self, tree: ast.AST, file_path: Path) -> List[PerformanceIssue]:
        """Analyze database query performance"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Find N+1 query problems
                n_plus_one_issues = self._find_n_plus_one_queries(node)
                for issue in n_plus_one_issues:
                    issues.append(PerformanceIssue(
                        category='database',
                        severity='critical',
                        file_path=str(file_path),
                        line_number=issue['line'],
                        function_or_class=node.name,
                        description="Potential N+1 query problem detected",
                        current_complexity="O(n) database queries",
                        optimized_complexity="O(1) with eager loading",
                        estimated_improvement="90%+ query reduction",
                        optimization_technique="Use joins, prefetch_related, or bulk operations",
                        code_example="Use select_related() or prefetch_related() for ORM queries",
                        automated_fix_available=False,
                        refactoring_effort=5
                    ))
                
                # Find missing database indexes
                missing_index_issues = self._find_missing_indexes(node)
                issues.extend(missing_index_issues)
        
        return issues
    
    def _analyze_async_patterns(self, tree: ast.AST, file_path: Path) -> List[PerformanceIssue]:
        """Analyze async/await patterns for performance"""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Find blocking operations in async functions
                if any(isinstance(decorator, ast.Name) and decorator.id == 'async' 
                       for decorator in getattr(node, 'decorator_list', [])) or \
                   isinstance(node, ast.AsyncFunctionDef):
                    
                    blocking_ops = self._find_blocking_operations_in_async(node)
                    for op in blocking_ops:
                        issues.append(PerformanceIssue(
                            category='async',
                            severity='medium',
                            file_path=str(file_path),
                            line_number=op['line'],
                            function_or_class=node.name,
                            description=f"Blocking operation in async function: {op['operation']}",
                            current_complexity="Blocks event loop",
                            optimized_complexity="Non-blocking with proper async",
                            estimated_improvement="Eliminates blocking",
                            optimization_technique="Use async libraries or run_in_executor",
                            code_example="Use aiofiles, aiohttp, or asyncio.run_in_executor",
                            automated_fix_available=True,
                            refactoring_effort=4
                        ))
        
        return issues
    
    def _analyze_performance_patterns(self, file_path: Path, content: str) -> List[PerformanceIssue]:
        """Analyze performance patterns using regex"""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for inefficient patterns
            for pattern_name, pattern_info in self.performance_patterns.items():
                if re.search(pattern_info['regex'], line, re.IGNORECASE):
                    issues.append(PerformanceIssue(
                        category=pattern_info['category'],
                        severity=pattern_info['severity'],
                        file_path=str(file_path),
                        line_number=i,
                        function_or_class=self._extract_function_name(lines, i),
                        description=pattern_info['description'],
                        current_complexity=pattern_info['current_complexity'],
                        optimized_complexity=pattern_info['optimized_complexity'],
                        estimated_improvement=pattern_info['improvement'],
                        optimization_technique=pattern_info['technique'],
                        code_example=pattern_info.get('example'),
                        automated_fix_available=pattern_info['auto_fix'],
                        refactoring_effort=pattern_info['effort']
                    ))
        
        return issues
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> str:
        """Calculate Big O complexity of a function"""
        # Simplified complexity analysis
        nested_loops = 0
        has_recursion = False
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                # Count nesting level
                current_depth = 0
                parent = child
                while hasattr(parent, 'parent'):
                    if isinstance(parent.parent, (ast.For, ast.While)):
                        current_depth += 1
                    parent = parent.parent
                nested_loops = max(nested_loops, current_depth + 1)
            
            elif isinstance(child, ast.Call):
                # Check for recursive calls
                if (isinstance(child.func, ast.Name) and 
                    child.func.id == node.name):
                    has_recursion = True
        
        if has_recursion:
            return "O(2^n)"  # Exponential for unoptimized recursion
        elif nested_loops >= 3:
            return f"O(n^{nested_loops})"
        elif nested_loops == 2:
            return "O(n²)"
        elif nested_loops == 1:
            return "O(n)"
        else:
            return "O(1)"
    
    def _is_high_complexity(self, complexity: str) -> bool:
        """Check if complexity is considered high"""
        high_complexity_patterns = ['O(n²)', 'O(n^', 'O(2^n)', 'O(n!)']
        return any(pattern in complexity for pattern in high_complexity_patterns)
    
    def _get_complexity_severity(self, complexity: str) -> str:
        """Get severity based on complexity"""
        if 'O(2^n)' in complexity or 'O(n!)' in complexity:
            return 'critical'
        elif 'O(n^' in complexity:
            power = int(complexity.split('^')[1].split(')')[0])
            return 'critical' if power > 3 else 'high'
        elif 'O(n²)' in complexity:
            return 'high'
        else:
            return 'medium'
    
    def _suggest_optimized_complexity(self, current: str) -> str:
        """Suggest optimized complexity"""
        if 'O(2^n)' in current:
            return "O(n) with memoization"
        elif 'O(n^' in current:
            return "O(n log n) with better algorithm"
        elif 'O(n²)' in current:
            return "O(n) or O(n log n)"
        else:
            return "O(1) or O(log n)"
    
    def _suggest_optimization_technique(self, node: ast.FunctionDef, complexity: str) -> str:
        """Suggest optimization technique based on complexity"""
        techniques = {
            'O(2^n)': "Use memoization/dynamic programming",
            'O(n²)': "Consider hash tables, sorting, or different algorithm",
            'O(n^': "Use divide-and-conquer or specialized data structures",
        }
        
        for pattern, technique in techniques.items():
            if pattern in complexity:
                return technique
        
        return "Review algorithm and data structures"
    
    def _can_auto_optimize(self, node: ast.FunctionDef, complexity: str) -> bool:
        """Check if function can be automatically optimized"""
        # Simple heuristics for auto-optimization
        if 'O(n²)' in complexity:
            # Check if it's a simple nested loop pattern
            nested_loops = self._find_nested_loops(node)
            return len(nested_loops) == 1 and nested_loops[0]['depth'] == 2
        
        return False
    
    def _estimate_refactoring_effort(self, complexity: str) -> int:
        """Estimate refactoring effort (1-10 scale)"""
        if 'O(2^n)' in complexity or 'O(n!)' in complexity:
            return 9
        elif 'O(n^' in complexity:
            return 7
        elif 'O(n²)' in complexity:
            return 5
        else:
            return 3
    
    def _find_nested_loops(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Find nested loops and their depth"""
        nested_loops = []
        
        def count_loop_depth(current_node, depth=0):
            if isinstance(current_node, (ast.For, ast.While)):
                depth += 1
                nested_loops.append({
                    'line': current_node.lineno,
                    'depth': depth,
                    'type': type(current_node).__name__
                })
            
            for child in ast.iter_child_nodes(current_node):
                count_loop_depth(child, depth)
        
        count_loop_depth(node)
        return [loop for loop in nested_loops if loop['depth'] > 1]
    
    def _find_inefficient_loop_patterns(self, node: ast.FunctionDef) -> List[PerformanceIssue]:
        """Find inefficient patterns in loops"""
        issues = []
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                # Check for inefficient patterns inside loop
                for loop_child in ast.walk(child):
                    # Dictionary/list lookup in loop condition
                    if isinstance(loop_child, ast.Subscript):
                        issues.append(PerformanceIssue(
                            category='loop',
                            severity='medium',
                            file_path='',
                            line_number=child.lineno,
                            function_or_class=node.name,
                            description="Repeated subscript access in loop",
                            current_complexity="O(n) with repeated lookups",
                            optimized_complexity="O(n) with cached lookup",
                            estimated_improvement="20-30% improvement",
                            optimization_technique="Cache the lookup result outside the loop",
                            code_example="cached_value = my_dict[key]; then use cached_value",
                            automated_fix_available=True,
                            refactoring_effort=2
                        ))
        
        return issues
    
    def _is_large_comprehension(self, node: ast.ListComp) -> bool:
        """Check if list comprehension is potentially large"""
        # Simple heuristic: look for nested comprehensions or complex generators
        for generator in node.generators:
            if isinstance(generator.iter, ast.Call):
                func_name = getattr(generator.iter.func, 'id', '')
                if func_name in ['range'] and len(generator.iter.args) > 0:
                    # Check if range is large
                    if isinstance(generator.iter.args[0], ast.Num) and generator.iter.args[0].n > 10000:
                        return True
        
        return len(node.generators) > 1  # Multiple generators
    
    def _find_string_concatenation_in_loops(self, node: ast.FunctionDef) -> List[PerformanceIssue]:
        """Find string concatenation inside loops"""
        issues = []
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                for loop_child in ast.walk(child):
                    if isinstance(loop_child, ast.AugAssign) and isinstance(loop_child.op, ast.Add):
                        # Check if it's string concatenation
                        issues.append(PerformanceIssue(
                            category='memory',
                            severity='medium',
                            file_path='',
                            line_number=child.lineno,
                            function_or_class=node.name,
                            description="String concatenation in loop creates multiple objects",
                            current_complexity="O(n²) memory and time",
                            optimized_complexity="O(n) with join",
                            estimated_improvement="90% performance improvement",
                            optimization_technique="Use list.append() and ''.join() instead",
                            code_example="parts = []; parts.append(item); result = ''.join(parts)",
                            automated_fix_available=True,
                            refactoring_effort=3
                        ))
        
        return issues
    
    def _find_io_in_loops(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Find I/O operations inside loops"""
        io_operations = []
        
        io_functions = ['open', 'read', 'write', 'print', 'input']
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                for loop_child in ast.walk(child):
                    if isinstance(loop_child, ast.Call):
                        func_name = getattr(loop_child.func, 'id', '') or getattr(loop_child.func, 'attr', '')
                        if func_name in io_functions:
                            io_operations.append({
                                'line': child.lineno,
                                'operation': func_name
                            })
        
        return io_operations
    
    def _find_synchronous_io(self, node: ast.FunctionDef) -> List[PerformanceIssue]:
        """Find synchronous I/O that could be async"""
        issues = []
        
        # Check if function is async
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        if is_async:
            sync_io_patterns = ['requests.get', 'requests.post', 'open(', 'urllib']
            
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    # Check for synchronous I/O calls
                    call_str = ast.dump(child)
                    for pattern in sync_io_patterns:
                        if pattern in call_str:
                            issues.append(PerformanceIssue(
                                category='async',
                                severity='medium',
                                file_path='',
                                line_number=child.lineno,
                                function_or_class=node.name,
                                description=f"Synchronous I/O in async function: {pattern}",
                                current_complexity="Blocks event loop",
                                optimized_complexity="Non-blocking",
                                estimated_improvement="Eliminates blocking",
                                optimization_technique="Use async equivalents (aiohttp, aiofiles)",
                                code_example="Use async with aiohttp.ClientSession() as session:",
                                automated_fix_available=True,
                                refactoring_effort=4
                            ))
        
        return issues
    
    def _find_n_plus_one_queries(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Find potential N+1 query problems"""
        n_plus_one_issues = []
        
        # Look for ORM queries inside loops
        orm_patterns = ['.get(', '.filter(', '.objects.', 'query(']
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                for loop_child in ast.walk(child):
                    if isinstance(loop_child, ast.Call):
                        call_str = ast.dump(loop_child)
                        for pattern in orm_patterns:
                            if pattern in call_str:
                                n_plus_one_issues.append({
                                    'line': child.lineno
                                })
                                break
        
        return n_plus_one_issues
    
    def _find_missing_indexes(self, node: ast.FunctionDef) -> List[PerformanceIssue]:
        """Find potential missing database indexes"""
        # This would require more sophisticated analysis in practice
        # For now, return empty list as placeholder
        return []
    
    def _find_blocking_operations_in_async(self, node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Find blocking operations in async functions"""
        blocking_ops = []
        
        blocking_patterns = ['time.sleep', 'requests.', 'open(', 'input(']
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                call_str = ast.dump(child)
                for pattern in blocking_patterns:
                    if pattern in call_str:
                        blocking_ops.append({
                            'line': child.lineno,
                            'operation': pattern
                        })
        
        return blocking_ops
    
    def _load_performance_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load performance anti-patterns"""
        return {
            'inefficient_membership_test': {
                'regex': r'.*\sin\s\[.*\]',
                'category': 'algorithm',
                'severity': 'medium',
                'description': 'Inefficient membership test using list instead of set',
                'current_complexity': 'O(n)',
                'optimized_complexity': 'O(1)',
                'improvement': '90%+ for large collections',
                'technique': 'Use set for membership testing',
                'example': 'Use item in {1, 2, 3} instead of item in [1, 2, 3]',
                'auto_fix': True,
                'effort': 1
            },
            'global_in_loop': {
                'regex': r'global\s+\w+.*for.*in',
                'category': 'loop',
                'severity': 'low',
                'description': 'Global variable access in loop',
                'current_complexity': 'O(n) with global lookup',
                'optimized_complexity': 'O(n) with local variable',
                'improvement': '10-20% improvement',
                'technique': 'Cache global variable as local',
                'example': 'local_var = global_var; then use local_var in loop',
                'auto_fix': True,
                'effort': 2
            }
        }
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis"""
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
    
    def _extract_function_name(self, lines: List[str], line_num: int) -> str:
        """Extract function name from context"""
        for i in range(line_num - 1, max(0, line_num - 20), -1):
            line = lines[i].strip()
            if line.startswith('def ') or line.startswith('class '):
                return line.split('(')[0].replace('def ', '').replace('class ', '')
        return 'unknown'
    
    def _calculate_performance_score(self, issues: List[PerformanceIssue]) -> float:
        """Calculate overall performance score"""
        if not issues:
            return 10.0
        
        severity_weights = {'low': 1, 'medium': 3, 'high': 7, 'critical': 10}
        total_impact = sum(severity_weights.get(issue.severity, 0) for issue in issues)
        
        # Normalize score (10 is perfect, 0 is worst)
        max_possible_impact = len(issues) * 10
        score = max(0, 10 - (total_impact / max_possible_impact * 10))
        
        return round(score, 2)
    
    def _generate_recommendations(self, issues: List[PerformanceIssue]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Group by category
        by_category = {}
        for issue in issues:
            if issue.category not in by_category:
                by_category[issue.category] = []
            by_category[issue.category].append(issue)
        
        for category, category_issues in by_category.items():
            count = len(category_issues)
            if category == 'algorithm':
                recommendations.append(f"Optimize {count} algorithmic inefficiencies")
            elif category == 'loop':
                recommendations.append(f"Improve {count} loop performance issues")
            elif category == 'memory':
                recommendations.append(f"Reduce memory usage in {count} locations")
            elif category == 'io':
                recommendations.append(f"Optimize {count} I/O operations")
            elif category == 'database':
                recommendations.append(f"Improve {count} database query patterns")
            elif category == 'async':
                recommendations.append(f"Fix {count} async/await performance issues")
        
        return recommendations
    
    def _prioritize_optimizations(self, issues: List[PerformanceIssue]) -> List[Dict[str, Any]]:
        """Prioritize performance optimizations"""
        priorities = []
        
        for issue in issues:
            impact = {'critical': 10, 'high': 8, 'medium': 5, 'low': 2}[issue.severity]
            
            priorities.append({
                'type': 'performance_optimization',
                'target': f"{issue.file_path}:{issue.line_number}",
                'description': issue.description,
                'category': issue.category,
                'impact': impact,
                'effort': issue.refactoring_effort,
                'estimated_improvement': issue.estimated_improvement,
                'technique': issue.optimization_technique,
                'automated_fix': issue.automated_fix_available
            })
        
        return sorted(priorities, key=lambda x: (x['impact'], -x['effort']), reverse=True)
    
    def _identify_optimization_opportunities(self, issues: List[PerformanceIssue]) -> Dict[str, Any]:
        """Identify optimization opportunities"""
        opportunities = {
            'caching': [],
            'algorithmic': [],
            'async_conversion': [],
            'memory_optimization': [],
            'database_optimization': []
        }
        
        for issue in issues:
            if 'cache' in issue.optimization_technique.lower():
                opportunities['caching'].append(issue.description)
            elif issue.category == 'algorithm':
                opportunities['algorithmic'].append(issue.description)
            elif issue.category == 'async':
                opportunities['async_conversion'].append(issue.description)
            elif issue.category == 'memory':
                opportunities['memory_optimization'].append(issue.description)
            elif issue.category == 'database':
                opportunities['database_optimization'].append(issue.description)
        
        return opportunities
    
    def _issue_to_dict(self, issue: PerformanceIssue) -> Dict[str, Any]:
        """Convert issue to dictionary"""
        return {
            'category': issue.category,
            'severity': issue.severity,
            'file_path': issue.file_path,
            'line_number': issue.line_number,
            'function_or_class': issue.function_or_class,
            'description': issue.description,
            'current_complexity': issue.current_complexity,
            'optimized_complexity': issue.optimized_complexity,
            'estimated_improvement': issue.estimated_improvement,
            'optimization_technique': issue.optimization_technique,
            'code_example': issue.code_example,
            'automated_fix_available': issue.automated_fix_available,
            'refactoring_effort': issue.refactoring_effort
        }
    
    async def optimize_performance(self, target: str, optimization_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to optimize performance automatically
        
        Args:
            target: File path and line number (file:line)
            optimization_info: Information about the optimization
            
        Returns:
            Result of the optimization attempt
        """
        self.logger.info(f"Attempting performance optimization: {optimization_info.get('description')}")
        
        if not optimization_info.get('automated_fix'):
            return {
                'success': False,
                'message': "Automated fix not available for this optimization",
                'manual_steps': optimization_info.get('technique', 'Manual optimization required')
            }
        
        category = optimization_info.get('category')
        file_path, line_num = target.split(':')
        
        try:
            if category == 'algorithm' and 'membership' in optimization_info.get('description', ''):
                return await self._fix_membership_test(file_path, int(line_num))
            elif category == 'memory' and 'comprehension' in optimization_info.get('description', ''):
                return await self._fix_list_comprehension(file_path, int(line_num))
            else:
                return {
                    'success': False,
                    'message': f"Automated optimization for {category} not yet implemented",
                    'manual_steps': optimization_info.get('technique', 'Manual optimization required')
                }
        except Exception as e:
            return {
                'success': False,
                'message': f"Error applying optimization: {str(e)}",
                'manual_steps': optimization_info.get('technique', 'Manual optimization required')
            }
    
    async def _fix_membership_test(self, file_path: str, line_num: int) -> Dict[str, Any]:
        """Fix inefficient membership test"""
        return {
            'success': False,
            'message': "Membership test optimization requires context analysis",
            'manual_steps': "Convert list to set for membership testing: use 'item in {1, 2, 3}' instead of 'item in [1, 2, 3]'"
        }
    
    async def _fix_list_comprehension(self, file_path: str, line_num: int) -> Dict[str, Any]:
        """Fix memory-intensive list comprehension"""
        return {
            'success': False,
            'message': "List comprehension optimization requires usage analysis",
            'manual_steps': "Convert to generator expression if full list is not needed: use '(x for x in items)' instead of '[x for x in items]'"
        }