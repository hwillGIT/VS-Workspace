"""
Code Metrics Dashboard

This agent generates comprehensive code metrics dashboards with real-time monitoring,
interactive visualizations, and automated reporting for the trading system.
"""

import ast
import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
import statistics
from collections import defaultdict, Counter
import re

from ...core.base.agent import BaseAgent


@dataclass
class CodeMetric:
    """Represents a single code metric"""
    name: str
    value: float
    unit: str
    category: str  # 'size', 'complexity', 'quality', 'maintainability', 'performance'
    threshold_warning: float
    threshold_critical: float
    status: str  # 'good', 'warning', 'critical'
    trend: str  # 'improving', 'stable', 'degrading'
    description: str
    impact: str  # 'low', 'medium', 'high'


@dataclass
class FileMetrics:
    """Metrics for a single file"""
    file_path: str
    lines_of_code: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    halstead_complexity: Dict[str, float]
    maintainability_index: float
    technical_debt_ratio: float
    test_coverage: float
    duplicate_lines: int
    code_smells: List[str]
    security_hotspots: int
    performance_issues: int
    last_modified: datetime
    author_count: int
    change_frequency: float


@dataclass
class ModuleMetrics:
    """Metrics for a module/package"""
    module_name: str
    file_count: int
    total_loc: int
    average_complexity: float
    coupling_index: float
    cohesion_index: float
    stability_index: float
    abstractness: float
    distance_from_main: float
    test_coverage: float
    technical_debt_hours: float
    quality_gate_status: str


@dataclass
class ProjectMetrics:
    """Overall project metrics"""
    total_files: int
    total_loc: int
    total_test_files: int
    total_test_loc: int
    overall_complexity: float
    overall_maintainability: float
    overall_coverage: float
    technical_debt_hours: float
    code_duplication: float
    security_rating: str
    reliability_rating: str
    maintainability_rating: str
    quality_gate_status: str


@dataclass
class MetricTrend:
    """Represents a metric trend over time"""
    metric_name: str
    timestamps: List[datetime]
    values: List[float]
    trend_direction: str  # 'up', 'down', 'stable'
    trend_strength: float  # 0.0 to 1.0
    change_rate: float
    prediction_next_week: float
    prediction_confidence: float


@dataclass
class QualityGate:
    """Quality gate configuration and status"""
    name: str
    conditions: List[Dict[str, Any]]
    status: str  # 'passed', 'failed', 'warning'
    failed_conditions: List[str]
    score: float
    threshold: float


class CodeMetricsDashboard(BaseAgent):
    """
    Code Metrics Dashboard Agent
    
    Provides comprehensive code metrics including:
    - Lines of code and file statistics
    - Cyclomatic and cognitive complexity
    - Halstead complexity metrics
    - Maintainability index
    - Technical debt assessment
    - Test coverage analysis
    - Code duplication detection
    - Security and performance metrics
    - Trend analysis and predictions
    - Quality gates and alerts
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("CodeMetricsDashboard", config.get('code_metrics', {}))
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.include_tests = config.get('include_tests', True)
        self.min_file_size = config.get('min_file_size', 10)  # lines
        self.complexity_threshold = config.get('complexity_threshold', 10)
        self.maintainability_threshold = config.get('maintainability_threshold', 20)
        self.coverage_threshold = config.get('coverage_threshold', 80.0)
        self.duplication_threshold = config.get('duplication_threshold', 5.0)
        
        # Metrics storage
        self.file_metrics: Dict[str, FileMetrics] = {}
        self.module_metrics: Dict[str, ModuleMetrics] = {}
        self.project_metrics: Optional[ProjectMetrics] = None
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Quality gates
        self.quality_gates = self._load_quality_gates()
        
    async def generate_dashboard(self, target_path: str) -> Dict[str, Any]:
        """
        Generate comprehensive code metrics dashboard
        
        Args:
            target_path: Path to analyze
            
        Returns:
            Complete dashboard data
        """
        self.logger.info(f"Generating code metrics dashboard for {target_path}")
        
        # Clear previous metrics
        self.file_metrics.clear()
        self.module_metrics.clear()
        
        # Collect metrics
        await self._collect_file_metrics(target_path)
        await self._calculate_module_metrics()
        await self._calculate_project_metrics()
        
        # Analyze trends
        trends = await self._analyze_metric_trends()
        
        # Check quality gates
        quality_status = self._check_quality_gates()
        
        # Generate alerts
        alerts = self._generate_alerts()
        
        # Create dashboard data
        dashboard_data = {
            'summary': self._create_summary_section(),
            'project_metrics': self._project_metrics_to_dict(),
            'module_metrics': [self._module_metrics_to_dict(m) for m in self.module_metrics.values()],
            'file_metrics': [self._file_metrics_to_dict(f) for f in self.file_metrics.values()],
            'trends': [self._trend_to_dict(t) for t in trends],
            'quality_gates': [self._quality_gate_to_dict(qg) for qg in quality_status],
            'alerts': alerts,
            'recommendations': self._generate_recommendations(),
            'charts': await self._generate_chart_data(),
            'export_data': self._prepare_export_data(),
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Save historical data
        await self._save_historical_data()
        
        return dashboard_data
    
    async def _collect_file_metrics(self, target_path: str) -> None:
        """Collect metrics for all files"""
        path = Path(target_path)
        
        if path.is_file() and path.suffix == '.py':
            await self._analyze_file(path)
        elif path.is_dir():
            for py_file in path.rglob('*.py'):
                if not self._should_skip_file(py_file):
                    await self._analyze_file(py_file)
    
    async def _analyze_file(self, file_path: Path) -> None:
        """Analyze metrics for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            
            # Basic metrics
            loc = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
            
            if loc < self.min_file_size:
                return  # Skip very small files
            
            # Parse AST for complexity analysis
            try:
                tree = ast.parse(content)
                
                # Calculate complexities
                cyclomatic = self._calculate_cyclomatic_complexity(tree)
                cognitive = self._calculate_cognitive_complexity(tree)
                halstead = self._calculate_halstead_complexity(tree)
                
                # Calculate maintainability index
                maintainability = self._calculate_maintainability_index(loc, cyclomatic, halstead)
                
                # Detect code smells
                code_smells = self._detect_code_smells(tree, content)
                
                # Calculate technical debt
                debt_ratio = self._calculate_technical_debt_ratio(code_smells, loc, cyclomatic)
                
                # Get file statistics
                file_stats = file_path.stat()
                last_modified = datetime.fromtimestamp(file_stats.st_mtime)
                
                # Create file metrics
                metrics = FileMetrics(
                    file_path=str(file_path),
                    lines_of_code=loc,
                    cyclomatic_complexity=cyclomatic,
                    cognitive_complexity=cognitive,
                    halstead_complexity=halstead,
                    maintainability_index=maintainability,
                    technical_debt_ratio=debt_ratio,
                    test_coverage=await self._estimate_test_coverage(file_path),
                    duplicate_lines=self._count_duplicate_lines(content),
                    code_smells=code_smells,
                    security_hotspots=self._count_security_hotspots(content),
                    performance_issues=self._count_performance_issues(tree),
                    last_modified=last_modified,
                    author_count=1,  # Would need git analysis for accurate count
                    change_frequency=0.0  # Would need git analysis
                )
                
                self.file_metrics[str(file_path)] = metrics
                
            except SyntaxError as e:
                self.logger.warning(f"Syntax error in {file_path}: {e}")
                
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            # Decision points add to complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With, ast.AsyncWith):
                complexity += 1
            elif isinstance(node, ast.Assert):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # And/Or operators
                complexity += len(node.values) - 1
            elif isinstance(node, ast.Compare):
                # Multiple comparisons
                complexity += len(node.comparators)
            elif isinstance(node, ast.comprehension):
                # List/dict/set comprehensions
                complexity += 1
                if node.ifs:
                    complexity += len(node.ifs)
        
        return complexity
    
    def _calculate_cognitive_complexity(self, tree: ast.AST) -> int:
        """Calculate cognitive complexity (focuses on human readability)"""
        complexity = 0
        nesting_level = 0
        
        def analyze_node(node, level=0):
            nonlocal complexity, nesting_level
            
            increment = 0
            
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                increment = 1 + level
            elif isinstance(node, ast.Try):
                increment = 1 + level
            elif isinstance(node, ast.ExceptHandler):
                increment = 1 + level
            elif isinstance(node, ast.BoolOp):
                increment = 1
            elif isinstance(node, ast.Continue, ast.Break):
                increment = 1 + level
            elif isinstance(node, ast.FunctionDef):
                # Nested functions increase complexity
                if level > 0:
                    increment = 1
            
            complexity += increment
            
            # Increase nesting for certain constructs
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.Try, ast.With, ast.AsyncWith)):
                level += 1
            
            for child in ast.iter_child_nodes(node):
                analyze_node(child, level)
        
        for node in ast.iter_child_nodes(tree):
            analyze_node(node)
        
        return complexity
    
    def _calculate_halstead_complexity(self, tree: ast.AST) -> Dict[str, float]:
        """Calculate Halstead complexity metrics"""
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0
        
        for node in ast.walk(tree):
            # Operators
            if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)):
                operators.add(type(node).__name__)
                operator_count += 1
            elif isinstance(node, (ast.And, ast.Or, ast.Not)):
                operators.add(type(node).__name__)
                operator_count += 1
            elif isinstance(node, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                operators.add(type(node).__name__)
                operator_count += 1
            elif isinstance(node, (ast.If, ast.While, ast.For)):
                operators.add(type(node).__name__)
                operator_count += 1
            
            # Operands
            elif isinstance(node, ast.Name):
                operands.add(node.id)
                operand_count += 1
            elif isinstance(node, (ast.Str, ast.Num, ast.Constant)):
                value = getattr(node, 'value', str(node))
                operands.add(str(value))
                operand_count += 1
        
        # Halstead metrics
        n1 = len(operators)  # Number of distinct operators
        n2 = len(operands)   # Number of distinct operands
        N1 = operator_count  # Total operators
        N2 = operand_count   # Total operands
        
        vocabulary = n1 + n2
        length = N1 + N2
        
        if n2 == 0:
            return {'vocabulary': 0, 'length': 0, 'difficulty': 0, 'effort': 0, 'volume': 0}
        
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        volume = length * (vocabulary.bit_length() if vocabulary > 0 else 0)
        effort = difficulty * volume
        
        return {
            'vocabulary': vocabulary,
            'length': length,
            'difficulty': difficulty,
            'effort': effort,
            'volume': volume
        }
    
    def _calculate_maintainability_index(self, loc: int, cyclomatic: int, halstead: Dict[str, float]) -> float:
        """Calculate maintainability index"""
        if loc == 0:
            return 100.0
        
        # Microsoft's maintainability index formula
        volume = halstead.get('volume', 0)
        
        if volume == 0:
            mi = 171 - 5.2 * (cyclomatic / loc * 100) - 0.23 * cyclomatic - 16.2 * (loc / 100)
        else:
            mi = 171 - 5.2 * (volume / 100) - 0.23 * cyclomatic - 16.2 * (loc / 100)
        
        # Normalize to 0-100 scale
        return max(0, min(100, mi))
    
    def _detect_code_smells(self, tree: ast.AST, content: str) -> List[str]:
        """Detect common code smells"""
        smells = []
        lines = content.split('\n')
        
        # Long method smell
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = getattr(node, 'end_lineno', node.lineno) - node.lineno + 1
                if func_lines > 50:
                    smells.append(f"Long method: {node.name} ({func_lines} lines)")
        
        # Large class smell
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_lines = getattr(node, 'end_lineno', node.lineno) - node.lineno + 1
                if class_lines > 200:
                    smells.append(f"Large class: {node.name} ({class_lines} lines)")
        
        # Too many parameters
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                param_count = len(node.args.args)
                if param_count > 7:
                    smells.append(f"Too many parameters: {node.name} ({param_count} params)")
        
        # Duplicate code (simple check for identical lines)
        line_counts = Counter(line.strip() for line in lines if line.strip())
        for line, count in line_counts.items():
            if count > 3 and len(line) > 20:  # Ignore short lines
                smells.append(f"Duplicate code: {count} occurrences of similar line")
        
        # Magic numbers
        for node in ast.walk(tree):
            if isinstance(node, ast.Num) and isinstance(node.n, (int, float)):
                if node.n not in [0, 1, -1] and abs(node.n) > 1:
                    smells.append(f"Magic number: {node.n}")
        
        # Long parameter lists in function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if len(node.args) > 5:
                    smells.append("Long parameter list in function call")
        
        return smells
    
    def _calculate_technical_debt_ratio(self, code_smells: List[str], loc: int, complexity: int) -> float:
        """Calculate technical debt ratio"""
        # Simple formula based on code smells, complexity, and size
        smell_weight = len(code_smells) * 0.1
        complexity_weight = max(0, (complexity - 10) * 0.05)
        size_weight = max(0, (loc - 100) * 0.001)
        
        debt_ratio = smell_weight + complexity_weight + size_weight
        return min(1.0, debt_ratio)  # Cap at 100%
    
    async def _estimate_test_coverage(self, file_path: Path) -> float:
        """Estimate test coverage (simplified - would need actual coverage tools)"""
        # Look for corresponding test file
        test_patterns = [
            file_path.parent / f"test_{file_path.name}",
            file_path.parent / f"{file_path.stem}_test.py",
            file_path.parent / "tests" / file_path.name,
            file_path.parent.parent / "tests" / file_path.name
        ]
        
        for test_path in test_patterns:
            if test_path.exists():
                try:
                    with open(test_path, 'r', encoding='utf-8') as f:
                        test_content = f.read()
                    
                    # Simple heuristic: count test functions
                    test_count = len(re.findall(r'def test_\w+', test_content))
                    
                    # Estimate coverage based on test count vs functions in source
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source_content = f.read()
                    
                    func_count = len(re.findall(r'def \w+', source_content))
                    
                    if func_count > 0:
                        coverage = min(100.0, (test_count / func_count) * 100)
                        return coverage
                        
                except Exception:
                    pass
        
        return 0.0  # No tests found
    
    def _count_duplicate_lines(self, content: str) -> int:
        """Count duplicate lines in content"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        line_counts = Counter(lines)
        
        duplicate_count = 0
        for line, count in line_counts.items():
            if count > 1 and len(line) > 10:  # Ignore short lines
                duplicate_count += count - 1  # Subtract original
        
        return duplicate_count
    
    def _count_security_hotspots(self, content: str) -> int:
        """Count potential security hotspots"""
        hotspot_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'pickle\.loads',
            r'yaml\.load\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'shell\s*=\s*True',
            r'password\s*=',
            r'secret\s*=',
            r'api_key\s*='
        ]
        
        hotspots = 0
        for pattern in hotspot_patterns:
            hotspots += len(re.findall(pattern, content, re.IGNORECASE))
        
        return hotspots
    
    def _count_performance_issues(self, tree: ast.AST) -> int:
        """Count potential performance issues"""
        issues = 0
        
        for node in ast.walk(tree):
            # Nested loops
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child != node:
                        issues += 1
            
            # String concatenation in loops
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                        issues += 1
            
            # Large list comprehensions
            if isinstance(node, ast.ListComp):
                if len(node.generators) > 1:
                    issues += 1
        
        return issues
    
    async def _calculate_module_metrics(self) -> None:
        """Calculate metrics for each module/package"""
        modules = defaultdict(list)
        
        # Group files by module
        for file_path, metrics in self.file_metrics.items():
            module_name = str(Path(file_path).parent).replace('\\', '.')
            modules[module_name].append(metrics)
        
        # Calculate module metrics
        for module_name, file_list in modules.items():
            total_loc = sum(f.lines_of_code for f in file_list)
            avg_complexity = statistics.mean([f.cyclomatic_complexity for f in file_list])
            avg_maintainability = statistics.mean([f.maintainability_index for f in file_list])
            total_debt_hours = sum(f.technical_debt_ratio * f.lines_of_code * 0.1 for f in file_list)
            avg_coverage = statistics.mean([f.test_coverage for f in file_list])
            
            # Quality gate status
            quality_status = self._determine_module_quality_status(avg_complexity, avg_maintainability, avg_coverage)
            
            module_metrics = ModuleMetrics(
                module_name=module_name,
                file_count=len(file_list),
                total_loc=total_loc,
                average_complexity=avg_complexity,
                coupling_index=0.0,  # Would need dependency analysis
                cohesion_index=0.0,  # Would need semantic analysis
                stability_index=0.0,  # Would need dependency analysis
                abstractness=0.0,    # Would need interface analysis
                distance_from_main=0.0,  # Would need architecture analysis
                test_coverage=avg_coverage,
                technical_debt_hours=total_debt_hours,
                quality_gate_status=quality_status
            )
            
            self.module_metrics[module_name] = module_metrics
    
    def _determine_module_quality_status(self, complexity: float, maintainability: float, coverage: float) -> str:
        """Determine quality gate status for a module"""
        if complexity > 15 or maintainability < 20 or coverage < 50:
            return 'failed'
        elif complexity > 10 or maintainability < 40 or coverage < 70:
            return 'warning'
        else:
            return 'passed'
    
    async def _calculate_project_metrics(self) -> None:
        """Calculate overall project metrics"""
        if not self.file_metrics:
            return
        
        # Basic counts
        total_files = len(self.file_metrics)
        total_loc = sum(f.lines_of_code for f in self.file_metrics.values())
        
        # Test files
        test_files = [f for f in self.file_metrics.values() if 'test' in f.file_path.lower()]
        total_test_files = len(test_files)
        total_test_loc = sum(f.lines_of_code for f in test_files)
        
        # Averages
        complexities = [f.cyclomatic_complexity for f in self.file_metrics.values()]
        maintainabilities = [f.maintainability_index for f in self.file_metrics.values()]
        coverages = [f.test_coverage for f in self.file_metrics.values()]
        
        overall_complexity = statistics.mean(complexities) if complexities else 0
        overall_maintainability = statistics.mean(maintainabilities) if maintainabilities else 0
        overall_coverage = statistics.mean(coverages) if coverages else 0
        
        # Technical debt
        total_debt_hours = sum(f.technical_debt_ratio * f.lines_of_code * 0.1 for f in self.file_metrics.values())
        
        # Code duplication
        total_duplicates = sum(f.duplicate_lines for f in self.file_metrics.values())
        duplication_percentage = (total_duplicates / total_loc * 100) if total_loc > 0 else 0
        
        # Ratings
        security_rating = self._calculate_security_rating()
        reliability_rating = self._calculate_reliability_rating()
        maintainability_rating = self._calculate_maintainability_rating(overall_maintainability)
        
        # Quality gate
        quality_gate_status = self._determine_project_quality_status(
            overall_complexity, overall_maintainability, overall_coverage, duplication_percentage
        )
        
        self.project_metrics = ProjectMetrics(
            total_files=total_files,
            total_loc=total_loc,
            total_test_files=total_test_files,
            total_test_loc=total_test_loc,
            overall_complexity=overall_complexity,
            overall_maintainability=overall_maintainability,
            overall_coverage=overall_coverage,
            technical_debt_hours=total_debt_hours,
            code_duplication=duplication_percentage,
            security_rating=security_rating,
            reliability_rating=reliability_rating,
            maintainability_rating=maintainability_rating,
            quality_gate_status=quality_gate_status
        )
    
    def _calculate_security_rating(self) -> str:
        """Calculate security rating based on hotspots"""
        total_hotspots = sum(f.security_hotspots for f in self.file_metrics.values())
        total_files = len(self.file_metrics)
        
        if total_files == 0:
            return 'A'
        
        hotspots_per_file = total_hotspots / total_files
        
        if hotspots_per_file > 5:
            return 'E'
        elif hotspots_per_file > 3:
            return 'D'
        elif hotspots_per_file > 2:
            return 'C'
        elif hotspots_per_file > 1:
            return 'B'
        else:
            return 'A'
    
    def _calculate_reliability_rating(self) -> str:
        """Calculate reliability rating based on complexity and smells"""
        total_smells = sum(len(f.code_smells) for f in self.file_metrics.values())
        total_complexity = sum(f.cyclomatic_complexity for f in self.file_metrics.values())
        total_files = len(self.file_metrics)
        
        if total_files == 0:
            return 'A'
        
        avg_smells = total_smells / total_files
        avg_complexity = total_complexity / total_files
        
        reliability_score = avg_smells + (avg_complexity - 5) / 2
        
        if reliability_score > 10:
            return 'E'
        elif reliability_score > 6:
            return 'D'
        elif reliability_score > 4:
            return 'C'
        elif reliability_score > 2:
            return 'B'
        else:
            return 'A'
    
    def _calculate_maintainability_rating(self, maintainability: float) -> str:
        """Calculate maintainability rating"""
        if maintainability >= 80:
            return 'A'
        elif maintainability >= 60:
            return 'B'
        elif maintainability >= 40:
            return 'C'
        elif maintainability >= 20:
            return 'D'
        else:
            return 'E'
    
    def _determine_project_quality_status(self, complexity: float, maintainability: float, 
                                        coverage: float, duplication: float) -> str:
        """Determine overall project quality gate status"""
        failed_conditions = []
        
        if complexity > self.complexity_threshold:
            failed_conditions.append(f"Average complexity too high: {complexity:.1f}")
        
        if maintainability < self.maintainability_threshold:
            failed_conditions.append(f"Maintainability too low: {maintainability:.1f}")
        
        if coverage < self.coverage_threshold:
            failed_conditions.append(f"Test coverage too low: {coverage:.1f}%")
        
        if duplication > self.duplication_threshold:
            failed_conditions.append(f"Code duplication too high: {duplication:.1f}%")
        
        if len(failed_conditions) > 2:
            return 'failed'
        elif len(failed_conditions) > 0:
            return 'warning'
        else:
            return 'passed'
    
    async def _analyze_metric_trends(self) -> List[MetricTrend]:
        """Analyze trends in metrics over time"""
        trends = []
        
        # This would analyze historical data
        # For now, return placeholder trends
        
        metric_names = ['complexity', 'maintainability', 'coverage', 'technical_debt']
        
        for metric_name in metric_names:
            # Simulate trend data
            timestamps = [datetime.utcnow() - timedelta(days=i) for i in range(30, 0, -1)]
            values = [50 + i * 0.5 for i in range(30)]  # Trending upward
            
            trend = MetricTrend(
                metric_name=metric_name,
                timestamps=timestamps,
                values=values,
                trend_direction='up' if values[-1] > values[0] else 'down',
                trend_strength=0.7,
                change_rate=0.1,
                prediction_next_week=values[-1] + 1.0,
                prediction_confidence=0.8
            )
            
            trends.append(trend)
        
        return trends
    
    def _check_quality_gates(self) -> List[QualityGate]:
        """Check all quality gates"""
        results = []
        
        for gate in self.quality_gates:
            failed_conditions = []
            passed_conditions = 0
            
            for condition in gate['conditions']:
                metric_value = self._get_metric_value(condition['metric'])
                threshold = condition['threshold']
                operator = condition['operator']
                
                passed = False
                if operator == 'greater_than' and metric_value > threshold:
                    passed = True
                elif operator == 'less_than' and metric_value < threshold:
                    passed = True
                elif operator == 'equals' and abs(metric_value - threshold) < 0.01:
                    passed = True
                
                if passed:
                    passed_conditions += 1
                else:
                    failed_conditions.append(f"{condition['metric']} {operator} {threshold}")
            
            # Determine status
            total_conditions = len(gate['conditions'])
            pass_rate = passed_conditions / total_conditions if total_conditions > 0 else 0
            
            if pass_rate >= gate['threshold']:
                status = 'passed'
            elif pass_rate >= 0.5:
                status = 'warning'
            else:
                status = 'failed'
            
            quality_gate = QualityGate(
                name=gate['name'],
                conditions=gate['conditions'],
                status=status,
                failed_conditions=failed_conditions,
                score=pass_rate * 100,
                threshold=gate['threshold'] * 100
            )
            
            results.append(quality_gate)
        
        return results
    
    def _get_metric_value(self, metric_name: str) -> float:
        """Get current value of a metric"""
        if not self.project_metrics:
            return 0.0
        
        metric_map = {
            'complexity': self.project_metrics.overall_complexity,
            'maintainability': self.project_metrics.overall_maintainability,
            'coverage': self.project_metrics.overall_coverage,
            'duplication': self.project_metrics.code_duplication,
            'technical_debt': self.project_metrics.technical_debt_hours
        }
        
        return metric_map.get(metric_name, 0.0)
    
    def _generate_alerts(self) -> List[Dict[str, Any]]:
        """Generate alerts for critical issues"""
        alerts = []
        
        if not self.project_metrics:
            return alerts
        
        # Critical complexity alert
        if self.project_metrics.overall_complexity > self.complexity_threshold * 1.5:
            alerts.append({
                'type': 'critical',
                'category': 'complexity',
                'message': f"Average complexity is critically high: {self.project_metrics.overall_complexity:.1f}",
                'recommendation': "Refactor complex functions immediately",
                'affected_files': len([f for f in self.file_metrics.values() if f.cyclomatic_complexity > 15])
            })
        
        # Low coverage alert
        if self.project_metrics.overall_coverage < self.coverage_threshold / 2:
            alerts.append({
                'type': 'warning',
                'category': 'coverage',
                'message': f"Test coverage is very low: {self.project_metrics.overall_coverage:.1f}%",
                'recommendation': "Add unit tests for critical functions",
                'affected_files': len([f for f in self.file_metrics.values() if f.test_coverage < 25])
            })
        
        # High technical debt alert
        if self.project_metrics.technical_debt_hours > 100:
            alerts.append({
                'type': 'warning',
                'category': 'technical_debt',
                'message': f"Technical debt is high: {self.project_metrics.technical_debt_hours:.1f} hours",
                'recommendation': "Schedule technical debt reduction sprints",
                'affected_files': len([f for f in self.file_metrics.values() if f.technical_debt_ratio > 0.3])
            })
        
        # Security hotspots alert
        security_files = [f for f in self.file_metrics.values() if f.security_hotspots > 0]
        if len(security_files) > 0:
            total_hotspots = sum(f.security_hotspots for f in security_files)
            alerts.append({
                'type': 'warning',
                'category': 'security',
                'message': f"Found {total_hotspots} security hotspots in {len(security_files)} files",
                'recommendation': "Review and fix security vulnerabilities",
                'affected_files': len(security_files)
            })
        
        return alerts
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        if not self.project_metrics:
            return recommendations
        
        # Complexity recommendations
        if self.project_metrics.overall_complexity > self.complexity_threshold:
            high_complexity_files = [f for f in self.file_metrics.values() if f.cyclomatic_complexity > 15]
            recommendations.append(f"Refactor {len(high_complexity_files)} files with high complexity")
        
        # Coverage recommendations
        if self.project_metrics.overall_coverage < self.coverage_threshold:
            uncovered_files = [f for f in self.file_metrics.values() if f.test_coverage < 50]
            recommendations.append(f"Add tests for {len(uncovered_files)} files with low coverage")
        
        # Maintainability recommendations
        if self.project_metrics.overall_maintainability < 50:
            recommendations.append("Improve code maintainability by reducing complexity and technical debt")
        
        # Duplication recommendations
        if self.project_metrics.code_duplication > self.duplication_threshold:
            recommendations.append("Extract common code into reusable functions to reduce duplication")
        
        # Technical debt recommendations
        if self.project_metrics.technical_debt_hours > 50:
            recommendations.append("Address technical debt to improve long-term maintainability")
        
        return recommendations
    
    async def _generate_chart_data(self) -> Dict[str, Any]:
        """Generate data for dashboard charts"""
        if not self.project_metrics:
            return {}
        
        # Complexity distribution
        complexity_distribution = Counter()
        for f in self.file_metrics.values():
            if f.cyclomatic_complexity <= 5:
                complexity_distribution['Low (1-5)'] += 1
            elif f.cyclomatic_complexity <= 10:
                complexity_distribution['Medium (6-10)'] += 1
            elif f.cyclomatic_complexity <= 20:
                complexity_distribution['High (11-20)'] += 1
            else:
                complexity_distribution['Very High (>20)'] += 1
        
        # Coverage distribution
        coverage_distribution = Counter()
        for f in self.file_metrics.values():
            if f.test_coverage >= 80:
                coverage_distribution['Excellent (80%+)'] += 1
            elif f.test_coverage >= 60:
                coverage_distribution['Good (60-79%)'] += 1
            elif f.test_coverage >= 40:
                coverage_distribution['Fair (40-59%)'] += 1
            elif f.test_coverage > 0:
                coverage_distribution['Poor (1-39%)'] += 1
            else:
                coverage_distribution['No Tests'] += 1
        
        # File size distribution
        size_distribution = Counter()
        for f in self.file_metrics.values():
            if f.lines_of_code <= 100:
                size_distribution['Small (≤100)'] += 1
            elif f.lines_of_code <= 300:
                size_distribution['Medium (101-300)'] += 1
            elif f.lines_of_code <= 500:
                size_distribution['Large (301-500)'] += 1
            else:
                size_distribution['Very Large (>500)'] += 1
        
        return {
            'complexity_distribution': dict(complexity_distribution),
            'coverage_distribution': dict(coverage_distribution),
            'size_distribution': dict(size_distribution),
            'module_metrics': {
                module.module_name: {
                    'complexity': module.average_complexity,
                    'maintainability': 100 - module.technical_debt_hours,  # Inverse relationship
                    'coverage': module.test_coverage,
                    'files': module.file_count
                }
                for module in self.module_metrics.values()
            }
        }
    
    def _prepare_export_data(self) -> Dict[str, Any]:
        """Prepare data for export"""
        return {
            'project_summary': self._project_metrics_to_dict() if self.project_metrics else {},
            'file_details': [self._file_metrics_to_dict(f) for f in self.file_metrics.values()],
            'module_summaries': [self._module_metrics_to_dict(m) for m in self.module_metrics.values()],
            'export_timestamp': datetime.utcnow().isoformat()
        }
    
    async def _save_historical_data(self) -> None:
        """Save metrics to historical data store"""
        if not self.project_metrics:
            return
        
        timestamp = datetime.utcnow()
        
        # Save key metrics to history
        self.metric_history['complexity'].append((timestamp, self.project_metrics.overall_complexity))
        self.metric_history['maintainability'].append((timestamp, self.project_metrics.overall_maintainability))
        self.metric_history['coverage'].append((timestamp, self.project_metrics.overall_coverage))
        self.metric_history['technical_debt'].append((timestamp, self.project_metrics.technical_debt_hours))
        
        # Keep only last 100 data points
        for metric_name in self.metric_history:
            if len(self.metric_history[metric_name]) > 100:
                self.metric_history[metric_name] = self.metric_history[metric_name][-100:]
    
    def _create_summary_section(self) -> Dict[str, Any]:
        """Create dashboard summary section"""
        if not self.project_metrics:
            return {}
        
        return {
            'total_files': self.project_metrics.total_files,
            'total_lines': self.project_metrics.total_loc,
            'quality_rating': self._get_overall_quality_rating(),
            'quality_gate_status': self.project_metrics.quality_gate_status,
            'technical_debt_days': round(self.project_metrics.technical_debt_hours / 8, 1),
            'test_coverage_percentage': round(self.project_metrics.overall_coverage, 1),
            'complexity_score': round(self.project_metrics.overall_complexity, 1),
            'maintainability_score': round(self.project_metrics.overall_maintainability, 1)
        }
    
    def _get_overall_quality_rating(self) -> str:
        """Calculate overall quality rating"""
        if not self.project_metrics:
            return 'N/A'
        
        ratings = [
            self.project_metrics.security_rating,
            self.project_metrics.reliability_rating,
            self.project_metrics.maintainability_rating
        ]
        
        # Convert ratings to numeric values
        rating_values = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
        avg_rating = statistics.mean([rating_values[r] for r in ratings])
        
        # Convert back to letter grade
        if avg_rating >= 4.5:
            return 'A'
        elif avg_rating >= 3.5:
            return 'B'
        elif avg_rating >= 2.5:
            return 'C'
        elif avg_rating >= 1.5:
            return 'D'
        else:
            return 'E'
    
    def _load_quality_gates(self) -> List[Dict[str, Any]]:
        """Load quality gate configurations"""
        return [
            {
                'name': 'Complexity Gate',
                'conditions': [
                    {'metric': 'complexity', 'operator': 'less_than', 'threshold': self.complexity_threshold}
                ],
                'threshold': 1.0
            },
            {
                'name': 'Coverage Gate',
                'conditions': [
                    {'metric': 'coverage', 'operator': 'greater_than', 'threshold': self.coverage_threshold}
                ],
                'threshold': 1.0
            },
            {
                'name': 'Quality Gate',
                'conditions': [
                    {'metric': 'complexity', 'operator': 'less_than', 'threshold': self.complexity_threshold},
                    {'metric': 'coverage', 'operator': 'greater_than', 'threshold': self.coverage_threshold},
                    {'metric': 'duplication', 'operator': 'less_than', 'threshold': self.duplication_threshold}
                ],
                'threshold': 0.8  # 80% of conditions must pass
            }
        ]
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = ['__pycache__', '.git', '.venv', 'node_modules']
        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)
    
    # Conversion methods
    def _project_metrics_to_dict(self) -> Dict[str, Any]:
        """Convert project metrics to dictionary"""
        if not self.project_metrics:
            return {}
        
        return {
            'total_files': self.project_metrics.total_files,
            'total_loc': self.project_metrics.total_loc,
            'total_test_files': self.project_metrics.total_test_files,
            'total_test_loc': self.project_metrics.total_test_loc,
            'overall_complexity': round(self.project_metrics.overall_complexity, 2),
            'overall_maintainability': round(self.project_metrics.overall_maintainability, 2),
            'overall_coverage': round(self.project_metrics.overall_coverage, 2),
            'technical_debt_hours': round(self.project_metrics.technical_debt_hours, 2),
            'code_duplication': round(self.project_metrics.code_duplication, 2),
            'security_rating': self.project_metrics.security_rating,
            'reliability_rating': self.project_metrics.reliability_rating,
            'maintainability_rating': self.project_metrics.maintainability_rating,
            'quality_gate_status': self.project_metrics.quality_gate_status
        }
    
    def _module_metrics_to_dict(self, module: ModuleMetrics) -> Dict[str, Any]:
        """Convert module metrics to dictionary"""
        return {
            'module_name': module.module_name,
            'file_count': module.file_count,
            'total_loc': module.total_loc,
            'average_complexity': round(module.average_complexity, 2),
            'coupling_index': round(module.coupling_index, 2),
            'cohesion_index': round(module.cohesion_index, 2),
            'stability_index': round(module.stability_index, 2),
            'abstractness': round(module.abstractness, 2),
            'distance_from_main': round(module.distance_from_main, 2),
            'test_coverage': round(module.test_coverage, 2),
            'technical_debt_hours': round(module.technical_debt_hours, 2),
            'quality_gate_status': module.quality_gate_status
        }
    
    def _file_metrics_to_dict(self, file_metrics: FileMetrics) -> Dict[str, Any]:
        """Convert file metrics to dictionary"""
        return {
            'file_path': file_metrics.file_path,
            'lines_of_code': file_metrics.lines_of_code,
            'cyclomatic_complexity': file_metrics.cyclomatic_complexity,
            'cognitive_complexity': file_metrics.cognitive_complexity,
            'halstead_complexity': file_metrics.halstead_complexity,
            'maintainability_index': round(file_metrics.maintainability_index, 2),
            'technical_debt_ratio': round(file_metrics.technical_debt_ratio, 2),
            'test_coverage': round(file_metrics.test_coverage, 2),
            'duplicate_lines': file_metrics.duplicate_lines,
            'code_smells': file_metrics.code_smells[:5],  # Limit for display
            'security_hotspots': file_metrics.security_hotspots,
            'performance_issues': file_metrics.performance_issues,
            'last_modified': file_metrics.last_modified.isoformat(),
            'author_count': file_metrics.author_count,
            'change_frequency': file_metrics.change_frequency
        }
    
    def _trend_to_dict(self, trend: MetricTrend) -> Dict[str, Any]:
        """Convert trend to dictionary"""
        return {
            'metric_name': trend.metric_name,
            'timestamps': [ts.isoformat() for ts in trend.timestamps],
            'values': trend.values,
            'trend_direction': trend.trend_direction,
            'trend_strength': round(trend.trend_strength, 2),
            'change_rate': round(trend.change_rate, 4),
            'prediction_next_week': round(trend.prediction_next_week, 2),
            'prediction_confidence': round(trend.prediction_confidence, 2)
        }
    
    def _quality_gate_to_dict(self, gate: QualityGate) -> Dict[str, Any]:
        """Convert quality gate to dictionary"""
        return {
            'name': gate.name,
            'conditions': gate.conditions,
            'status': gate.status,
            'failed_conditions': gate.failed_conditions,
            'score': round(gate.score, 1),
            'threshold': round(gate.threshold, 1)
        }
    
    async def export_dashboard(self, format_type: str = 'json') -> str:
        """
        Export dashboard data in specified format
        
        Args:
            format_type: Export format ('json', 'csv', 'html')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == 'json':
            filename = f"code_metrics_dashboard_{timestamp}.json"
            data = self._prepare_export_data()
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format_type == 'html':
            filename = f"code_metrics_dashboard_{timestamp}.html"
            await self._generate_html_report(filename)
            
        else:
            return f"Unsupported export format: {format_type}"
        
        return filename
    
    async def _generate_html_report(self, filename: str) -> None:
        """Generate HTML dashboard report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Code Metrics Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-card {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .status-passed {{ color: green; }}
                .status-warning {{ color: orange; }}
                .status-failed {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Code Metrics Dashboard</h1>
            <p>Generated: {datetime.utcnow().isoformat()}</p>
            
            <div class="metric-card">
                <h2>Project Summary</h2>
                <p>Total Files: {self.project_metrics.total_files if self.project_metrics else 0}</p>
                <p>Total Lines: {self.project_metrics.total_loc if self.project_metrics else 0}</p>
                <p class="status-{self.project_metrics.quality_gate_status if self.project_metrics else 'unknown'}">
                    Quality Gate: {self.project_metrics.quality_gate_status.title() if self.project_metrics else 'Unknown'}
                </p>
            </div>
            
            <h2>File Metrics</h2>
            <table>
                <tr>
                    <th>File</th>
                    <th>LOC</th>
                    <th>Complexity</th>
                    <th>Maintainability</th>
                    <th>Coverage</th>
                </tr>
        """
        
        for file_metrics in sorted(self.file_metrics.values(), key=lambda x: x.cyclomatic_complexity, reverse=True)[:20]:
            html_content += f"""
                <tr>
                    <td>{Path(file_metrics.file_path).name}</td>
                    <td>{file_metrics.lines_of_code}</td>
                    <td>{file_metrics.cyclomatic_complexity}</td>
                    <td>{file_metrics.maintainability_index:.1f}</td>
                    <td>{file_metrics.test_coverage:.1f}%</td>
                </tr>
            """
        
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html_content)