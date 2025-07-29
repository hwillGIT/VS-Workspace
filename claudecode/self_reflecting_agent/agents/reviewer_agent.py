"""
Reviewer Agent implementation for code quality assessment and review.

The Reviewer Agent is responsible for:
- Conducting comprehensive code reviews for quality, security, and best practices
- Analyzing code complexity and maintainability metrics
- Identifying potential security vulnerabilities and anti-patterns
- Ensuring adherence to coding standards and conventions
- Providing actionable feedback and improvement suggestions
"""

import ast
import re
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

import dspy
from pydantic import BaseModel, Field

from .base_agent import BaseAgent


class ReviewSeverity(Enum):
    """Severity levels for review findings."""
    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class ReviewCategory(Enum):
    """Categories of review findings."""
    CODE_QUALITY = "code_quality"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    BEST_PRACTICES = "best_practices"


@dataclass
class ReviewFinding:
    """Represents a single review finding."""
    category: ReviewCategory
    severity: ReviewSeverity
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    suggestion: Optional[str] = None


class CodeReview(dspy.Signature):
    """DSPy signature for comprehensive code review."""
    
    code_content = dspy.InputField(desc="Code to be reviewed")
    file_path = dspy.InputField(desc="File path and context")
    review_criteria = dspy.InputField(desc="Specific criteria and standards to apply")
    
    overall_assessment = dspy.OutputField(desc="Overall code quality assessment and score (1-10)")
    quality_findings = dspy.OutputField(desc="JSON list of code quality findings with category, severity, title, description, line_number, suggestion")
    security_findings = dspy.OutputField(desc="JSON list of security-related findings")
    performance_findings = dspy.OutputField(desc="JSON list of performance-related findings")
    recommendations = dspy.OutputField(desc="High-level recommendations for improvement")


class SecurityAnalysis(dspy.Signature):
    """DSPy signature for focused security analysis."""
    
    code_content = dspy.InputField(desc="Code to analyze for security issues")
    language = dspy.InputField(desc="Programming language")
    context = dspy.InputField(desc="Application context and security requirements")
    
    vulnerability_assessment = dspy.OutputField(desc="Assessment of potential security vulnerabilities")
    security_score = dspy.OutputField(desc="Security score (1-10) with justification")
    critical_issues = dspy.OutputField(desc="JSON list of critical security issues")
    recommendations = dspy.OutputField(desc="Security improvement recommendations")


class ArchitectureReview(dspy.Signature):
    """DSPy signature for architectural and design review."""
    
    code_structure = dspy.InputField(desc="Code structure and architecture")
    design_patterns = dspy.InputField(desc="Design patterns and architectural decisions")
    requirements = dspy.InputField(desc="Functional and non-functional requirements")
    
    architecture_assessment = dspy.OutputField(desc="Assessment of architectural quality and design decisions")
    design_patterns_analysis = dspy.OutputField(desc="Analysis of design pattern usage and appropriateness")
    scalability_assessment = dspy.OutputField(desc="Assessment of scalability and maintainability")
    improvement_suggestions = dspy.OutputField(desc="Suggestions for architectural improvements")


class ReviewerAgent(BaseAgent):
    """
    Reviewer Agent responsible for comprehensive code quality assessment.
    
    This agent conducts thorough code reviews focusing on quality, security,
    performance, maintainability, and adherence to best practices.
    """
    
    def __init__(self, agent_id: str = "reviewer", **kwargs):
        super().__init__(agent_id, **kwargs)
        
        # DSPy modules for review capabilities
        if self.dspy_enabled:
            self.code_reviewer = dspy.TypedChainOfThought(CodeReview)
            self.security_analyzer = dspy.TypedChainOfThought(SecurityAnalysis)
            self.architecture_reviewer = dspy.TypedChainOfThought(ArchitectureReview)
        
        # Review configuration
        self.review_standards = self.config.get("review_standards", {})
        self.security_rules = self.config.get("security_rules", [])
        self.complexity_threshold = self.config.get("complexity_threshold", 10)
        self.max_function_length = self.config.get("max_function_length", 50)
        self.max_file_length = self.config.get("max_file_length", 500)
        
        # Quality metrics thresholds
        self.quality_thresholds = {
            "min_score": self.config.get("min_quality_score", 7.0),
            "max_complexity": self.config.get("max_complexity", 10),
            "max_nesting": self.config.get("max_nesting", 4),
            "min_test_coverage": self.config.get("min_test_coverage", 0.8)
        }
        
        # Common anti-patterns and code smells
        self.anti_patterns = [
            "god_class", "long_method", "large_class", "duplicate_code",
            "dead_code", "magic_numbers", "long_parameter_list"
        ]
        
        # Security vulnerability patterns
        self.security_patterns = {
            "sql_injection": [r"SELECT.*\+.*", r"INSERT.*\+.*", r"UPDATE.*\+.*"],
            "xss": [r"innerHTML.*\+", r"document\.write.*\+"],
            "path_traversal": [r"\.\.\/", r"\.\.\\\\"],
            "hardcoded_secrets": [r"password\s*=\s*['\"]", r"api_key\s*=\s*['\"]", r"secret\s*=\s*['\"]"]
        }
        
        self.logger.info("Reviewer Agent initialized and ready for code review")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Reviewer Agent."""
        return """You are the Reviewer Agent in a Self-Reflecting Claude Code Agent system.

Your primary responsibilities are:
1. Conducting comprehensive code reviews for quality, security, and maintainability
2. Identifying potential bugs, security vulnerabilities, and performance issues
3. Ensuring adherence to coding standards and best practices
4. Analyzing code complexity and suggesting improvements
5. Reviewing architectural decisions and design patterns
6. Providing constructive feedback and actionable recommendations

When reviewing code, focus on:
- Code quality and readability
- Security vulnerabilities and best practices
- Performance implications and optimizations
- Maintainability and technical debt
- Test coverage and testing strategies
- Documentation quality
- Adherence to SOLID principles and design patterns
- Error handling and edge cases

Provide specific, actionable feedback with clear explanations of why issues matter
and how they can be addressed. Rate code quality on a scale of 1-10 and provide
detailed justification for your assessment."""
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a code review task.
        
        Args:
            task: Task specification containing code, files, or review requirements
            
        Returns:
            Result dictionary with review findings, scores, and recommendations
        """
        start_time = datetime.now()
        
        try:
            review_type = task.get("type", "code_review")
            self.logger.info(f"Processing {review_type} task: {task.get('title', 'Code Review')}")
            
            # Update state
            self.state.current_task = task.get('title', 'Code Review')
            
            # Route to appropriate review handler
            if review_type == "security_review":
                result = await self._handle_security_review(task)
            elif review_type == "architecture_review":
                result = await self._handle_architecture_review(task)
            elif review_type == "performance_review":
                result = await self._handle_performance_review(task)
            else:
                result = await self._handle_comprehensive_review(task)
            
            # Update metrics
            response_time = (datetime.now() - start_time).total_seconds()
            success = result.get("status") == "completed"
            self.update_metrics(response_time, success)
            
            # Store in memory
            await self.update_memory(
                f"Completed {review_type}: {task.get('title', 'Review')} - Score: {result.get('overall_score', 'N/A')}",
                {"task_type": review_type, "result": result}
            )
            
            result["response_time"] = response_time
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing review task: {str(e)}")
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(response_time, False)
            
            return {
                "status": "failed",
                "error": str(e),
                "response_time": response_time
            }
    
    async def _handle_comprehensive_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle comprehensive code review including all aspects."""
        
        code_content = task.get("code", "")
        file_path = task.get("file_path", "unknown")
        files = task.get("files", [])
        
        all_findings = []
        file_scores = []
        
        # Review individual files or provided code
        if files:
            for file_info in files:
                file_result = await self._review_single_file(
                    file_info.get("content", ""),
                    file_info.get("path", ""),
                    file_info.get("language", "python")
                )
                all_findings.extend(file_result["findings"])
                file_scores.append(file_result["score"])
        elif code_content:
            file_result = await self._review_single_file(code_content, file_path)
            all_findings.extend(file_result["findings"])
            file_scores.append(file_result["score"])
        
        # Calculate overall metrics
        overall_score = sum(file_scores) / len(file_scores) if file_scores else 0
        
        # Categorize findings
        findings_by_category = {}
        findings_by_severity = {}
        
        for finding in all_findings:
            category = finding.category.value
            severity = finding.severity.value
            
            if category not in findings_by_category:
                findings_by_category[category] = []
            findings_by_category[category].append(finding)
            
            if severity not in findings_by_severity:
                findings_by_severity[severity] = []
            findings_by_severity[severity].append(finding)
        
        # Generate summary recommendations
        recommendations = await self._generate_summary_recommendations(all_findings, overall_score)
        
        return {
            "status": "completed",
            "overall_score": round(overall_score, 2),
            "total_findings": len(all_findings),
            "findings_by_category": {k: len(v) for k, v in findings_by_category.items()},
            "findings_by_severity": {k: len(v) for k, v in findings_by_severity.items()},
            "detailed_findings": [self._finding_to_dict(f) for f in all_findings],
            "recommendations": recommendations,
            "quality_gate_passed": overall_score >= self.quality_thresholds["min_score"],
            "critical_issues": len(findings_by_severity.get("critical", [])),
            "review_summary": await self._generate_review_summary(all_findings, overall_score)
        }
    
    async def _review_single_file(
        self, 
        code_content: str, 
        file_path: str, 
        language: str = "python"
    ) -> Dict[str, Any]:
        """Review a single file and return findings and score."""
        
        findings = []
        
        # Static analysis checks
        static_findings = await self._perform_static_analysis(code_content, file_path, language)
        findings.extend(static_findings)
        
        # Security analysis
        security_findings = await self._perform_security_analysis(code_content, language)
        findings.extend(security_findings)
        
        # Code quality analysis
        quality_findings = await self._perform_quality_analysis(code_content, file_path)
        findings.extend(quality_findings)
        
        # DSPy-powered comprehensive review if available
        if self.dspy_enabled:
            dspy_findings = await self._perform_dspy_review(code_content, file_path)
            findings.extend(dspy_findings)
        
        # Calculate file score
        score = await self._calculate_file_score(findings, code_content)
        
        return {
            "file_path": file_path,
            "score": score,
            "findings": findings,
            "language": language
        }
    
    async def _perform_static_analysis(
        self, 
        code_content: str, 
        file_path: str, 
        language: str
    ) -> List[ReviewFinding]:
        """Perform static analysis checks on the code."""
        
        findings = []
        
        if language.lower() == "python":
            # Python-specific analysis
            try:
                tree = ast.parse(code_content)
                
                # Check for complexity issues
                complexity_findings = await self._check_complexity(tree, file_path)
                findings.extend(complexity_findings)
                
                # Check for code structure issues
                structure_findings = await self._check_code_structure(tree, file_path)
                findings.extend(structure_findings)
                
                # Check for naming conventions
                naming_findings = await self._check_naming_conventions(tree, file_path)
                findings.extend(naming_findings)
                
            except SyntaxError as e:
                findings.append(ReviewFinding(
                    category=ReviewCategory.CODE_QUALITY,
                    severity=ReviewSeverity.CRITICAL,
                    title="Syntax Error",
                    description=f"Code contains syntax error: {str(e)}",
                    file_path=file_path,
                    line_number=getattr(e, 'lineno', None)
                ))
        
        # Language-agnostic checks
        general_findings = await self._check_general_patterns(code_content, file_path)
        findings.extend(general_findings)
        
        return findings
    
    async def _perform_security_analysis(self, code_content: str, language: str) -> List[ReviewFinding]:
        """Perform security analysis on the code."""
        
        findings = []
        
        # Check for common security patterns
        for vulnerability_type, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, code_content, re.IGNORECASE)
                for match in matches:
                    line_number = code_content[:match.start()].count('\n') + 1
                    
                    findings.append(ReviewFinding(
                        category=ReviewCategory.SECURITY,
                        severity=ReviewSeverity.MAJOR,
                        title=f"Potential {vulnerability_type.replace('_', ' ').title()}",
                        description=f"Code pattern suggests potential {vulnerability_type} vulnerability",
                        line_number=line_number,
                        code_snippet=match.group(),
                        suggestion=f"Review and validate input handling for {vulnerability_type} prevention"
                    ))
        
        # Check for hardcoded credentials
        credential_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]
        
        for pattern in credential_patterns:
            matches = re.finditer(pattern, code_content, re.IGNORECASE)
            for match in matches:
                line_number = code_content[:match.start()].count('\n') + 1
                
                findings.append(ReviewFinding(
                    category=ReviewCategory.SECURITY,
                    severity=ReviewSeverity.CRITICAL,
                    title="Hardcoded Credentials",
                    description="Potential hardcoded credentials found in source code",
                    line_number=line_number,
                    code_snippet=match.group(),
                    suggestion="Move credentials to environment variables or secure configuration"
                ))
        
        return findings
    
    async def _perform_quality_analysis(self, code_content: str, file_path: str) -> List[ReviewFinding]:
        """Perform code quality analysis."""
        
        findings = []
        lines = code_content.split('\n')
        
        # Check file length
        if len(lines) > self.max_file_length:
            findings.append(ReviewFinding(
                category=ReviewCategory.MAINTAINABILITY,
                severity=ReviewSeverity.MAJOR,
                title="File Too Long",
                description=f"File has {len(lines)} lines, exceeds maximum of {self.max_file_length}",
                file_path=file_path,
                suggestion="Consider breaking this file into smaller, more focused modules"
            ))
        
        # Check for long lines
        for i, line in enumerate(lines, 1):
            if len(line) > 100:  # Standard line length limit
                findings.append(ReviewFinding(
                    category=ReviewCategory.CODE_QUALITY,
                    severity=ReviewSeverity.MINOR,
                    title="Line Too Long",
                    description=f"Line {i} has {len(line)} characters, exceeds 100 character limit",
                    file_path=file_path,
                    line_number=i,
                    suggestion="Break long lines for better readability"
                ))
        
        # Check for TODO/FIXME comments
        todo_pattern = r'#\s*(TODO|FIXME|HACK|XXX):?\s*(.+)'
        for match in re.finditer(todo_pattern, code_content, re.IGNORECASE):
            line_number = code_content[:match.start()].count('\n') + 1
            
            findings.append(ReviewFinding(
                category=ReviewCategory.MAINTAINABILITY,
                severity=ReviewSeverity.INFO,
                title="Technical Debt Marker",
                description=f"Found {match.group(1).upper()}: {match.group(2)}",
                file_path=file_path,
                line_number=line_number,
                suggestion="Address technical debt items before production deployment"
            ))
        
        return findings
    
    async def _check_complexity(self, tree: ast.AST, file_path: str) -> List[ReviewFinding]:
        """Check for complexity issues in Python AST."""
        
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Calculate cyclomatic complexity
                complexity = self._calculate_cyclomatic_complexity(node)
                
                if complexity > self.complexity_threshold:
                    findings.append(ReviewFinding(
                        category=ReviewCategory.MAINTAINABILITY,
                        severity=ReviewSeverity.MAJOR,
                        title="High Cyclomatic Complexity",
                        description=f"Function '{node.name}' has complexity of {complexity}, exceeds threshold of {self.complexity_threshold}",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Consider breaking this function into smaller, more focused functions"
                    ))
                
                # Check function length
                func_length = getattr(node, 'end_lineno', node.lineno) - node.lineno + 1
                if func_length > self.max_function_length:
                    findings.append(ReviewFinding(
                        category=ReviewCategory.MAINTAINABILITY,
                        severity=ReviewSeverity.MAJOR,
                        title="Function Too Long",
                        description=f"Function '{node.name}' has {func_length} lines, exceeds maximum of {self.max_function_length}",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Break this function into smaller, more focused functions"
                    ))
        
        return findings
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                                ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    async def _check_code_structure(self, tree: ast.AST, file_path: str) -> List[ReviewFinding]:
        """Check code structure and organization."""
        
        findings = []
        
        # Check for classes with too many methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 20:  # Arbitrary threshold for large classes
                    findings.append(ReviewFinding(
                        category=ReviewCategory.MAINTAINABILITY,
                        severity=ReviewSeverity.MAJOR,
                        title="Large Class",
                        description=f"Class '{node.name}' has {len(methods)} methods, consider splitting responsibilities",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Consider applying Single Responsibility Principle and splitting the class"
                    ))
        
        return findings
    
    async def _check_naming_conventions(self, tree: ast.AST, file_path: str) -> List[ReviewFinding]:
        """Check naming conventions."""
        
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function naming (should be snake_case)
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name) and not node.name.startswith('__'):
                    findings.append(ReviewFinding(
                        category=ReviewCategory.CODE_QUALITY,
                        severity=ReviewSeverity.MINOR,
                        title="Function Naming Convention",
                        description=f"Function '{node.name}' doesn't follow snake_case convention",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Use snake_case for function names"
                    ))
            
            elif isinstance(node, ast.ClassDef):
                # Check class naming (should be PascalCase)
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    findings.append(ReviewFinding(
                        category=ReviewCategory.CODE_QUALITY,
                        severity=ReviewSeverity.MINOR,
                        title="Class Naming Convention",
                        description=f"Class '{node.name}' doesn't follow PascalCase convention",
                        file_path=file_path,
                        line_number=node.lineno,
                        suggestion="Use PascalCase for class names"
                    ))
        
        return findings
    
    async def _check_general_patterns(self, code_content: str, file_path: str) -> List[ReviewFinding]:
        """Check for general code patterns and anti-patterns."""
        
        findings = []
        
        # Check for magic numbers
        magic_number_pattern = r'\b(?<![\.\w])\d{2,}\b(?!\s*[:\.])'
        for match in re.finditer(magic_number_pattern, code_content):
            line_number = code_content[:match.start()].count('\n') + 1
            
            findings.append(ReviewFinding(
                category=ReviewCategory.MAINTAINABILITY,
                severity=ReviewSeverity.MINOR,
                title="Magic Number",
                description=f"Magic number {match.group()} found at line {line_number}",
                file_path=file_path,
                line_number=line_number,
                code_snippet=match.group(),
                suggestion="Consider defining this as a named constant"
            ))
        
        # Check for commented out code
        commented_code_pattern = r'^\s*#.*[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*.*$'
        for match in re.finditer(commented_code_pattern, code_content, re.MULTILINE):
            line_number = code_content[:match.start()].count('\n') + 1
            
            findings.append(ReviewFinding(
                category=ReviewCategory.MAINTAINABILITY,
                severity=ReviewSeverity.MINOR,
                title="Commented Out Code",
                description="Potentially commented out code found",
                file_path=file_path,
                line_number=line_number,
                suggestion="Remove commented code or add explanation if needed"
            ))
        
        return findings
    
    async def _perform_dspy_review(self, code_content: str, file_path: str) -> List[ReviewFinding]:
        """Perform DSPy-powered comprehensive review."""
        
        findings = []
        
        try:
            review_criteria = self._get_review_criteria()
            
            result = self.code_reviewer(
                code_content=code_content,
                file_path=file_path,
                review_criteria=review_criteria
            )
            
            # Parse findings from DSPy output
            import json
            
            # Parse quality findings
            try:
                quality_findings = json.loads(result.quality_findings)
                for finding_data in quality_findings:
                    findings.append(ReviewFinding(
                        category=ReviewCategory(finding_data.get("category", "code_quality")),
                        severity=ReviewSeverity(finding_data.get("severity", "minor")),
                        title=finding_data.get("title", ""),
                        description=finding_data.get("description", ""),
                        file_path=file_path,
                        line_number=finding_data.get("line_number"),
                        suggestion=finding_data.get("suggestion")
                    ))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass  # Skip if parsing fails
            
            # Parse security findings
            try:
                security_findings = json.loads(result.security_findings)
                for finding_data in security_findings:
                    findings.append(ReviewFinding(
                        category=ReviewCategory.SECURITY,
                        severity=ReviewSeverity(finding_data.get("severity", "major")),
                        title=finding_data.get("title", ""),
                        description=finding_data.get("description", ""),
                        file_path=file_path,
                        line_number=finding_data.get("line_number"),
                        suggestion=finding_data.get("suggestion")
                    ))
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
                
        except Exception as e:
            self.logger.warning(f"DSPy review failed: {e}")
        
        return findings
    
    def _get_review_criteria(self) -> str:
        """Get review criteria and standards."""
        
        criteria = [
            "Code quality and readability",
            "Security best practices",
            "Performance considerations", 
            "Error handling and edge cases",
            "Testing and testability",
            "Documentation and comments",
            "SOLID principles adherence",
            "Design patterns usage",
            "Code organization and structure"
        ]
        
        return "\n".join([f"- {criterion}" for criterion in criteria])
    
    async def _calculate_file_score(self, findings: List[ReviewFinding], code_content: str) -> float:
        """Calculate overall quality score for a file."""
        
        base_score = 10.0
        
        # Deduct points based on findings severity
        severity_deductions = {
            ReviewSeverity.INFO: 0.0,
            ReviewSeverity.MINOR: 0.1,
            ReviewSeverity.MAJOR: 0.5,
            ReviewSeverity.CRITICAL: 2.0
        }
        
        for finding in findings:
            base_score -= severity_deductions.get(finding.severity, 0.1)
        
        # Apply minimum score
        return max(base_score, 1.0)
    
    async def _generate_summary_recommendations(
        self, 
        findings: List[ReviewFinding], 
        overall_score: float
    ) -> List[str]:
        """Generate high-level recommendations based on findings."""
        
        recommendations = []
        
        # Category-based recommendations
        categories = {}
        for finding in findings:
            category = finding.category.value
            if category not in categories:
                categories[category] = []
            categories[category].append(finding)
        
        # Security recommendations
        if ReviewCategory.SECURITY.value in categories:
            security_count = len(categories[ReviewCategory.SECURITY.value])
            recommendations.append(
                f"Address {security_count} security findings to improve application security"
            )
        
        # Maintainability recommendations
        if ReviewCategory.MAINTAINABILITY.value in categories:
            maint_count = len(categories[ReviewCategory.MAINTAINABILITY.value])
            recommendations.append(
                f"Refactor code to address {maint_count} maintainability issues"
            )
        
        # Quality recommendations
        if overall_score < self.quality_thresholds["min_score"]:
            recommendations.append(
                f"Overall quality score ({overall_score:.1f}) is below threshold ({self.quality_thresholds['min_score']})"
            )
        
        # Critical issues
        critical_findings = [f for f in findings if f.severity == ReviewSeverity.CRITICAL]
        if critical_findings:
            recommendations.append(
                f"Immediately address {len(critical_findings)} critical issues before deployment"
            )
        
        return recommendations
    
    async def _generate_review_summary(
        self, 
        findings: List[ReviewFinding], 
        overall_score: float
    ) -> str:
        """Generate a comprehensive review summary."""
        
        summary_parts = []
        
        # Overall assessment
        if overall_score >= 8.0:
            assessment = "Excellent"
        elif overall_score >= 7.0:
            assessment = "Good"
        elif overall_score >= 5.0:
            assessment = "Fair"
        else:
            assessment = "Poor"
        
        summary_parts.append(f"Overall Assessment: {assessment} (Score: {overall_score:.1f}/10)")
        
        # Findings summary
        if findings:
            critical = len([f for f in findings if f.severity == ReviewSeverity.CRITICAL])
            major = len([f for f in findings if f.severity == ReviewSeverity.MAJOR])
            minor = len([f for f in findings if f.severity == ReviewSeverity.MINOR])
            
            summary_parts.append(
                f"Findings: {critical} critical, {major} major, {minor} minor issues"
            )
        else:
            summary_parts.append("No significant issues found")
        
        # Key areas for improvement
        categories = {}
        for finding in findings:
            category = finding.category.value
            categories[category] = categories.get(category, 0) + 1
        
        if categories:
            top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
            areas = ", ".join([f"{cat} ({count})" for cat, count in top_categories])
            summary_parts.append(f"Primary areas for improvement: {areas}")
        
        return ". ".join(summary_parts)
    
    def _finding_to_dict(self, finding: ReviewFinding) -> Dict[str, Any]:
        """Convert ReviewFinding to dictionary for serialization."""
        
        return {
            "category": finding.category.value,
            "severity": finding.severity.value,
            "title": finding.title,
            "description": finding.description,
            "file_path": finding.file_path,
            "line_number": finding.line_number,
            "code_snippet": finding.code_snippet,
            "suggestion": finding.suggestion
        }
    
    async def _handle_security_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle focused security review."""
        
        code_content = task.get("code", "")
        language = task.get("language", "python")
        context = task.get("context", "")
        
        security_findings = await self._perform_security_analysis(code_content, language)
        
        if self.dspy_enabled:
            try:
                result = self.security_analyzer(
                    code_content=code_content,
                    language=language,
                    context=context
                )
                
                security_score = float(re.search(r'(\d+(?:\.\d+)?)', result.security_score).group(1))
                
                return {
                    "status": "completed",
                    "security_score": security_score,
                    "vulnerability_assessment": result.vulnerability_assessment,
                    "security_findings": [self._finding_to_dict(f) for f in security_findings],
                    "recommendations": result.recommendations.split('\n') if result.recommendations else []
                }
            except Exception as e:
                self.logger.warning(f"DSPy security analysis failed: {e}")
        
        # Fallback security review
        security_score = 10.0 - (len(security_findings) * 0.5)
        
        return {
            "status": "completed",
            "security_score": max(security_score, 1.0),
            "security_findings": [self._finding_to_dict(f) for f in security_findings],
            "recommendations": ["Review and address identified security patterns"]
        }
    
    async def _handle_architecture_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle architectural review."""
        
        code_structure = task.get("code_structure", "")
        design_patterns = task.get("design_patterns", "")
        requirements = task.get("requirements", {})
        
        if self.dspy_enabled:
            try:
                result = self.architecture_reviewer(
                    code_structure=code_structure,
                    design_patterns=design_patterns,
                    requirements=str(requirements)
                )
                
                return {
                    "status": "completed",
                    "architecture_assessment": result.architecture_assessment,
                    "design_patterns_analysis": result.design_patterns_analysis,
                    "scalability_assessment": result.scalability_assessment,
                    "improvement_suggestions": result.improvement_suggestions.split('\n') if result.improvement_suggestions else []
                }
            except Exception as e:
                self.logger.warning(f"DSPy architecture review failed: {e}")
        
        # Fallback architecture review
        return {
            "status": "completed",
            "architecture_assessment": "Architecture review completed with basic analysis",
            "improvement_suggestions": ["Consider applying SOLID principles", "Review design patterns usage"]
        }
    
    async def _handle_performance_review(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance-focused review."""
        
        code_content = task.get("code", "")
        
        # Basic performance analysis
        performance_findings = []
        
        # Check for common performance issues
        performance_patterns = {
            "nested_loops": r'for\s+.*:\s*\n\s*for\s+.*:',
            "string_concatenation": r'\+\s*=\s*["\']',
            "repeated_db_calls": r'execute\s*\(',
        }
        
        for issue_type, pattern in performance_patterns.items():
            matches = re.finditer(pattern, code_content)
            for match in matches:
                line_number = code_content[:match.start()].count('\n') + 1
                
                performance_findings.append(ReviewFinding(
                    category=ReviewCategory.PERFORMANCE,
                    severity=ReviewSeverity.MAJOR,
                    title=f"Potential {issue_type.replace('_', ' ').title()} Issue",
                    description=f"Pattern suggests potential {issue_type} performance issue",
                    line_number=line_number,
                    code_snippet=match.group(),
                    suggestion=f"Review and optimize {issue_type} pattern"
                ))
        
        return {
            "status": "completed",
            "performance_findings": [self._finding_to_dict(f) for f in performance_findings],
            "performance_score": max(10.0 - len(performance_findings), 1.0),
            "recommendations": ["Profile code execution", "Consider algorithmic optimizations"]
        }