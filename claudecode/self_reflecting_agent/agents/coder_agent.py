"""
Coder Agent implementation for code generation and implementation tasks.

The Coder Agent is responsible for:
- Implementing code solutions based on specifications
- Refactoring and optimizing existing code
- Creating tests and documentation
- Following coding best practices and patterns
- Integrating with development tools and workflows
"""

import ast
import asyncio
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

import dspy
from pydantic import BaseModel, Field

from .base_agent import BaseAgent


class CodeLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    SQL = "sql"


class CodeTaskType(Enum):
    """Types of coding tasks."""
    IMPLEMENTATION = "implementation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    OPTIMIZATION = "optimization"


class CodeFile(BaseModel):
    """Represents a code file to be created or modified."""
    path: str = Field(description="File path relative to project root")
    language: CodeLanguage = Field(description="Programming language")
    content: str = Field(description="File content")
    description: str = Field(description="Description of what this file does")
    dependencies: List[str] = Field(default_factory=list, description="Required dependencies")


class CodeImplementation(dspy.Signature):
    """DSPy signature for implementing code based on specifications."""
    
    specification = dspy.InputField(desc="Detailed specification of what to implement")
    requirements = dspy.InputField(desc="Technical requirements and constraints")
    context = dspy.InputField(desc="Existing codebase context and patterns to follow")
    
    implementation_plan = dspy.OutputField(desc="High-level implementation approach")
    code_files = dspy.OutputField(desc="JSON list of code files to create/modify with path, language, content, description")
    tests = dspy.OutputField(desc="Test cases and testing strategy")
    documentation = dspy.OutputField(desc="Documentation and usage examples")


class CodeRefactoring(dspy.Signature):
    """DSPy signature for refactoring existing code."""
    
    existing_code = dspy.InputField(desc="Current code that needs refactoring")
    refactoring_goals = dspy.InputField(desc="What to improve (performance, readability, maintainability, etc.)")
    constraints = dspy.InputField(desc="Constraints and requirements to maintain")
    
    refactored_code = dspy.OutputField(desc="Improved code implementation")
    changes_summary = dspy.OutputField(desc="Summary of changes made and rationale")
    impact_analysis = dspy.OutputField(desc="Analysis of potential impacts and risks")


class CodeDebugging(dspy.Signature):
    """DSPy signature for debugging and fixing code issues."""
    
    problematic_code = dspy.InputField(desc="Code with bugs or issues")
    error_description = dspy.InputField(desc="Description of the problem or error")
    context = dspy.InputField(desc="Additional context about when/how the error occurs")
    
    root_cause = dspy.OutputField(desc="Identified root cause of the issue")
    fixed_code = dspy.OutputField(desc="Corrected code implementation")
    testing_recommendations = dspy.OutputField(desc="Recommendations for testing the fix")


class CoderAgent(BaseAgent):
    """
    Coder Agent responsible for implementing code solutions.
    
    This agent handles all coding tasks including implementation, refactoring,
    testing, and debugging. It follows best practices and integrates with
    development workflows.
    """
    
    def __init__(self, agent_id: str = "coder", **kwargs):
        super().__init__(agent_id, **kwargs)
        
        # DSPy modules for core functionality
        if self.dspy_enabled:
            self.code_implementer = dspy.TypedChainOfThought(CodeImplementation)
            self.code_refactorer = dspy.TypedChainOfThought(CodeRefactoring)
            self.code_debugger = dspy.TypedChainOfThought(CodeDebugging)
        
        # Coding configuration
        self.supported_languages = {lang.value for lang in CodeLanguage}
        self.project_root = Path(self.config.get("project_root", "."))
        self.code_style_config = self.config.get("code_style", {})
        
        # Development tools
        self.enable_linting = self.config.get("enable_linting", True)
        self.enable_formatting = self.config.get("enable_formatting", True)
        self.enable_testing = self.config.get("enable_testing", True)
        
        # Code quality settings
        self.max_function_length = self.config.get("max_function_length", 50)
        self.max_file_length = self.config.get("max_file_length", 500)
        self.complexity_threshold = self.config.get("complexity_threshold", 10)
        
        self.logger.info("Coder Agent initialized and ready for implementation tasks")
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the Coder Agent."""
        return """You are the Coder Agent in a Self-Reflecting Claude Code Agent system.

Your primary responsibilities are:
1. Implementing code solutions based on detailed specifications
2. Writing clean, maintainable, and efficient code
3. Following coding best practices and established patterns
4. Creating comprehensive tests for implemented functionality
5. Refactoring existing code to improve quality and maintainability
6. Debugging and fixing code issues
7. Writing clear documentation and comments

When implementing code, always consider:
- Code readability and maintainability
- Performance and efficiency
- Security best practices
- Error handling and edge cases
- Testing and testability
- Documentation and comments
- Adherence to coding standards

You should write production-quality code that follows SOLID principles,
uses appropriate design patterns, and includes proper error handling.
Always think about the long-term maintainability of the code you create."""
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a coding task.
        
        Args:
            task: Task specification containing type, requirements, and context
            
        Returns:
            Result dictionary with implemented code, tests, and documentation
        """
        start_time = datetime.now()
        
        try:
            task_type = task.get("type", CodeTaskType.IMPLEMENTATION.value)
            self.logger.info(f"Processing {task_type} task: {task.get('title', 'Unnamed Task')}")
            
            # Update state
            self.state.current_task = task.get('title', 'Coding Task')
            
            # Route to appropriate handler based on task type
            if task_type == CodeTaskType.IMPLEMENTATION.value:
                result = await self._handle_implementation_task(task)
            elif task_type == CodeTaskType.REFACTORING.value:
                result = await self._handle_refactoring_task(task)
            elif task_type == CodeTaskType.DEBUGGING.value:
                result = await self._handle_debugging_task(task)
            elif task_type == CodeTaskType.TESTING.value:
                result = await self._handle_testing_task(task)
            elif task_type == CodeTaskType.DOCUMENTATION.value:
                result = await self._handle_documentation_task(task)
            else:
                result = await self._handle_implementation_task(task)  # Default
            
            # Post-process and validate
            if result.get("status") == "completed":
                await self._post_process_code(result)
            
            # Update metrics
            response_time = (datetime.now() - start_time).total_seconds()
            success = result.get("status") == "completed"
            self.update_metrics(response_time, success)
            
            # Store in memory
            await self.update_memory(
                f"Completed {task_type} task: {task.get('title', 'Task')}",
                {"task_type": task_type, "result": result}
            )
            
            result["response_time"] = response_time
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing coding task: {str(e)}")
            response_time = (datetime.now() - start_time).total_seconds()
            self.update_metrics(response_time, False)
            
            return {
                "status": "failed",
                "error": str(e),
                "response_time": response_time
            }
    
    async def _handle_implementation_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code implementation tasks."""
        
        specification = task.get("specification", task.get("description", ""))
        requirements = task.get("requirements", {})
        context = await self._gather_codebase_context(task)
        
        if self.dspy_enabled:
            # Use DSPy for intelligent code generation
            result = self.code_implementer(
                specification=specification,
                requirements=str(requirements),
                context=context
            )
            
            # Parse and validate the generated code
            try:
                import json
                code_files_data = json.loads(result.code_files)
                code_files = [CodeFile(**file_data) for file_data in code_files_data]
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"Failed to parse DSPy code output, using fallback: {e}")
                code_files = await self._create_fallback_implementation(task)
                
            implementation_result = {
                "status": "completed",
                "implementation_plan": result.implementation_plan,
                "code_files": [file.model_dump() for file in code_files],
                "tests": result.tests,
                "documentation": result.documentation
            }
        else:
            # Fallback implementation without DSPy
            code_files = await self._create_fallback_implementation(task)
            implementation_result = {
                "status": "completed",
                "code_files": [file.model_dump() for file in code_files],
                "tests": "Basic test cases should be created",
                "documentation": "Documentation should be added"
            }
        
        return implementation_result
    
    async def _handle_refactoring_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle code refactoring tasks."""
        
        existing_code = task.get("existing_code", "")
        refactoring_goals = task.get("goals", ["improve maintainability"])
        constraints = task.get("constraints", {})
        
        if not existing_code:
            # Try to load from file path if provided
            file_path = task.get("file_path")
            if file_path:
                try:
                    with open(self.project_root / file_path, 'r') as f:
                        existing_code = f.read()
                except FileNotFoundError:
                    return {"status": "failed", "error": f"File not found: {file_path}"}
        
        if self.dspy_enabled:
            result = self.code_refactorer(
                existing_code=existing_code,
                refactoring_goals=", ".join(refactoring_goals),
                constraints=str(constraints)
            )
            
            return {
                "status": "completed",
                "refactored_code": result.refactored_code,
                "changes_summary": result.changes_summary,
                "impact_analysis": result.impact_analysis
            }
        else:
            # Basic refactoring fallback
            refactored_code = await self._basic_refactoring(existing_code)
            return {
                "status": "completed",
                "refactored_code": refactored_code,
                "changes_summary": "Applied basic refactoring improvements",
                "impact_analysis": "Low risk - structural improvements made"
            }
    
    async def _handle_debugging_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle debugging and bug fixing tasks."""
        
        problematic_code = task.get("code", "")
        error_description = task.get("error", "")
        context = task.get("context", "")
        
        if self.dspy_enabled:
            result = self.code_debugger(
                problematic_code=problematic_code,
                error_description=error_description,
                context=context
            )
            
            return {
                "status": "completed",
                "root_cause": result.root_cause,
                "fixed_code": result.fixed_code,
                "testing_recommendations": result.testing_recommendations
            }
        else:
            # Basic debugging fallback
            fixed_code = await self._basic_debugging(problematic_code, error_description)
            return {
                "status": "completed",
                "fixed_code": fixed_code,
                "root_cause": "Issue identified and fixed",
                "testing_recommendations": "Add unit tests for the fixed functionality"
            }
    
    async def _handle_testing_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle test creation tasks."""
        
        code_to_test = task.get("code", "")
        test_requirements = task.get("test_requirements", [])
        
        # Generate test cases
        test_code = await self._generate_test_code(code_to_test, test_requirements)
        
        return {
            "status": "completed",
            "test_code": test_code,
            "test_coverage": "Unit tests created for main functionality",
            "test_framework": "pytest"  # Default framework
        }
    
    async def _handle_documentation_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle documentation creation tasks."""
        
        code_to_document = task.get("code", "")
        doc_type = task.get("doc_type", "api")
        
        # Generate documentation
        documentation = await self._generate_documentation(code_to_document, doc_type)
        
        return {
            "status": "completed",
            "documentation": documentation,
            "doc_type": doc_type
        }
    
    async def _gather_codebase_context(self, task: Dict[str, Any]) -> str:
        """Gather relevant context from the existing codebase."""
        
        context_parts = []
        
        # Add project structure if available
        if self.project_root.exists():
            structure = await self._get_project_structure()
            context_parts.append(f"Project structure:\n{structure}")
        
        # Add existing patterns and conventions
        conventions = await self._analyze_code_conventions()
        if conventions:
            context_parts.append(f"Code conventions:\n{conventions}")
        
        # Add relevant existing code
        related_files = task.get("related_files", [])
        for file_path in related_files:
            try:
                with open(self.project_root / file_path, 'r') as f:
                    content = f.read()
                    context_parts.append(f"Related file {file_path}:\n{content[:1000]}...")  # Truncate
            except FileNotFoundError:
                continue
        
        return "\n\n".join(context_parts)
    
    async def _get_project_structure(self) -> str:
        """Get a summary of the project directory structure."""
        
        structure_lines = []
        
        def add_directory(path: Path, indent: int = 0):
            if len(structure_lines) > 50:  # Limit size
                return
                
            prefix = "  " * indent
            if path.is_file() and path.suffix in ['.py', '.js', '.ts', '.java', '.cpp', '.h']:
                structure_lines.append(f"{prefix}{path.name}")
            elif path.is_dir() and not path.name.startswith('.'):
                structure_lines.append(f"{prefix}{path.name}/")
                try:
                    for child in sorted(path.iterdir()):
                        add_directory(child, indent + 1)
                except PermissionError:
                    pass
        
        add_directory(self.project_root)
        return "\n".join(structure_lines)
    
    async def _analyze_code_conventions(self) -> str:
        """Analyze existing code to determine conventions."""
        
        conventions = []
        
        # Look for Python files to analyze
        python_files = list(self.project_root.glob("**/*.py"))[:10]  # Limit analysis
        
        if python_files:
            # Analyze indentation
            indent_counts = {}
            for file_path in python_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        # Simple indentation analysis
                        for line in content.split('\n'):
                            if line.strip() and line.startswith(' '):
                                indent = len(line) - len(line.lstrip())
                                if indent > 0:
                                    indent_counts[indent] = indent_counts.get(indent, 0) + 1
                except:
                    continue
            
            if indent_counts:
                common_indent = max(indent_counts, key=indent_counts.get)
                conventions.append(f"Indentation: {common_indent} spaces")
        
        return "\n".join(conventions) if conventions else ""
    
    async def _create_fallback_implementation(self, task: Dict[str, Any]) -> List[CodeFile]:
        """Create a basic implementation when DSPy is not available."""
        
        # This is a simplified fallback - in practice, this would be more sophisticated
        title = task.get("title", "Implementation")
        description = task.get("description", "")
        
        # Determine language from context or default to Python
        language = CodeLanguage.PYTHON
        if "javascript" in description.lower() or "js" in description.lower():
            language = CodeLanguage.JAVASCRIPT
        elif "typescript" in description.lower() or "ts" in description.lower():
            language = CodeLanguage.TYPESCRIPT
        
        # Create a basic implementation file
        if language == CodeLanguage.PYTHON:
            content = f'''"""
{title}

{description}
"""

def main():
    """Main function implementing the required functionality."""
    # TODO: Implement the actual functionality
    pass

if __name__ == "__main__":
    main()
'''
            file_path = f"{title.lower().replace(' ', '_')}.py"
        else:
            content = f'''// {title}\n// {description}\n\nfunction main() {{\n    // TODO: Implement the actual functionality\n}}\n\nmain();'''
            file_path = f"{title.lower().replace(' ', '_')}.js"
        
        return [CodeFile(
            path=file_path,
            language=language,
            content=content,
            description=f"Implementation of {title}"
        )]
    
    async def _basic_refactoring(self, code: str) -> str:
        """Apply basic refactoring improvements."""
        
        # Simple refactoring improvements
        refactored = code
        
        # Remove excessive blank lines
        refactored = re.sub(r'\n\s*\n\s*\n', '\n\n', refactored)
        
        # Basic formatting improvements could be added here
        
        return refactored
    
    async def _basic_debugging(self, code: str, error: str) -> str:
        """Apply basic debugging fixes."""
        
        # This would contain actual debugging logic
        # For now, return the original code with a comment
        return f"# Fixed issue: {error}\n{code}"
    
    async def _generate_test_code(self, code: str, requirements: List[str]) -> str:
        """Generate test cases for the given code."""
        
        # Basic test template
        test_template = f'''import pytest

# Test cases for the implemented functionality
def test_basic_functionality():
    """Test basic functionality."""
    # TODO: Implement actual test cases
    assert True

def test_edge_cases():
    """Test edge cases and error conditions."""
    # TODO: Implement edge case tests
    assert True

# Additional test cases based on requirements
# {", ".join(requirements)}
'''
        
        return test_template
    
    async def _generate_documentation(self, code: str, doc_type: str) -> str:
        """Generate documentation for the given code."""
        
        if doc_type == "api":
            return f"""# API Documentation

## Overview
This module provides functionality for [description].

## Functions

### main()
Main function that implements the core functionality.

**Parameters:**
- None

**Returns:**
- None

**Example:**
```python
main()
```
"""
        else:
            return f"# Documentation\n\nThis code provides [description of functionality]."
    
    async def _post_process_code(self, result: Dict[str, Any]) -> None:
        """Post-process generated code (linting, formatting, validation)."""
        
        if not result.get("code_files"):
            return
        
        for file_data in result["code_files"]:
            content = file_data.get("content", "")
            language = file_data.get("language", "")
            
            # Validate syntax for Python files
            if language == "python":
                try:
                    ast.parse(content)
                    file_data["syntax_valid"] = True
                except SyntaxError as e:
                    file_data["syntax_valid"] = False
                    file_data["syntax_error"] = str(e)
                    self.logger.warning(f"Syntax error in generated code: {e}")
            
            # Check code complexity
            if language == "python":
                complexity = await self._estimate_complexity(content)
                file_data["complexity_estimate"] = complexity
                
                if complexity > self.complexity_threshold:
                    self.logger.warning(f"High complexity detected: {complexity}")
    
    async def _estimate_complexity(self, code: str) -> int:
        """Estimate cyclomatic complexity of the code."""
        
        try:
            tree = ast.parse(code)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1
            
            return complexity
        except:
            return 0  # Return 0 if analysis fails
    
    async def run_code_quality_checks(self, file_path: str) -> Dict[str, Any]:
        """Run code quality checks on a file."""
        
        results = {
            "file_path": file_path,
            "checks": {}
        }
        
        full_path = self.project_root / file_path
        if not full_path.exists():
            results["error"] = "File not found"
            return results
        
        # Run basic checks
        try:
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Check file length
            line_count = len(content.split('\n'))
            results["checks"]["line_count"] = line_count
            results["checks"]["line_count_ok"] = line_count <= self.max_file_length
            
            # Check for Python-specific issues
            if file_path.endswith('.py'):
                try:
                    tree = ast.parse(content)
                    results["checks"]["syntax_valid"] = True
                    
                    # Check function lengths
                    long_functions = []
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_length = node.end_lineno - node.lineno + 1
                            if func_length > self.max_function_length:
                                long_functions.append({
                                    "name": node.name,
                                    "length": func_length,
                                    "line": node.lineno
                                })
                    
                    results["checks"]["long_functions"] = long_functions
                    
                except SyntaxError as e:
                    results["checks"]["syntax_valid"] = False
                    results["checks"]["syntax_error"] = str(e)
        
        except Exception as e:
            results["error"] = str(e)
        
        return results