"""
Context-Aware Command Suite

Combines ChromaDB semantic context with workflow-specific command templates.
Creates intelligent, memory-enhanced prompts for development workflows.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from chroma_context_manager import ChromaContextManager, ContextLevel


class CommandNamespace(Enum):
    """Available command namespaces"""
    PROJECT = "project"
    DEV = "dev" 
    TEST = "test"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DEPLOY = "deploy"
    DOCS = "docs"
    DEBUG = "debug"
    CONTEXT = "context"


@dataclass
class WorkflowContext:
    """Context specific to a workflow type"""
    namespace: CommandNamespace
    focus_keywords: List[str]
    required_context_types: List[str]
    priority_levels: List[ContextLevel]


class ContextAwareCommandSuite:
    """
    Intelligent command suite that enhances static templates with dynamic context.
    
    Features:
    - Namespace-based command organization
    - Context-aware prompt generation
    - Workflow-specific memory retrieval
    - Historical pattern recognition
    """
    
    def __init__(self, project: str = "default"):
        self.context_manager = ChromaContextManager()
        self.context_manager.set_project(project)
        self.project = project
        
        # Define workflow contexts
        self.workflow_contexts = self._initialize_workflow_contexts()
        
        # Command templates directory
        self.commands_dir = Path(__file__).parent / "commands"
        self.commands_dir.mkdir(exist_ok=True)
        
        # Initialize built-in commands
        self._create_builtin_commands()
    
    def _initialize_workflow_contexts(self) -> Dict[CommandNamespace, WorkflowContext]:
        """Define context requirements for each workflow type"""
        return {
            CommandNamespace.DEV: WorkflowContext(
                namespace=CommandNamespace.DEV,
                focus_keywords=["code", "implementation", "refactor", "debug", "algorithm"],
                required_context_types=["code_pattern", "technical_decision", "bug_fix"],
                priority_levels=[ContextLevel.PROJECT, ContextLevel.SESSION, ContextLevel.IMMEDIATE]
            ),
            CommandNamespace.SECURITY: WorkflowContext(
                namespace=CommandNamespace.SECURITY,
                focus_keywords=["security", "authentication", "authorization", "vulnerability", "encryption"],
                required_context_types=["security_decision", "best_practice", "audit_finding"],
                priority_levels=[ContextLevel.GLOBAL, ContextLevel.PROJECT, ContextLevel.SESSION]
            ),
            CommandNamespace.PERFORMANCE: WorkflowContext(
                namespace=CommandNamespace.PERFORMANCE,
                focus_keywords=["performance", "optimization", "bottleneck", "scaling", "caching"],
                required_context_types=["performance_metric", "optimization", "technical_decision"],
                priority_levels=[ContextLevel.PROJECT, ContextLevel.SESSION, ContextLevel.GLOBAL]
            ),
            CommandNamespace.TEST: WorkflowContext(
                namespace=CommandNamespace.TEST,
                focus_keywords=["test", "testing", "coverage", "quality", "validation"],
                required_context_types=["test_pattern", "bug_fix", "quality_metric"],
                priority_levels=[ContextLevel.PROJECT, ContextLevel.SESSION, ContextLevel.IMMEDIATE]
            ),
            CommandNamespace.PROJECT: WorkflowContext(
                namespace=CommandNamespace.PROJECT,
                focus_keywords=["architecture", "design", "planning", "requirements", "structure"],
                required_context_types=["architectural_decision", "requirement", "design_pattern"],
                priority_levels=[ContextLevel.PROJECT, ContextLevel.GLOBAL, ContextLevel.SESSION]
            )
        }
    
    def _create_builtin_commands(self):
        """Create built-in command templates"""
        commands = {
            "dev/code-review.md": self._get_code_review_template(),
            "security/security-audit.md": self._get_security_audit_template(),
            "performance/performance-audit.md": self._get_performance_audit_template(),
            "test/generate-tests.md": self._get_test_generation_template(),
            "project/architecture-review.md": self._get_architecture_review_template(),
            "debug/troubleshoot.md": self._get_debug_template(),
            "context/export-focused.md": self._get_context_export_template()
        }
        
        for cmd_path, template in commands.items():
            full_path = self.commands_dir / cmd_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if not full_path.exists():
                full_path.write_text(template, encoding='utf-8')
    
    def execute_command(self, 
                       namespace: str, 
                       command: str, 
                       context_query: Optional[str] = None,
                       max_context_items: int = 10) -> str:
        """
        Execute a context-aware command.
        
        Args:
            namespace: Command namespace (dev, security, etc.)
            command: Specific command name
            context_query: Optional specific context to search for
            max_context_items: Maximum context items to include
            
        Returns:
            Enhanced prompt with relevant context
        """
        try:
            namespace_enum = CommandNamespace(namespace)
        except ValueError:
            raise ValueError(f"Unknown namespace: {namespace}")
        
        # Load command template
        cmd_path = self.commands_dir / namespace / f"{command}.md"
        if not cmd_path.exists():
            raise FileNotFoundError(f"Command not found: /{namespace}:{command}")
        
        template = cmd_path.read_text(encoding='utf-8')
        
        # Get workflow-specific context
        workflow_context = self.workflow_contexts.get(namespace_enum)
        if workflow_context:
            context = self._get_workflow_context(
                workflow_context, 
                context_query or command,
                max_context_items
            )
        else:
            # Fallback to general context search
            context = self._get_general_context(context_query or command, max_context_items)
        
        # Combine template with context
        enhanced_prompt = self._enhance_template_with_context(template, context, namespace, command)
        
        return enhanced_prompt
    
    def _get_workflow_context(self, 
                            workflow: WorkflowContext, 
                            query: str, 
                            max_items: int) -> Dict[str, List[Dict]]:
        """Get context specific to a workflow type"""
        context = {}
        
        # Search each priority level
        for level in workflow.priority_levels:
            # Combine query with workflow keywords for better matching
            enhanced_query = f"{query} {' '.join(workflow.focus_keywords[:3])}"
            
            # Create filters that work with ChromaDB (can't use list in filters)
            results = self.context_manager.search_context(
                enhanced_query,
                level=level,
                n_results=max_items // len(workflow.priority_levels)
            )
            
            # Filter results by context type if specified
            if workflow.required_context_types:
                filtered_results = []
                for result in results:
                    if result.get('metadata', {}).get('type') in workflow.required_context_types:
                        filtered_results.append(result)
                results = filtered_results
            
            if results:
                context[level.value] = results
        
        return context
    
    def _get_general_context(self, query: str, max_items: int) -> Dict[str, List[Dict]]:
        """Get general context when no specific workflow is defined"""
        results = self.context_manager.search_context(query, n_results=max_items)
        
        # Group by level
        context = {}
        for result in results:
            level = result['level']
            if level not in context:
                context[level] = []
            context[level].append(result)
        
        return context
    
    def _enhance_template_with_context(self, 
                                     template: str, 
                                     context: Dict[str, List[Dict]], 
                                     namespace: str, 
                                     command: str) -> str:
        """Enhance command template with relevant context"""
        
        # Build context section
        context_sections = []
        context_sections.append(f"# Context-Enhanced /{namespace}:{command}")
        context_sections.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        context_sections.append(f"**Project:** {self.project}\n")
        
        # Add relevant context by level
        if context:
            context_sections.append("## Relevant Project Context")
            
            level_names = {
                "global": "🌐 Universal Best Practices",
                "project": "🏗️ Project-Specific Knowledge", 
                "session": "📝 Recent Decisions & Changes",
                "immediate": "💬 Current Conversation"
            }
            
            for level, items in context.items():
                if items:
                    context_sections.append(f"\n### {level_names.get(level, level.title())}")
                    
                    for i, item in enumerate(items[:5], 1):  # Limit to top 5 per level
                        relevance = max(0, 1 - item['distance']) * 100
                        if relevance > 20:  # Only include reasonably relevant items
                            context_sections.append(f"\n**{i}. [{relevance:.0f}% relevant]**")
                            
                            # Add metadata context
                            if 'type' in item['metadata']:
                                context_sections.append(f"*Type: {item['metadata']['type']}*")
                            
                            # Add content
                            content = item['content'][:500]  # Limit length
                            if len(item['content']) > 500:
                                content += "..."
                            context_sections.append(f"{content}\n")
        
        # Add separator
        context_sections.append("\n" + "="*80)
        context_sections.append("## Command Template\n")
        
        # Combine context with template
        enhanced_prompt = "\n".join(context_sections) + template
        
        return enhanced_prompt
    
    def list_commands(self, namespace: Optional[str] = None) -> Dict[str, List[str]]:
        """List available commands by namespace"""
        commands = {}
        
        if namespace:
            # List commands in specific namespace
            ns_dir = self.commands_dir / namespace
            if ns_dir.exists():
                commands[namespace] = [
                    f.stem for f in ns_dir.glob("*.md")
                ]
        else:
            # List all commands
            for ns_dir in self.commands_dir.iterdir():
                if ns_dir.is_dir():
                    ns_name = ns_dir.name
                    commands[ns_name] = [
                        f.stem for f in ns_dir.glob("*.md")
                    ]
        
        return commands
    
    def add_context_to_workflow(self, 
                               namespace: str, 
                               content: str, 
                               context_type: str,
                               metadata: Optional[Dict[str, Any]] = None):
        """Add context specific to a workflow"""
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "workflow": namespace,
            "type": context_type,
            "added_via": "command_suite"
        })
        
        # Determine appropriate context level based on workflow
        level = ContextLevel.SESSION  # Default
        if namespace in ["security", "performance"]:
            level = ContextLevel.PROJECT  # These tend to be project-wide
        elif namespace == "project":
            level = ContextLevel.PROJECT
        
        self.context_manager.add_context(content, level, metadata)
    
    # Command Templates
    def _get_code_review_template(self) -> str:
        return """
Perform a comprehensive code review with the following focus areas:

## Code Quality Analysis
1. **Architecture & Design Patterns**
   - Evaluate overall code structure and organization
   - Check for proper separation of concerns
   - Identify design pattern usage and appropriateness

2. **Code Readability & Maintainability**
   - Assess variable and function naming conventions
   - Review code documentation and comments
   - Check for code complexity and potential simplifications

3. **Performance Considerations**
   - Identify potential performance bottlenecks
   - Review algorithm efficiency
   - Check for proper resource management

4. **Security Review**
   - Look for common security vulnerabilities
   - Review input validation and sanitization
   - Check for proper error handling

## Specific Recommendations
Based on the context provided above, pay special attention to:
- Previously identified patterns and anti-patterns in this project
- Recent technical decisions that might affect this code
- Security considerations relevant to this codebase
- Performance optimizations used elsewhere in the project

## Output Format
Provide specific, actionable recommendations with:
- Priority level (High/Medium/Low)
- Code examples where applicable
- References to project-specific standards or decisions
"""

    def _get_security_audit_template(self) -> str:
        return """
Conduct a thorough security audit focusing on:

## Security Assessment Areas

1. **Authentication & Authorization**
   - Review authentication mechanisms
   - Check authorization logic and access controls
   - Validate session management

2. **Input Validation & Sanitization**
   - Examine all input validation points
   - Check for injection vulnerabilities (SQL, XSS, etc.)
   - Review data sanitization processes

3. **Data Protection**
   - Assess data encryption at rest and in transit
   - Review handling of sensitive information
   - Check for proper secret management

4. **Infrastructure Security**
   - Review deployment configuration
   - Check for secure communication protocols
   - Assess error handling and information disclosure

## Context-Specific Analysis
Based on project history and previous security decisions:
- Apply security standards established for this project
- Reference previous security audits and their resolutions
- Consider project-specific threat models and requirements

## Deliverables
- Prioritized list of security findings
- Specific remediation recommendations
- References to security best practices relevant to this project
"""

    def _get_performance_audit_template(self) -> str:
        return """
Analyze application performance across key areas:

## Performance Analysis

1. **Code Performance**
   - Identify algorithmic inefficiencies
   - Review database query optimization
   - Check for unnecessary computations or loops

2. **Resource Utilization**
   - Assess memory usage patterns
   - Review CPU-intensive operations
   - Check for resource leaks

3. **Scalability Considerations**
   - Evaluate scalability bottlenecks
   - Review caching strategies
   - Assess load handling capabilities

4. **Network & I/O Performance**
   - Review API response times
   - Check for unnecessary network calls
   - Assess file I/O efficiency

## Historical Performance Context
Consider previous performance optimizations and decisions:
- Apply performance patterns successful in this project
- Reference previous bottlenecks and their solutions
- Consider performance requirements and SLAs

## Recommendations
Provide prioritized performance improvements with:
- Expected performance impact
- Implementation complexity
- Resource requirements
"""

    def _get_test_generation_template(self) -> str:
        return """
Generate comprehensive test cases covering:

## Test Strategy

1. **Unit Tests**
   - Test individual function/method behavior
   - Cover edge cases and error conditions
   - Ensure proper mocking of dependencies

2. **Integration Tests**
   - Test component interactions
   - Verify data flow between modules
   - Check external service integrations

3. **End-to-End Tests**
   - Test complete user workflows
   - Verify system behavior from user perspective
   - Check critical business processes

## Context-Driven Testing
Based on project history and patterns:
- Apply testing patterns used successfully in this project
- Reference previous bug reports for test case ideas
- Consider project-specific quality requirements

## Test Implementation
Generate specific test code with:
- Clear test descriptions and purpose
- Proper setup and teardown procedures
- Assertions that validate expected behavior
- Error case handling
"""

    def _get_architecture_review_template(self) -> str:
        return """
Review system architecture and design:

## Architecture Analysis

1. **System Design**
   - Evaluate overall system architecture
   - Review component relationships and dependencies
   - Assess design pattern implementation

2. **Scalability & Performance**
   - Analyze system scalability characteristics
   - Review performance bottlenecks
   - Assess resource utilization patterns

3. **Maintainability & Extensibility**
   - Evaluate code organization and modularity
   - Review interface design and contracts
   - Assess system flexibility for future changes

## Historical Architecture Context
Consider previous architectural decisions:
- Reference established architectural patterns in this project
- Apply lessons learned from previous design decisions
- Consider project-specific constraints and requirements

## Recommendations
Provide architectural improvements with:
- Impact on system quality attributes
- Implementation strategy and timeline
- Risk assessment and mitigation approaches
"""

    def _get_debug_template(self) -> str:
        return """
Systematic debugging and troubleshooting:

## Problem Analysis

1. **Issue Identification**
   - Clearly define the problem or unexpected behavior
   - Identify affected components or systems
   - Determine scope and impact

2. **Root Cause Analysis**
   - Trace the issue to its source
   - Identify contributing factors
   - Review recent changes that might be related

3. **Solution Development**
   - Propose multiple solution approaches
   - Evaluate pros and cons of each approach
   - Select optimal solution strategy

## Context-Informed Debugging
Leverage project knowledge:
- Reference similar issues resolved in the past
- Apply debugging patterns successful in this project
- Consider project-specific constraints and requirements

## Resolution Plan
Provide detailed resolution with:
- Step-by-step implementation plan
- Testing strategy to verify the fix
- Prevention measures for future occurrences
"""

    def _get_context_export_template(self) -> str:
        return """
Export focused context for specific workflow:

## Context Export Configuration

**Workflow Focus:** {workflow_type}
**Context Query:** {context_query}
**Project:** {project_name}

## Relevant Context
The above context has been filtered based on:
- Semantic similarity to your current work
- Historical relevance to similar tasks
- Project-specific patterns and decisions

## Usage Instructions
This context can be used to:
- Inform decision-making for similar tasks
- Reference previous solutions and patterns
- Maintain consistency with project standards
- Learn from past successes and challenges

Use this context to enhance your current work while maintaining project continuity.
"""


# CLI Interface
def main():
    """Command-line interface for the Context-Aware Command Suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Context-Aware Command Suite")
    parser.add_argument("project", help="Project name")
    parser.add_argument("command", help="Command in format 'namespace:command'")
    parser.add_argument("-q", "--query", help="Specific context query")
    parser.add_argument("-n", "--max-items", type=int, default=10, help="Max context items")
    parser.add_argument("-l", "--list", action="store_true", help="List available commands")
    
    args = parser.parse_args()
    
    suite = ContextAwareCommandSuite(args.project)
    
    if args.list:
        commands = suite.list_commands()
        print("Available Commands:")
        for namespace, cmds in commands.items():
            print(f"\n/{namespace}:")
            for cmd in cmds:
                print(f"  - {cmd}")
        return
    
    # Parse command
    if ":" not in args.command:
        print("Error: Command must be in format 'namespace:command'")
        return
    
    namespace, command = args.command.split(":", 1)
    
    try:
        result = suite.execute_command(namespace, command, args.query, args.max_items)
        print(result)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()