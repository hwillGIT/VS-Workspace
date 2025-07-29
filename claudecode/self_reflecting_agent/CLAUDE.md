# Self-Reflecting Claude Code Agent Integration

This file configures the Self-Reflecting Agent system to work optimally with this project.

## Project Context

**Project Type**: Multi-Agent AI Development Framework  
**Primary Domain**: Software Development  
**Technologies**: Python, LangGraph, DSPy, FastAPI, PyYAML, asyncio

## Core Capabilities

- **Multi-Agent Architecture**: Manager-Worker pattern with specialized domain agents
- **Self-Improvement**: Continuous learning through LLM-as-Judge evaluation
- **Domain Expertise**: Specialized agents for software development, security, performance
- **Context Engineering**: Dynamic context management to prevent context poisoning
- **Hybrid RAG**: BM25 + vector search with intelligent fusion
- **Persistent Memory**: Long-term memory with mem0 integration

## Agent Configuration

### Preferred Workflows
- `comprehensive_project_planning` - Multi-perspective project planning
- `architecture_review` - System design and scalability analysis
- `code_quality_audit` - Code quality and best practices review
- `system_analysis` - Security, dependencies, and performance analysis

### Domain Agents Available
- **architect**: System design, scalability, microservices
- **security_auditor**: Vulnerability assessment, compliance
- **performance_auditor**: Performance optimization, bottlenecks
- **design_patterns_expert**: Pattern identification, refactoring
- **solid_principles_expert**: SOLID principles evaluation
- **documentation_agent**: Technical documentation, API docs
- **dependency_analyzer**: Dependency analysis, security scanning
- **technical_analyst**: Code metrics, complexity analysis
- **migration_planner**: Migration strategies, compatibility

## Development Guidelines

### Code Quality Standards
- Follow SOLID principles strictly
- Implement comprehensive error handling
- Use type hints throughout
- Maintain test coverage > 80%
- Document all public APIs

### Architecture Principles
- Favor composition over inheritance
- Use dependency injection for testability
- Implement proper separation of concerns
- Design for horizontal scalability
- Follow security-by-design principles

### Testing Requirements
- Unit tests for all core functionality
- Integration tests for workflows
- Property-based testing for DSPy components
- Performance benchmarks for critical paths

## Project Constraints

- **Compatibility**: Python 3.8+ required
- **Dependencies**: Minimize external dependencies where possible
- **Performance**: Sub-second response times for simple tasks
- **Memory**: Efficient context management for large codebases
- **Security**: No secrets in configuration files

## Integration Hooks

### Pre-execution Checks
- Validate project structure
- Check for required dependencies
- Verify configuration files
- Load domain-specific contexts

### Post-execution Actions
- Update project documentation
- Cache generated artifacts
- Update memory with new learnings
- Log execution metrics

## Optimization Preferences

### Context Management
- Prioritize recent file changes
- Include relevant architecture documentation
- Load domain-specific knowledge
- Maintain conversation continuity

### Agent Selection
- Use architect for design questions
- Use security_auditor for security reviews
- Use performance_auditor for optimization
- Use comprehensive_project_planning for complex projects

### Memory Integration
- Store successful patterns and solutions
- Remember project-specific preferences
- Cache frequently used code templates
- Track performance improvements over time

## Usage Examples

```bash
# Comprehensive project planning
sra workflow software_development comprehensive_project_planning "Plan a microservices e-commerce platform"

# Architecture review
sra workflow software_development architecture_review "Review the current system design for scalability"

# Code quality audit
sra workflow software_development code_quality_audit "Audit the codebase for SOLID principles and design patterns"

# General development task
sra task "Implement async error handling with proper logging and metrics"
```

## Notes

This agent system is designed to be a sophisticated development partner that learns and improves over time. It combines multiple AI techniques to provide comprehensive development assistance while maintaining high code quality and architectural standards.