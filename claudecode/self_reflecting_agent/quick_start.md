# Self-Reflecting Agent Quick Start Guide

This guide will get you up and running with the Self-Reflecting Claude Code Agent system in minutes.

## Installation

### 1. Install the Package Globally

```bash
# Navigate to the agent system directory
cd D:\VS Workspace\claudecode\self_reflecting_agent

# Run the installation script
python install.py
```

The installation script will:
- Install the package globally using pip
- Create global configuration directories
- Verify CLI commands are available
- Set up initial configuration

### 2. Verify Installation

```bash
# Test the CLI commands
sra --help
self-reflecting-agent --help

# Show system information
sra info
```

## Basic Usage

### 1. Execute Development Tasks

```bash
# Simple task execution
sra task "Create a REST API for user management"

# Task with additional requirements
sra task "Build a web scraper" --requirements '{"rate_limit": "respectful", "format": "JSON"}'

# Task with constraints
sra task "Optimize database queries" --constraints '{"no_breaking_changes": true}'
```

### 2. Use Domain Workflows

```bash
# Architecture review
sra workflow software_development architecture_review "Review my microservices design"

# Code quality audit
sra workflow software_development code_quality_audit "Audit Python code for best practices"

# Comprehensive project planning
sra workflow software_development comprehensive_project_planning "Plan an e-commerce platform"

# Web application planning
sra workflow software_development web_application_planning "Plan a social media app"
```

### 3. Get System Information

```bash
# Show available domains and agents
sra info

# Save results to project directory
sra task "Create documentation" --save-results
```

## Project Integration

### 1. Create CLAUDE.md Configuration

The agent system automatically detects and uses `CLAUDE.md` files in your project root. Create one to customize agent behavior:

```markdown
# My Project Agent Configuration

## Project Context
**Project Type**: Web Application  
**Technologies**: Python, FastAPI, PostgreSQL, React, TypeScript

## Agent Configuration
### Preferred Workflows
- `comprehensive_project_planning` - Multi-perspective project planning
- `architecture_review` - System design and scalability analysis
- `code_quality_audit` - Code quality and best practices review

### Domain Agents Available
- **architect**: System design, scalability, microservices
- **security_auditor**: Vulnerability assessment, compliance
- **performance_auditor**: Performance optimization, bottlenecks

## Development Guidelines
### Code Quality Standards
- Follow SOLID principles strictly
- Implement comprehensive error handling
- Use type hints throughout
- Maintain test coverage > 80%

### Architecture Principles
- Favor composition over inheritance
- Use dependency injection for testability
- Design for horizontal scalability

## Project Constraints
- **Performance**: Sub-second response times for API endpoints
- **Security**: Follow OWASP guidelines
- **Testing**: All features must have automated tests
```

### 2. Automatic Project Detection

The system automatically detects:
- **Project Type**: Based on files like `package.json`, `requirements.txt`, etc.
- **Technologies**: From CLAUDE.md and project files
- **Context**: Git repositories, documentation, existing code

### 3. Persistent Memory

The agent system maintains memory across sessions:
- **Project History**: Remembers previous tasks and solutions
- **Preferences**: Learns your coding style and preferences  
- **Context**: Maintains awareness of project evolution

## Available Workflows

### Software Development Domain

#### Multi-Perspective Planning
- **comprehensive_project_planning**: 5-perspective comprehensive planning (architecture, security, performance, quality, devops)
- **web_application_planning**: 4-perspective web app planning with UX focus
- **microservices_planning**: 4-perspective distributed systems planning

#### Traditional Workflows  
- **architecture_review**: Comprehensive architecture analysis (parallel)
- **code_quality_audit**: Code quality assessment (sequential)
- **system_analysis**: Security, dependencies, and performance analysis (parallel)
- **migration_planning**: Migration strategy development (sequential)

## Advanced Usage

### 1. Multi-Perspective Planning

```bash
# Comprehensive planning with all perspectives
sra workflow software_development comprehensive_project_planning "Plan a video streaming platform" --context '{
  "requirements": {
    "functional": {
      "video_upload": "support multiple formats",
      "streaming": "adaptive bitrate streaming",
      "user_management": "profiles and subscriptions"
    },
    "non_functional": {
      "performance": "handle 100k concurrent users",
      "scalability": "global CDN deployment",
      "availability": "99.99% uptime"
    }
  },
  "constraints": {
    "timeline": "8 months to production",
    "budget": "enterprise level"
  }
}'
```

### 2. Specialized Agent Usage

The system provides specialized agents for different aspects of software development:

```bash
# Architecture-focused analysis
sra workflow software_development architecture_review "Analyze scalability of current system"

# Security-focused review
sra workflow software_development system_analysis "Security audit of authentication system" 

# Performance optimization
sra workflow software_development system_analysis "Identify performance bottlenecks"
```

### 3. Context-Aware Development

The agent system uses project context for better results:

```bash
# In a Python project, it will automatically:
# - Use Python best practices
# - Suggest appropriate frameworks
# - Consider existing dependencies
# - Apply Python-specific design patterns

# In a JavaScript project, it will:
# - Use modern ES6+ features
# - Consider React/Vue/Angular patterns
# - Apply web performance optimizations
# - Suggest appropriate testing frameworks
```

## Global Configuration

### Configuration Locations
- **Windows**: `%APPDATA%\SelfReflectingAgent\`
- **Unix/Mac**: `~/.self_reflecting_agent/`

### Configuration Files
- `global_config.yaml`: Global agent settings
- `projects.json`: Project database and history
- Logs and cache files

### Customizing Global Settings

Edit `global_config.yaml` to customize:
- Default agent models and temperatures
- Memory and self-improvement settings  
- Project type preferences
- Domain agent configurations

## Troubleshooting

### Common Issues

1. **Commands not found after installation**
   - Restart your terminal
   - Check if Python Scripts directory is in PATH
   - Verify installation with `pip list | grep self-reflecting-agent`

2. **Permission errors on Windows**
   - Run terminal as Administrator
   - Or install with `--user` flag: `pip install -e . --user`

3. **Module import errors**
   - Ensure all dependencies are installed
   - Check Python version is 3.8+
   - Try reinstalling: `pip uninstall self-reflecting-agent && python install.py`

### Getting Help

```bash
# Show help for all commands
sra --help

# Show help for specific command
sra task --help
sra workflow --help

# Show detailed system information
sra info
```

## Next Steps

1. **Explore Examples**: Check the `examples/` directory for detailed usage examples
2. **Read Documentation**: See `README.md` for comprehensive information
3. **Customize Configuration**: Create project-specific `CLAUDE.md` files
4. **Try Workflows**: Experiment with different domain workflows
5. **Provide Feedback**: The system learns and improves from usage

## Integration with Claude Code

The agent system automatically integrates with Claude Code:
- Detects when you start Claude Code in a new directory
- Shows available capabilities and suggested actions
- Provides context-aware development assistance
- Maintains project memory across sessions

Start Claude Code in any project directory and look for the agent system activation message!