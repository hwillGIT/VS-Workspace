# Self-Reflecting Claude Code Agent

A sophisticated agent framework combining LangGraph orchestration with DSPy optimization for autonomous code development.

## Architecture Overview

This system implements a hybrid architecture that combines:
- **LangGraph**: Stateful orchestration and workflow management
- **DSPy**: Optimizable agent cognition and prompting
- **Hybrid RAG**: BM25 + vector search with Reciprocal Rank Fusion
- **mem0**: Persistent memory with dual storage (vector + graph database)
- **MLflow**: Observability and experiment tracking
- **Context Engineering**: Dynamic context management to prevent context poisoning

## Key Features

- **Multi-Agent System**: Manager-Worker pattern with specialized agents
- **Self-Improvement Loop**: Continuous learning through LLM-as-Judge evaluation
- **Graduated Autonomy**: Human-in-the-loop before full automation
- **Specification-First**: DESIGN.md and PLAN.md guided development
- **Parallel Planning Integration**: Compatible with existing parallel planning framework

## Directory Structure

```
self_reflecting_agent/
├── agents/              # Core agent implementations
├── domains/             # Domain-specific specialized agents
│   ├── software_development/  # Software engineering agents
│   ├── financial_trading/     # Trading and finance agents (optional)
│   └── data_science/          # Data science agents (optional)
├── workflows/           # LangGraph workflow definitions
├── rag/                # Hybrid RAG system
├── memory/             # mem0 integration and memory management
├── context/            # Context engineering framework
├── evaluation/         # Self-improvement and evaluation
├── tools/              # Agent tools and utilities
├── templates/          # Specification templates
├── examples/           # Usage examples
└── tests/              # Comprehensive test suite
```

## Quick Start

```python
from self_reflecting_agent import SelfReflectingAgent

# Initialize the agent system
agent = SelfReflectingAgent(
    project_path="./my_project",
    enable_memory=True,
    enable_self_improvement=True
)

# Execute a development task
result = await agent.execute_task(
    "Implement a REST API for user management with authentication"
)

# Execute a general development task
result = await agent.execute_task(
    task_description="Create a secure user authentication system",
    requirements={"security": "high", "frameworks": ["FastAPI", "JWT"]},
    constraints={"timeline": "1 week", "testing": "required"}
)
```

## Domain-Specific Agents

The system now supports **domain-specific agents** organized by expertise areas:

### Software Development Domain

Specialized agents for comprehensive software engineering:

```python
# Execute a domain-specific workflow
result = await agent.execute_domain_workflow(
    domain_name="software_development",
    workflow_name="architecture_review", 
    task_description="Review microservices architecture for scalability",
    task_context={
        "system_type": "microservices",
        "expected_load": "10k concurrent users",
        "technologies": ["Python", "FastAPI", "PostgreSQL", "Redis"]
    }
)

# Use specific domain agents directly
architect = agent.get_domain_agent("software_development", "architect")
security_auditor = agent.get_domain_agent("software_development", "security_auditor")
performance_auditor = agent.get_domain_agent("software_development", "performance_auditor")
```

**Available Software Development Agents:**
- **architect**: System design, scalability analysis, technical debt assessment
- **security_auditor**: Vulnerability assessment, secure coding practices, compliance
- **performance_auditor**: Performance profiling, optimization, bottleneck analysis
- **design_patterns_expert**: Pattern identification, refactoring guidance
- **solid_principles_expert**: SOLID principles evaluation and improvement
- **documentation_agent**: API docs, architectural documentation, user guides
- **dependency_analyzer**: Circular dependencies, security vulnerabilities, licensing
- **technical_analyst**: Code metrics, complexity analysis, maintainability assessment
- **migration_planner**: Migration strategies, compatibility planning, risk assessment

**Available Workflows:**

*Traditional Agent Workflows:*
- **architecture_review**: Comprehensive architecture analysis (parallel)
- **code_quality_audit**: Code quality assessment (sequential)
- **system_analysis**: Security, dependencies, and performance analysis (parallel)
- **migration_planning**: Migration strategy development (sequential)

*Multi-Perspective Planning Workflows:*
- **comprehensive_project_planning**: 5-perspective comprehensive planning (architecture, security, performance, quality, devops)
- **web_application_planning**: 4-perspective web app planning with UX focus
- **microservices_planning**: 4-perspective distributed systems planning

### Configuration

Domain agents are configured in `domains/software_development/config.yaml`:

```yaml
agents:
  architect:
    model: "gpt-4o"
    max_tokens: 8000
    temperature: 0.1
    specializations:
      - "system_design"
      - "scalability_analysis"
      - "microservices_design"
      
workflows:
  architecture_review:
    agents: ["architect", "security_auditor", "performance_auditor"]
    sequence: "parallel"
    aggregation: "consensus"
```

### Multi-Perspective Planning

The system now includes **multi-perspective planning** capabilities inspired by parallel planning frameworks:

```python
# Comprehensive project planning from 5 perspectives
result = await agent.execute_domain_workflow(
    domain_name="software_development",
    workflow_name="comprehensive_project_planning",
    task_description="Plan an e-commerce platform",
    task_context={
        "project_type": "e_commerce_platform",
        "requirements": {
            "functional": {
                "user_management": "authentication, profiles, roles",
                "product_catalog": "browse, search, inventory",
                "payment_processing": "secure transactions, multiple methods"
            },
            "non_functional": {
                "performance": "10k concurrent users",
                "scalability": "horizontal scaling",
                "security": "PCI DSS compliance"
            }
        },
        "constraints": {
            "timeline": "6 months to MVP",
            "budget": "moderate constraints"
        }
    }
)

# Returns synthesized plan with:
# - Executive summary
# - Technical specification from all perspectives
# - Implementation roadmap
# - Individual perspective plans
# - Conflict resolutions
# - Validation results
```

**Planning Perspectives Available:**
- **Architecture**: System design, scalability, technical debt assessment
- **Security**: Threat modeling, compliance, data protection
- **Performance**: Optimization, caching, resource management
- **Code Quality**: Patterns, testing, maintainability
- **DevOps**: CI/CD, infrastructure, monitoring

### Usage Examples

```python
# Check available domains
domains = agent.list_available_domains()
print(f"Available domains: {domains}")

# List agents in a domain
sw_agents = agent.list_domain_agents("software_development")
print(f"Software development agents: {sw_agents}")

# Get domain statistics
stats = agent.get_domain_statistics()
```

## Phase Implementation

- **Phase 1**: Basic scaffolding, core agents, workflows
- **Phase 2**: Advanced features, memory, self-improvement, observability
- **Phase 3**: Domain-specific agents and specialized workflows

See individual component documentation for detailed usage information.