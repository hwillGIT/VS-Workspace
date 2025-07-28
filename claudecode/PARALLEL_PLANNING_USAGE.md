# Parallel Planning Mode - Complete Usage Guide

This guide shows you how to use the parallel planning framework to implement sophisticated plan mode thinking across different types of projects.

## 🚀 Quick Start

### 1. Generate a Planning Configuration

```bash
# Generate a trading system planning configuration
cd parallel_planning/tools
python plan_generator.py generate --template trading_system --output my_trading_plan.json

# Generate a web application planning configuration  
python plan_generator.py generate --template web_application --output my_web_app_plan.json

# Create a custom configuration with specific perspectives
python plan_generator.py custom --project-type data_platform --perspectives technical security performance operational --output custom_plan.json
```

### 2. Execute Parallel Planning

```python
from parallel_planning.core.parallel_planner import ParallelPlanner, PlanningContext

# Create planner from configuration
planner = ParallelPlanner("my_trading_plan.json")

# Set planning context
context = PlanningContext(
    project_type="trading_system",
    problem_description="Build high-frequency algorithmic trading platform with sub-millisecond latency",
    requirements={
        "functional": {
            "trading_strategies": ["market_making", "statistical_arbitrage"],
            "asset_classes": ["equities", "options"],
            "latency_requirements": "sub_millisecond"
        }
    },
    constraints={
        "budget": "$2M",
        "timeline": "12 months",
        "team_size": "10 developers"
    },
    stakeholders=["traders", "risk_managers", "compliance"]
)
planner.set_planning_context(context)

# Execute parallel planning
results = await planner.execute_parallel_planning()

# Save results
planner.save_results(Path("trading_system_plan.json"))
```

### 3. Use Pre-built Workflows

```python
from parallel_planning.core.parallel_planner import (
    create_trading_system_planner,
    create_software_development_planner
)

# Trading system planning
planner = create_trading_system_planner(
    "Build algorithmic trading platform",
    {"latency": "sub_millisecond", "throughput": "100k_orders_per_second"}
)

# Web application planning
planner = create_software_development_planner(
    "Build e-commerce platform", 
    {"users": "1M+", "availability": "99.9%"}
)

results = await planner.execute_parallel_planning()
```

## 📋 Available Project Templates

### Trading System Template
```bash
python plan_generator.py generate --template trading_system --output config.json
```
**Perspectives**: Trading Logic, Risk Management, Performance & Latency, Market Data, Regulatory Compliance, Infrastructure

**Best For**: Algorithmic trading platforms, market making systems, portfolio management

### Web Application Template
```bash
python plan_generator.py generate --template web_application --output config.json
```
**Perspectives**: Frontend Architecture, Backend Architecture, Security & Privacy, Performance, DevOps

**Best For**: Web applications, e-commerce platforms, content management systems

### Data Platform Template
```bash
python plan_generator.py generate --template data_platform --output config.json
```
**Perspectives**: Data Architecture, Processing Engine, Storage & Retrieval, Analytics, Operations

**Best For**: Data warehouses, ETL pipelines, analytics platforms

## 🎯 Perspective Types

### Technical Perspectives
- **Technical Architecture** (`code-architect`) - System design, technology stack, implementation approach
- **Performance & Scalability** (`general-purpose`) - Performance optimization, scalability planning
- **Data Architecture** (`code-architect`) - Data modeling, storage, processing pipelines

### Quality & Security Perspectives  
- **Security & Compliance** (`code-reviewer`) - Security controls, threat modeling, compliance
- **Quality Assurance** (`code-reviewer`) - Testing strategies, quality gates, validation

### Business & Operations Perspectives
- **User Experience** (`general-purpose`) - Usability, accessibility, interface design
- **Operations & DevOps** (`code-architect`) - Deployment, monitoring, maintenance
- **Business Requirements** (`general-purpose`) - Stakeholder needs, success criteria

## 🔧 Customization Examples

### Custom Trading System with Specific Focus

```json
{
  "project_type": "high_frequency_trading",
  "perspectives": [
    {
      "perspective_id": "ultra_low_latency",
      "name": "Ultra-Low Latency Optimization",
      "agent_type": "code-architect",
      "focus_areas": ["hardware_acceleration", "kernel_bypass", "lock_free_programming"],
      "priority": 1,
      "constraints": {"latency_budget": "100_microseconds"}
    },
    {
      "perspective_id": "regulatory_compliance",
      "name": "Regulatory Compliance",
      "agent_type": "code-reviewer", 
      "focus_areas": ["mifid_ii", "reg_nms", "audit_trails"],
      "priority": 1,
      "constraints": {"compliance_mandatory": true}
    }
  ]
}
```

### Custom Web Application with Security Focus

```json
{
  "project_type": "secure_web_application",
  "perspectives": [
    {
      "perspective_id": "security_architecture",
      "name": "Security Architecture",
      "agent_type": "code-reviewer",
      "focus_areas": ["zero_trust", "encryption", "access_control"],
      "priority": 1,
      "constraints": {"security_level": "high"}
    },
    {
      "perspective_id": "privacy_compliance", 
      "name": "Privacy Compliance",
      "agent_type": "code-reviewer",
      "focus_areas": ["gdpr", "ccpa", "data_protection"],
      "priority": 1,
      "constraints": {"privacy_mandatory": true}
    }
  ]
}
```

## 🎨 Advanced Usage Patterns

### Multi-Phase Planning

```python
# Phase 1: High-level architecture planning
architecture_planner = ParallelPlanner("architecture_config.json")
architecture_results = await architecture_planner.execute_parallel_planning()

# Phase 2: Detailed implementation planning using architecture results
implementation_context = PlanningContext(
    project_type="implementation_planning",
    problem_description="Detailed implementation based on architecture",
    requirements={"architecture_decisions": architecture_results},
    # ... other context
)

implementation_planner = ParallelPlanner("implementation_config.json")
implementation_planner.set_planning_context(implementation_context)
implementation_results = await implementation_planner.execute_parallel_planning()
```

### Conditional Perspective Planning

```python
# Dynamic perspective selection based on project characteristics
perspectives = ["technical", "security"]

if project_requirements.get("high_performance"):
    perspectives.append("performance")

if project_requirements.get("regulatory_domain"):
    perspectives.append("compliance")

if project_requirements.get("user_facing"):
    perspectives.append("user_experience")

# Generate custom configuration
generator.create_custom_template(
    project_type="dynamic_project",
    perspectives=perspectives,
    output_path=Path("dynamic_config.json")
)
```

### Plan Validation and Iteration

```python
# Execute initial planning
results = await planner.execute_parallel_planning()

# Validate results
validation_issues = validate_plan_completeness(results)

if validation_issues:
    # Refine planning context based on issues
    refined_context = refine_context_based_on_issues(context, validation_issues)
    planner.set_planning_context(refined_context)
    
    # Re-execute with refined context
    results = await planner.execute_parallel_planning()
```

## 🛠️ CLI Tool Usage

### List Available Options

```bash
# List all available templates
python plan_generator.py list-templates

# List all specifications  
python plan_generator.py list-specs
```

### Generate Configurations

```bash
# Generate from template
python plan_generator.py generate \
  --template trading_system \
  --output my_config.json

# Generate with customizations
python plan_generator.py generate \
  --template web_application \
  --output my_config.json \
  --customize customizations.json

# Create completely custom template
python plan_generator.py custom \
  --project-type microservices_platform \
  --perspectives technical security performance operational \
  --output microservices_config.json
```

### Execute Planning

```bash
# Execute planning workflow
python plan_generator.py plan \
  --config my_config.json \
  --problem "Build scalable microservices platform" \
  --requirements requirements.json

# Validate configuration
python plan_generator.py validate --config my_config.json
```

## 📊 Understanding Results

### Perspective Plans
Each perspective produces a detailed plan with:
- **Summary**: High-level approach from that perspective
- **Key Decisions**: Critical decisions and trade-offs
- **Implementation Steps**: Detailed steps with timelines
- **Risks**: Identified risks and mitigation strategies
- **Dependencies**: Dependencies on other perspectives or external factors

### Unified Plan
The synthesis produces:
- **Project Summary**: Integrated overview
- **Implementation Approach**: Unified strategy
- **Architecture Overview**: Integrated architecture
- **Development Phases**: Step-by-step implementation plan
- **Risk Assessment**: Comprehensive risk analysis
- **Resource Requirements**: Complete resource planning

### Conflict Resolution
The system identifies and resolves:
- **Timeline Conflicts**: Different perspectives with conflicting timelines
- **Resource Conflicts**: Competing resource requirements
- **Technical Conflicts**: Incompatible technical decisions
- **Priority Conflicts**: Different priority assessments

## 🎯 Best Practices

### Problem Definition
- **Be Specific**: Provide detailed problem descriptions
- **Include Context**: Add business context and constraints
- **Define Success**: Clear success criteria and metrics
- **Identify Stakeholders**: List all relevant stakeholders

### Perspective Selection
- **Match Project Needs**: Choose perspectives relevant to your project
- **Balance Coverage**: Ensure all critical areas are covered
- **Avoid Redundancy**: Don't overlap perspectives unnecessarily
- **Consider Dependencies**: Account for perspective interdependencies

### Configuration Management
- **Version Control**: Track configuration changes
- **Documentation**: Document customizations and rationale
- **Validation**: Always validate configurations before use
- **Iteration**: Refine configurations based on results

### Result Analysis
- **Review All Perspectives**: Don't ignore any perspective results
- **Understand Conflicts**: Analyze why conflicts occur
- **Validate Synthesis**: Ensure unified plan makes sense
- **Plan Implementation**: Use results to guide actual implementation

## 🔄 Integration with Development Workflow

### Pre-Development Planning
```python
# Use for initial project planning
planning_results = await execute_parallel_planning(project_requirements)
architecture_decisions = extract_architecture_decisions(planning_results)
implementation_roadmap = generate_roadmap(planning_results)
```

### Mid-Development Replanning
```python
# Use for major changes or pivots
current_state = assess_current_implementation()
new_requirements = gather_updated_requirements()
revised_plan = await replan_with_current_state(current_state, new_requirements)
```

### Post-Development Review
```python
# Use for retrospective analysis
implementation_results = assess_implementation_quality()
planning_accuracy = compare_plan_vs_reality(original_plan, implementation_results)
lessons_learned = extract_lessons_learned(planning_accuracy)
```

## 🚀 Production Deployment

The parallel planning framework is designed to integrate with your existing development workflow:

1. **Planning Phase**: Use parallel planning for comprehensive project planning
2. **Implementation Phase**: Follow the unified plan with regular check-ins
3. **Review Phase**: Compare actual implementation against planned approach
4. **Iteration Phase**: Use learnings to improve future planning

This comprehensive parallel planning system enables sophisticated, multi-perspective project planning that considers all critical aspects while maintaining coherent, actionable outcomes.