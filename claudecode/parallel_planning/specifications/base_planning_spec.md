# Base Planning Specification

This document defines the fundamental structure and requirements for all parallel planning implementations. It serves as the foundation for project-specific planning specifications.

## Overview

The base planning specification establishes:
- Core planning concepts and terminology
- Standard workflow phases and processes
- Common perspective types and their roles
- Integration requirements with parallel agent systems
- Quality assurance and validation standards

## Core Concepts

### Planning Workflow

The parallel planning workflow consists of five main phases:

1. **Problem Analysis Phase**
   - Problem decomposition and analysis
   - Requirement gathering and constraint identification
   - Stakeholder analysis and success criteria definition

2. **Parallel Planning Phase** 
   - Multiple perspectives plan simultaneously
   - Each perspective focuses on their domain expertise
   - Independent analysis from specialized viewpoints

3. **Synthesis Phase**
   - Combine perspective plans into unified approach
   - Resolve conflicts and optimize trade-offs
   - Create coherent implementation strategy

4. **Validation Phase**
   - Validate plan completeness and consistency
   - Check feasibility and resource requirements
   - Ensure all perspectives are properly integrated

5. **Review and Approval Phase**
   - Present unified plan to stakeholders
   - Collect feedback and iterate if needed
   - Obtain approval for implementation

### Planning Perspectives

All planning implementations must support these core perspective types:

#### Technical Perspective
- **Agent Type**: `code-architect`
- **Focus Areas**: Architecture, technology stack, implementation approach
- **Key Responsibilities**:
  - System architecture design
  - Technology selection and rationale
  - Implementation methodology
  - Technical risk assessment
  - Development approach and standards

#### Security Perspective  
- **Agent Type**: `code-reviewer`
- **Focus Areas**: Security controls, compliance, threat modeling
- **Key Responsibilities**:
  - Threat modeling and risk assessment
  - Security control design
  - Compliance requirement analysis
  - Data protection strategies
  - Security testing approaches

#### Performance Perspective
- **Agent Type**: `general-purpose`
- **Focus Areas**: Scalability, performance optimization, resource management
- **Key Responsibilities**:
  - Performance requirement analysis
  - Scalability planning
  - Resource optimization strategies
  - Performance testing design
  - Capacity planning

#### Operational Perspective
- **Agent Type**: `code-architect` 
- **Focus Areas**: Deployment, monitoring, maintenance, DevOps
- **Key Responsibilities**:
  - Deployment strategy design
  - Monitoring and alerting setup
  - Backup and disaster recovery
  - Maintenance procedures
  - DevOps pipeline design

#### User Experience Perspective
- **Agent Type**: `general-purpose`
- **Focus Areas**: Usability, accessibility, user interface design
- **Key Responsibilities**:
  - User journey analysis
  - Interface design principles
  - Accessibility requirements
  - User testing strategies
  - Experience optimization

#### Business Perspective
- **Agent Type**: `general-purpose`
- **Focus Areas**: Requirements, timeline, budget, stakeholder concerns
- **Key Responsibilities**:
  - Business requirement analysis
  - Timeline and milestone planning
  - Resource and budget estimation
  - Stakeholder communication
  - Success metrics definition

### Planning Context Structure

All planning sessions must include:

```json
{
  "project_type": "string",
  "problem_description": "string", 
  "requirements": {
    "functional": [],
    "non_functional": [],
    "business": []
  },
  "constraints": {
    "technical": [],
    "business": [],
    "regulatory": [],
    "timeline": [],
    "budget": []
  },
  "stakeholders": [],
  "success_criteria": [],
  "timeline": "string",
  "budget": "string"
}
```

## Planning Outputs

### Perspective Plan Structure

Each perspective must produce a plan with:

```json
{
  "perspective_id": "string",
  "perspective_name": "string", 
  "summary": "string",
  "key_decisions": [],
  "implementation_steps": [
    {
      "step_id": "string",
      "description": "string",
      "dependencies": [],
      "estimated_effort": "string",
      "resources_required": {},
      "deliverables": [],
      "success_criteria": []
    }
  ],
  "risks": [
    {
      "risk_id": "string",
      "description": "string",
      "probability": "low|medium|high",
      "impact": "low|medium|high", 
      "mitigation_strategies": []
    }
  ],
  "assumptions": [],
  "dependencies": [],
  "resources_required": {
    "personnel": {},
    "technology": {},
    "infrastructure": {},
    "budget": {}
  },
  "timeline_estimate": "string",
  "confidence_level": 0.0
}
```

### Unified Plan Structure

The synthesized plan must include:

```json
{
  "project_summary": "string",
  "implementation_approach": "string",
  "architecture_overview": "string",
  "development_phases": [
    {
      "phase_id": "string",
      "name": "string",
      "description": "string", 
      "duration": "string",
      "deliverables": [],
      "success_criteria": [],
      "dependencies": [],
      "resources": {}
    }
  ],
  "risk_assessment": {
    "overall_risk_level": "low|medium|high",
    "key_risks": [],
    "mitigation_strategies": [],
    "contingency_plans": []
  },
  "resource_requirements": {
    "total_effort": "string",
    "team_composition": {},
    "technology_stack": [],
    "infrastructure_needs": {},
    "budget_estimate": {}
  },
  "timeline": {
    "total_duration": "string",
    "key_milestones": [],
    "critical_path": [],
    "buffer_time": "string"
  },
  "quality_assurance": {
    "testing_strategy": {},
    "quality_gates": [],
    "acceptance_criteria": [],
    "review_processes": []
  },
  "deployment_strategy": {
    "deployment_approach": "string",
    "environments": [],
    "rollout_plan": {},
    "rollback_procedures": []
  },
  "monitoring_strategy": {
    "success_metrics": [],
    "monitoring_tools": [],
    "alerting_rules": [],
    "reporting_requirements": []
  },
  "success_criteria": [],
  "assumptions": [],
  "constraints": [],
  "next_steps": []
}
```

## Quality Standards

### Completeness Requirements

All plans must address:
- ✅ All perspective areas relevant to the project type
- ✅ Complete implementation approach
- ✅ Risk assessment and mitigation strategies
- ✅ Resource requirements and timeline
- ✅ Quality assurance and testing approach
- ✅ Deployment and operational considerations

### Consistency Requirements

Plans must demonstrate:
- ✅ Consistent terminology and concepts across perspectives
- ✅ Aligned timelines and dependencies
- ✅ Compatible technology and architecture choices
- ✅ Coherent resource allocation
- ✅ Integrated risk management approach

### Feasibility Requirements

Plans must be:
- ✅ Technically feasible with proposed approach
- ✅ Achievable within timeline and resource constraints
- ✅ Compliant with identified constraints
- ✅ Aligned with stakeholder capabilities
- ✅ Realistic in scope and complexity

## Integration Requirements

### Agent Integration

The planning system must integrate with:
- **Claude Code Task Tool** - For executing individual planning agents
- **Parallel Agent Launcher** - For coordinating concurrent planning
- **Result Processing** - For synthesizing and validating plans

### Configuration Integration

Support for:
- **Project Templates** - Predefined configurations for common project types
- **Perspective Templates** - Reusable perspective configurations
- **Synthesis Strategies** - Configurable approaches to plan synthesis
- **Validation Rules** - Customizable validation criteria

### Output Integration

Generation of:
- **Structured Data** - JSON format for programmatic processing
- **Documentation** - Human-readable plans and specifications
- **Project Files** - Integration with project management tools
- **Reports** - Executive summaries and detailed analysis

## Validation Framework

### Automated Validation

- **Structure Validation** - Ensure all required fields are present
- **Dependency Validation** - Check for circular dependencies
- **Consistency Validation** - Identify conflicting recommendations
- **Completeness Validation** - Verify all perspectives are addressed

### Manual Validation

- **Feasibility Review** - Expert review of technical feasibility
- **Stakeholder Review** - Validation with project stakeholders
- **Risk Assessment** - Independent risk analysis
- **Quality Review** - Overall quality and coherence assessment

## Extensibility Guidelines

### Adding New Perspectives

1. Define perspective configuration with required fields
2. Create agent prompts specific to the perspective domain
3. Implement result parsing for perspective outputs
4. Add perspective to relevant project type templates
5. Include perspective in synthesis and validation logic

### Adding New Project Types

1. Analyze project-specific planning requirements
2. Identify relevant perspectives for the project type
3. Create project-specific templates and configurations
4. Define project-specific validation criteria
5. Document project-specific best practices

### Customizing Synthesis Strategies

1. Define synthesis approach and objectives
2. Implement conflict resolution strategies
3. Create synthesis templates and prompts
4. Add validation rules for synthesized plans
5. Test with representative planning scenarios

## Best Practices

### Perspective Design

- **Clear Scope** - Each perspective should have well-defined responsibilities
- **Minimal Overlap** - Reduce redundancy between perspectives
- **Comprehensive Coverage** - Ensure all project aspects are covered
- **Appropriate Expertise** - Match agent types to perspective requirements

### Prompt Engineering

- **Specific Instructions** - Provide clear, actionable guidance
- **Context Awareness** - Include relevant project context
- **Output Structure** - Specify required output format
- **Quality Criteria** - Define expectations for plan quality

### Result Processing

- **Structured Parsing** - Extract key information systematically
- **Error Handling** - Handle incomplete or invalid responses
- **Quality Assessment** - Evaluate plan quality and completeness
- **Conflict Detection** - Identify and flag conflicting recommendations

### Documentation

- **Clear Specifications** - Document all planning requirements
- **Usage Examples** - Provide concrete usage examples
- **Configuration Guides** - Explain configuration options
- **Troubleshooting** - Address common issues and solutions

This base specification provides the foundation for implementing consistent, high-quality parallel planning across different project types and domains.