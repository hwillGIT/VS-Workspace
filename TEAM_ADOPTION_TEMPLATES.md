# Team Adoption Templates: Ready-to-Use Smart Iterative Architecture

> **"From theory to practice in minutes—templates that make architectural thinking natural."**

## 🚀 Immediate Start Kit

### Individual Developer Starter Pack

#### Daily Architectural Context Card
```markdown
# My Daily Architecture Context

## Morning Setup (2 minutes)
**Today's Focus**: [Feature/Bug/Refactor]
**Architectural Layer**: [Business/Application/Data/Infrastructure]
**Relevant Patterns**: [DDD/Microservices/Event-Driven/Layered/etc]
**Quality Attributes**: [Performance/Security/Maintainability/Scalability]

## Key Questions
- What architectural boundaries am I working near?
- What established patterns should I follow?
- What could go wrong architecturally?
- How will I validate my approach?

## Persona Check
My primary persona today: [Risk Mitigator/Complexity Orchestrator/Multi-Perspective Communicator/Continuous Steward]
```

#### Smart Iterative Decision Template
```markdown
# Architecture Decision: [Brief Title]

## Context (1 minute)
**Problem**: What architectural challenge am I solving?
**Constraints**: Time, team skills, existing patterns, quality requirements

## Options (3 minutes)
### Option 1: [Minimal Change]
- **Approach**: Extend existing pattern
- **Pros**: Low risk, familiar to team, quick implementation
- **Cons**: May not address long-term needs
- **Effort**: [Low/Medium/High]

### Option 2: [Strategic Investment]  
- **Approach**: Introduce new architectural capability
- **Pros**: Better long-term solution, addresses root cause
- **Cons**: Higher complexity, learning curve
- **Effort**: [Low/Medium/High]

### Option 3: [Hybrid Solution]
- **Approach**: Combine elements thoughtfully
- **Pros**: Balanced approach, gradual improvement
- **Cons**: Potential complexity, need careful integration
- **Effort**: [Low/Medium/High]

## Decision (1 minute)
**Chosen**: Option X
**Rationale**: [Why this choice given current constraints]
**Validation Plan**: [How I'll know if this was right choice]
**Rollback Strategy**: [What I'll do if this doesn't work]
```

#### Git Commit Template
```bash
# ~/.gitmessage
# [type]: [subject] [ARCH-CONTEXT]
#
# ARCHITECTURAL CONTEXT:
# Level: [Strategic/Tactical/Solution/Implementation]
# Domain: [Business/Information/Application/Technology]
# Pattern: [Pattern being followed]
# Quality Focus: [Performance/Security/Maintainability/Scalability]
#
# OPTIONS CONSIDERED:
# 1. [Option 1] - [brief pros/cons]
# 2. [Option 2] - [brief pros/cons]  
# 3. [Option 3] - [brief pros/cons]
# Chosen: [Option X] because [rationale]
#
# ARCHITECTURAL IMPACT:
# ✓ [Positive impacts]
# ⚠ [Risks or debt created]
# 📋 [Follow-up needed]
#
# LEARNING CAPTURED:
# - [What worked well]
# - [What was challenging]
# - [What to consider next time]
```

### Team Leader Starter Pack

#### Team Architecture Health Dashboard Template
```yaml
# team-architecture-dashboard.yml
# Update weekly during team standup

Team: [Team Name]
Sprint: [Current Sprint]
Date: [YYYY-MM-DD]

Daily_Metrics:
  features_with_arch_context: "X/Y features (Z%)"
  commits_with_arch_info: "X/Y commits (Z%)"
  options_generated_avg: "X.Y per decision"
  arch_standup_participation: "X/Y members (Z%)"

Weekly_Trends:
  pattern_consistency_score: "Z% (↗↘→)"
  architectural_debt_items: "X items (↗↘→)"
  cross_cutting_concerns_addressed: "X/Y changes (Z%)"
  stakeholder_comprehension_score: "X/10 (↗↘→)"

Team_Health:
  architectural_confidence: "X/10"
  framework_adoption_satisfaction: "X/10"
  architectural_knowledge_sharing: "X/10"
  framework_customization_needs: "[List any needed adaptations]"

Actions_This_Week:
  - "[Action item 1]"
  - "[Action item 2]"
  - "[Action item 3]"
```

#### Daily Architecture Standup Template
```markdown
# Daily Architecture Standup (15 minutes)

## Individual Check-ins (2 min per person)
**[Name]**:
- **Yesterday's Arch Impact**: [What architectural learning occurred?]
- **Today's Arch Context**: [What architectural layer/pattern focus?]
- **Arch Impediments**: [Any architectural blockers or concerns?]

## Team Architectural Pulse (3 min)
- **Pattern Consistency**: Any deviations noticed?
- **Emerging Patterns**: New approaches worth discussing?
- **Cross-Cutting Concerns**: Security/performance/scalability issues?
- **Knowledge Gaps**: What architectural knowledge do we need?

## Architecture Experiments (2 min)
- **Today's Validations**: What arch assumptions can we test?
- **Quick Options**: Any decisions needing option generation?
- **Learning Opportunities**: What can we document/share?
```

#### Weekly Architecture Evolution Session Template
```markdown
# Weekly Architecture Evolution Session (1 hour)

## Architecture Health Review (15 min)
### Metrics Review
- Review team architecture dashboard
- Identify positive and concerning trends
- Celebrate architectural wins

### Pattern Effectiveness Assessment  
- Which patterns served us well this week?
- Which patterns caused friction?
- Any anti-patterns emerging?

## Decisions and Options Review (20 min)
### Decision Quality Assessment
- Review week's architectural decisions
- Assess quality of options generated
- Document decision rationale completeness

### Options Generation Improvement
- Did we consider enough alternatives?
- Were trade-offs clearly understood?
- How can we improve option quality?

## Knowledge Capture and Sharing (15 min)
### Pattern Documentation
- New patterns that emerged this week
- Existing patterns that need updating
- Anti-patterns to avoid

### Learning Documentation
- Architectural insights discovered
- Stakeholder feedback received
- Tool and process improvements identified

## Next Week Planning (10 min)
### Architectural Focus Areas
- Key architectural contexts for next week
- Planned architectural validation experiments
- Cross-cutting concerns to emphasize

### Team Development
- Architectural skills to develop
- Knowledge sharing sessions needed
- Framework customizations to consider
```

### Project Manager/Scrum Master Starter Pack

#### Sprint Planning Architecture Integration Template
```markdown
# Sprint Planning: Architecture Integration

## Pre-Planning Architecture Assessment (15 min)
### Architectural Context Review
- What architectural layer(s) will this sprint focus on?
- Which quality attributes are most important?
- What architectural patterns are relevant?
- Are there architectural dependencies with other teams?

### Architectural Risk Assessment
- What could go wrong architecturally this sprint?
- Which stories have high architectural impact?
- What architectural assumptions need validation?
- Are there architectural knowledge gaps in the team?

## Story Architectural Context (5 min per story)
### For Each User Story:
**Story**: [Story title and description]
**Architectural Impact**: [High/Medium/Low]
**Patterns Involved**: [List relevant patterns]
**Quality Attributes**: [Performance/Security/Maintainability/etc]
**Cross-Cutting Concerns**: [Security/Logging/Monitoring/etc]
**Options Needed**: [Decisions requiring option generation]
**Dependencies**: [Other teams/systems affected]

### Architectural Story Sizing
- Account for architectural context setting time
- Include option generation and evaluation time  
- Plan for architectural documentation updates
- Consider architectural validation experiments

## Sprint Architecture Goals
### Architecture-Specific Objectives
- [ ] [Architectural pattern to implement/improve]
- [ ] [Quality attribute to optimize]
- [ ] [Cross-cutting concern to address]
- [ ] [Architectural debt to resolve]

### Knowledge and Skill Development
- [ ] [Architectural knowledge to acquire]
- [ ] [Pattern to learn or improve]
- [ ] [Tool or technique to adopt]
- [ ] [Cross-team architectural coordination needed]
```

#### Retrospective Architecture Template
```markdown
# Sprint Retrospective: Architecture Focus

## Architecture What Went Well (10 min)
### Positive Architectural Outcomes
- Which architectural decisions paid off?
- What patterns worked especially well?
- How did architectural context help the team?
- What architectural knowledge did we gain?

### Process Successes
- How well did architectural planning work?
- Were architectural impediments resolved quickly?
- Did architectural documentation stay current?
- How effective was architectural communication?

## Architecture What Didn't Go Well (10 min)
### Architectural Challenges
- Which architectural decisions caused problems?
- What patterns created friction?
- Where did we lack architectural context?
- What architectural debt did we accumulate?

### Process Issues
- Where did architectural planning fall short?
- What architectural impediments slowed us down?
- How did architectural documentation lag?
- Where was architectural communication unclear?

## Architecture Improvements (10 min)
### Process Improvements
- How can we better integrate architecture into planning?
- What architectural tools or techniques should we try?
- How can we improve architectural decision quality?
- What architectural knowledge do we need to develop?

### Framework Adaptations
- What aspects of Smart Iterative Architecture need customization?
- How can we reduce friction in architectural practices?
- What additional architectural support do we need?
- How can we improve architectural measurement?

## Architecture Action Items (5 min)
### Next Sprint Commitments
- [ ] [Architectural process improvement to try]
- [ ] [Architectural knowledge to acquire]
- [ ] [Architectural tool or technique to implement]
- [ ] [Architectural debt to address]
- [ ] [Cross-team architectural coordination to improve]
```

## 🛠️ Tool Configuration Templates

### Git Hook Templates

#### Pre-commit Hook (Architectural Context Validation)
```bash
#!/bin/bash
# .git/hooks/pre-commit
# Smart Iterative Architecture Pre-commit Hook

echo "🏗️  Smart Iterative Architecture: Validating architectural context..."

# Check if commit message includes architectural context
if git diff --cached --name-only | grep -E '\.(js|ts|py|java|cs|cpp|go|rs)$' > /dev/null; then
    if ! grep -q "\[ARCH-CONTEXT\]" .git/COMMIT_EDITMSG 2>/dev/null; then
        echo "❌ Architectural context missing from commit message"
        echo "💡 Please include [ARCH-CONTEXT] and architectural information"
        echo "   Use: git commit --template ~/.gitmessage"
        exit 1
    fi
fi

# Check for architectural boundary violations (customizable patterns)
echo "🔍 Checking architectural boundary integrity..."

# Example: Prevent direct database access from presentation layer
if git diff --cached --name-only | grep -E 'controllers?|views?|components?' | xargs git diff --cached | grep -E '(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)' > /dev/null; then
    echo "⚠️  Potential architectural boundary violation detected"
    echo "💡 Database queries found in presentation layer"
    echo "   Consider using service/repository layer instead"
    exit 1
fi

echo "✅ Architectural context validation passed"
```

#### Post-commit Hook (Learning Capture)
```bash
#!/bin/bash
# .git/hooks/post-commit
# Smart Iterative Architecture Post-commit Learning Capture

# Extract architectural learning from commit message
if git log -1 --pretty=%B | grep -q "\[ARCH-LEARNING\]"; then
    echo "🧠 Capturing architectural learning..."
    
    # Extract learning content
    LEARNING=$(git log -1 --pretty=%B | sed -n '/LEARNING CAPTURED:/,/^$/p' | tail -n +2)
    
    if [ ! -z "$LEARNING" ]; then
        # Append to team learning log
        echo "## $(date '+%Y-%m-%d %H:%M') - $(git log -1 --pretty=%an)" >> architectural-learning.md
        echo "$LEARNING" >> architectural-learning.md
        echo "" >> architectural-learning.md
        
        echo "✅ Architectural learning captured in architectural-learning.md"
    fi
fi

# Update architectural metrics
echo "📊 Updating architectural metrics..."
python .architectural-tools/update-metrics.py

echo "🎯 Architectural context processing complete"
```

### IDE Integration Templates

#### VS Code Settings for Smart Iterative Architecture
```json
{
  "git.template": "~/.gitmessage",
  "git.enableCommitSigning": true,
  "editor.rulers": [72, 100],
  "files.associations": {
    "*.architectural": "yaml",
    "architectural-*.md": "markdown"
  },
  "markdown.extension.toc.levels": "2..6",
  "todo-tree.general.tags": [
    "ARCH-TODO",
    "ARCH-DEBT", 
    "ARCH-REVIEW",
    "ARCH-VALIDATE"
  ],
  "todo-tree.highlights.customHighlight": {
    "ARCH-TODO": {
      "icon": "tools",
      "foreground": "#FFD700"
    },
    "ARCH-DEBT": {
      "icon": "alert",
      "foreground": "#FF6B6B"
    }
  }
}
```

#### VS Code Snippets for Architectural Context
```json
{
  "Architectural Decision Template": {
    "prefix": "arch-decision",
    "body": [
      "# Architecture Decision: ${1:Brief Title}",
      "",
      "## Context",
      "**Problem**: ${2:What architectural challenge?}",
      "**Constraints**: ${3:Time, skills, patterns, quality requirements}",
      "",
      "## Options",
      "### Option 1: ${4:Minimal Change}",
      "- **Approach**: ${5:Extend existing pattern}",
      "- **Pros**: ${6:Low risk, familiar}",
      "- **Cons**: ${7:May not address long-term}",
      "",
      "### Option 2: ${8:Strategic Investment}",
      "- **Approach**: ${9:New architectural capability}",
      "- **Pros**: ${10:Better long-term solution}",
      "- **Cons**: ${11:Higher complexity}",
      "",
      "## Decision",
      "**Chosen**: ${12:Option X}",
      "**Rationale**: ${13:Why this choice}",
      "**Validation**: ${14:How to validate}",
      "$0"
    ],
    "description": "Template for architectural decisions"
  },
  
  "Architectural Context Comment": {
    "prefix": "arch-context",
    "body": [
      "// ARCHITECTURAL CONTEXT:",
      "// Layer: ${1:Business/Application/Data/Infrastructure}",
      "// Pattern: ${2:DDD/Microservices/Event-Driven/Layered}",
      "// Quality Focus: ${3:Performance/Security/Maintainability}",
      "// Rationale: ${4:Why this approach}",
      "$0"
    ],
    "description": "Add architectural context to code"
  }
}
```

### Team Dashboard Templates

#### Simple HTML Dashboard Template
```html
<!DOCTYPE html>
<html>
<head>
    <title>Smart Iterative Architecture Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .good { background-color: #d4edda; }
        .warning { background-color: #fff3cd; }
        .danger { background-color: #f8d7da; }
        .trend-up { color: #28a745; }
        .trend-down { color: #dc3545; }
        .trend-stable { color: #6c757d; }
    </style>
</head>
<body>
    <h1>Team Architecture Health Dashboard</h1>
    <p>Last Updated: <span id="lastUpdated">Loading...</span></p>
    
    <h2>Daily Metrics</h2>
    <div class="metric good">
        <strong>Architectural Context</strong><br>
        <span id="archContext">Loading...</span>% of features
    </div>
    <div class="metric good">
        <strong>Options Generated</strong><br>
        <span id="optionsAvg">Loading...</span> per decision
    </div>
    <div class="metric warning">
        <strong>Commit Context</strong><br>
        <span id="commitContext">Loading...</span>% of commits
    </div>
    
    <h2>Weekly Trends</h2>
    <div class="metric good">
        <strong>Pattern Consistency</strong><br>
        <span id="patternScore">Loading...</span>% <span id="patternTrend" class="trend-up">↗</span>
    </div>
    <div class="metric warning">
        <strong>Architectural Debt</strong><br>
        <span id="debtItems">Loading...</span> items <span id="debtTrend" class="trend-stable">→</span>
    </div>
    
    <h2>Team Health</h2>
    <div class="metric good">
        <strong>Framework Satisfaction</strong><br>
        <span id="satisfaction">Loading...</span>/10
    </div>
    
    <script>
        // Load metrics from JSON file or API
        fetch('architectural-metrics.json')
            .then(response => response.json())
            .then(data => {
                document.getElementById('lastUpdated').textContent = data.lastUpdated;
                document.getElementById('archContext').textContent = data.dailyMetrics.architecturalContext;
                document.getElementById('optionsAvg').textContent = data.dailyMetrics.optionsAverage;
                document.getElementById('commitContext').textContent = data.dailyMetrics.commitContext;
                document.getElementById('patternScore').textContent = data.weeklyTrends.patternConsistency;
                document.getElementById('debtItems').textContent = data.weeklyTrends.architecturalDebt;
                document.getElementById('satisfaction').textContent = data.teamHealth.frameworkSatisfaction;
            })
            .catch(error => console.error('Error loading metrics:', error));
    </script>
</body>
</html>
```

#### Metrics Data Template
```json
{
  "lastUpdated": "2025-08-01T10:30:00Z",
  "team": "Backend Services Team",
  "sprint": "Sprint 23",
  
  "dailyMetrics": {
    "architecturalContext": 87,
    "optionsAverage": 2.8,
    "commitContext": 73,
    "standupParticipation": 95
  },
  
  "weeklyTrends": {
    "patternConsistency": 92,
    "patternTrend": "up",
    "architecturalDebt": 8,
    "debtTrend": "down",
    "crossCuttingConcerns": 89,
    "stakeholderComprehension": 7.5
  },
  
  "teamHealth": {
    "architecturalConfidence": 8.2,
    "frameworkSatisfaction": 8.7,
    "knowledgeSharing": 7.9,
    "customizationNeeds": ["Lighter documentation", "Better tool integration"]
  },
  
  "actionsThisWeek": [
    "Implement automated architectural debt detection",
    "Schedule architectural pattern workshop",
    "Improve commit message template adoption"
  ]
}
```

## 🎓 Training Session Templates

### 1-Hour Framework Introduction Workshop
```markdown
# Smart Iterative Architecture Introduction (60 minutes)

## Opening & Context (10 minutes)
### Ice Breaker Question
"Think of a time when poor software architecture slowed you down. What happened?"

### Workshop Objectives
By the end of this session, you will:
- Understand Smart Iterative Architecture core principles
- Identify your primary architectural persona
- Practice the basic Smart Iterative cycle
- Leave with templates for immediate use

## Core Concepts (20 minutes)
### The Nervous System Metaphor (5 min)
- Architecture as system coordination mechanism
- Micro-evolutions vs big-bang changes
- Cruft as system dysfunction

### Four Architectural Personas (10 min)
**Exercise**: Read persona descriptions, identify your primary
- Risk Mitigator: "What could go wrong?"
- Complexity Orchestrator: "What are the building blocks?"  
- Multi-Perspective Communicator: "Who needs to understand this?"
- Continuous Steward: "How does this fit our evolution?"

### Options-First Decision Making (5 min)
**Principle**: Always generate 2-3 alternatives
**Exercise**: Quick practice with current team challenge

## Hands-On Practice (20 minutes)
### Smart Iterative Cycle Walkthrough (10 min)
Using a real current feature:
1. Architectural Context Setting (2 min)
2. Options Generation (3 min)
3. Decision with Rationale (2 min)
4. Implementation Planning (3 min)

### Commit Message Practice (10 min)
Transform a recent commit message to include architectural context
**Before/After comparison and feedback**

## Tool Setup & Next Steps (10 minutes)
### Immediate Setup
- Git commit template installation
- VS Code snippets (if applicable)
- Dashboard bookmark

### This Week Commitments
Each person commits to:
- [ ] Use architectural context for next 3 features
- [ ] Practice options generation for next significant decision
- [ ] Try new commit message template for 1 week

### Questions & Wrap-up
**Key Takeaway**: Architecture awareness in every iteration, not just big decisions
```

### Architecture Decision Workshop Template
```markdown
# Architecture Decision Workshop (90 minutes)

## Preparation (Send 24 hours ahead)
**Scenario**: [Describe realistic architectural challenge team faces]
**Background Materials**: [Current system context, constraints, stakeholders]
**Pre-work**: Each participant should come with one initial approach idea

## Workshop Flow

### Problem Exploration (15 minutes)
- **Constraint Mapping**: What are our real limitations?
- **Stakeholder Needs**: Who cares about this decision and why?
- **Quality Attributes**: What non-functional requirements matter most?
- **Success Criteria**: How will we know we made a good choice?

### Option Generation (30 minutes)
**Individual Brainstorming (10 min)**:
- Each person generates 2-3 approaches independently
- Use template: Approach / Pros / Cons / Effort / Risk

**Small Group Synthesis (15 min)**:
- Groups of 2-3 combine and refine options
- Focus on making options distinct and viable
- Identify key trade-offs

**Full Group Sharing (5 min)**:
- Each group presents their top 2 options
- Capture all options on shared surface

### Option Evaluation (25 minutes)
**Trade-off Analysis (15 min)**:
- Map each option against quality attributes
- Assess implementation effort and risk
- Consider alignment with current architecture

**Stakeholder Perspective Review (10 min)**:
- How would each stakeholder view each option?
- What would each option enable or prevent?
- Which options support future evolution?

### Decision Making (15 minutes)
**Convergence Process**:
- Eliminate clearly inferior options
- Deep dive on remaining 2-3 options
- Use decision criteria from problem exploration
- Make decision with clear rationale

**Documentation**:
- Complete decision template together
- Assign follow-up actions
- Plan validation approach

### Wrap-up (5 minutes)
**Lessons Learned**:
- What made this decision process effective?
- How can we improve architectural decision making?
- What tools or techniques should we adopt?
```

These templates provide immediate, practical tools for teams to begin implementing Smart Iterative Architecture principles from day one, with minimal setup and maximum impact.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "1", "content": "Integrate all architecture frameworks into unified Smart Iterative system", "status": "completed", "priority": "high"}, {"id": "2", "content": "Create comprehensive implementation roadmap", "status": "completed", "priority": "high"}, {"id": "3", "content": "Build unified documentation index", "status": "completed", "priority": "high"}, {"id": "4", "content": "Create team adoption templates", "status": "completed", "priority": "high"}]