# Smart Iterative Coding - Implementation Guide

## 🚀 Quick Start (Day 1)

### Morning: Environment Setup (2 hours)
```bash
# 1. Install pre-commit hooks
pip install pre-commit
pre-commit install

# 2. Configure git for micro-commits
git config --global commit.template ~/.gitmessage
echo "[LEARNING]" > ~/.gitmessage

# 3. Set up test watchers
npm install --save-dev jest-watch-typeahead
# or
pip install pytest-watch
```

### Afternoon: First Smart Iteration (2 hours)
Pick a small feature and practice:
1. Write ONE test (5 min)
2. Make it pass (10 min)
3. Commit immediately
4. Write next test
5. Repeat 5 times

Goal: 5 commits in 2 hours

## 📊 Week 1 Milestones

### Day 1-2: Individual Practice
- [ ] Each developer completes 10+ micro-commits
- [ ] Document first "gotcha" in `/docs/gotchas/`
- [ ] Experience first successful rollback

### Day 3-4: Team Coordination  
- [ ] Run first Smart Iterative standup
- [ ] Set up team dashboard
- [ ] Create first knowledge graph entry

### Day 5: Reflection & Adjustment
- [ ] Team retrospective on the process
- [ ] Identify top 3 improvements
- [ ] Celebrate smallest valuable commit

## 🛠️ Technical Setup

### 1. Pre-commit Configuration
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: Run tests
        entry: npm test
        language: system
        pass_filenames: false
        stages: [commit]
      
      - id: lint
        name: Lint files
        entry: npm run lint
        language: system
        files: \.(js|ts|jsx|tsx)$
        
      - id: types
        name: Type check
        entry: npm run typecheck
        language: system
        pass_filenames: false
```

### 2. Git Hooks for Learning Capture
```bash
#!/bin/bash
# .git/hooks/post-commit

# Check for [LEARNING] tag
if git log -1 --pretty=%B | grep -q "\[LEARNING\]"; then
    # Extract learning and add to knowledge base
    python scripts/capture_learning.py
    
    # Notify team
    echo "🧠 Learning captured! Check /docs/gotchas/ for updates."
fi
```

### 3. IDE Integration

#### VS Code Settings
```json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "jest.autoRun": {
    "watch": true,
    "onStartup": ["all-tests"]
  },
  "gitlens.currentLine.enabled": true,
  "gitlens.codeLens.enabled": true
}
```

#### VS Code Extensions
- Jest Runner
- ESLint
- GitLens
- Conventional Commits

## 🎯 Success Metrics Tracking

### Daily Metrics Script
```python
#!/usr/bin/env python3
# scripts/daily_metrics.py

import subprocess
import json
from datetime import datetime, timedelta

def get_daily_metrics():
    yesterday = datetime.now() - timedelta(days=1)
    
    # Get commit count
    commits = subprocess.check_output([
        'git', 'log', 
        '--since', yesterday.isoformat(),
        '--oneline'
    ]).decode().strip().split('\n')
    
    # Get average commit size
    sizes = []
    for commit in commits:
        sha = commit.split()[0]
        stat = subprocess.check_output([
            'git', 'show', '--stat', sha
        ]).decode()
        # Parse lines changed
        
    # Get rollback count
    rollbacks = [c for c in commits if 'revert' in c.lower()]
    
    return {
        'date': datetime.now().isoformat(),
        'commits': len(commits),
        'average_size': sum(sizes) / len(sizes) if sizes else 0,
        'rollbacks': len(rollbacks),
        'learnings': len([c for c in commits if '[LEARNING]' in c])
    }

if __name__ == '__main__':
    metrics = get_daily_metrics()
    print(json.dumps(metrics, indent=2))
```

## 🔄 Common Scenarios

### Scenario 1: Feature Too Big
**Problem**: "I can't break this down into smaller pieces!"

**Solution**:
```markdown
Big Feature: User Authentication

Break down by:
1. Test: Can create user object
2. Test: User object validates email
3. Test: User object hashes password
4. Test: Can save user to database
5. Test: Can retrieve user by email
6. Test: Password verification works
7. Test: Login returns token
8. Test: Token validates correctly
```

### Scenario 2: Rollback Needed
**Problem**: "Something broke in production!"

**Smart Iterative Response**:
```bash
# 1. Identify breaking commit (< 1 min)
git log --oneline -10

# 2. Rollback (< 30 sec)
git revert HEAD --no-edit

# 3. Push fix (< 30 sec)
git push

# 4. Document learning (< 5 min)
echo "## Rollback: Authentication Breaking

**Issue**: Regex too strict for email validation
**Impact**: 20% of users couldn't log in  
**Fix**: Reverted to previous regex
**Learning**: Always test with production data samples
**Prevention**: Added edge case tests
" >> docs/rollbacks/$(date +%Y-%m-%d).md
```

### Scenario 3: Test-First Resistance
**Problem**: "I don't know what to test yet!"

**Solution**: Write a description test
```javascript
describe('Payment Processor', () => {
  it.todo('should handle successful payment')
  it.todo('should handle declined cards')  
  it.todo('should handle network timeouts')
  it.todo('should prevent duplicate charges')
  it.todo('should log all transactions')
})
```

Then implement one at a time!

## 🏁 30-Day Roadmap

### Week 1: Foundation
- Individual habit building
- Tool setup and configuration
- First learnings captured

### Week 2: Team Practices
- Daily standups transformed
- First rollback celebration
- Knowledge graph prototype

### Week 3: Acceleration
- Metrics dashboard live
- Pattern library started
- Cross-team learning

### Week 4: Culture Lock-in
- Monthly review and awards
- Process improvements identified
- Success stories documented

## 🎉 Celebrating Success

### Quick Wins Board
```
┌─────────────────────────────────────────┐
│            QUICK WINS                   │
├─────────────────────────────────────────┤
│ 🏆 Fastest Rollback: Sarah (90 seconds) │
│ 🔬 Best Test Name: "should_cry_when_null"│
│ 📦 Smallest Fix: Tom (1 line, big impact)│
│ 🧠 Learning Champion: Alex (5 this week) │
│ 🚀 Velocity King: Jamie (45 commits)    │
└─────────────────────────────────────────┘
```

### Team Health Indicators
- ✅ Commits per day increasing
- ✅ Rollback time decreasing  
- ✅ Test coverage climbing
- ✅ Documentation growing
- ✅ Team confidence rising

## 🤝 Getting Help

### When Stuck
1. Check `/docs/patterns/` for similar solutions
2. Search knowledge graph for past learnings
3. Ask team: "Who's solved something like this?"
4. Document your solution for the next person

### Resources
- Smart Iterative Champions: Assign 2-3 team members
- Weekly Office Hours: Q&A on the process
- Slack Channel: #smart-iterative-coding
- Dashboard: http://your-team-dashboard

---

*Remember: Smart Iterative Coding is a journey, not a destination. Start small, iterate often, and let the process evolve with your team.*