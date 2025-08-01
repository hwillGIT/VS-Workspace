# Smart Iterative Team Rituals

> **Transform daily practices into cultural habits**

## 🌅 Daily Standup 2.0 Template

### Format (15 minutes max)
```
Traditional: Yesterday / Today / Blockers
Smart Iterative: Commits / Learnings / Next Micro-Step
```

### Example Round-Robin

**Developer A**: 
> "Yesterday: 7 commits on user auth. Learned our session timeout was too aggressive - users complained. Already pushed fix with 5-minute extension. Next: Adding remember-me option, starting with test cases."

**Developer B**:
> "Yesterday: 3 commits, had to rollback one - cookie parser couldn't handle Safari. The rollback saved us from blocking iOS users. Found the issue, committed fix with Safari-specific test. Next: Implement CSRF token rotation."

**Developer C**:
> "Yesterday: 12 small commits refactoring payment flow. Each step kept tests green. Discovered we can extract a reusable transaction state machine. Next: Document the pattern, then apply to subscription flow."

### Facilitator Prompts
- "Any interesting rollbacks we can learn from?"
- "Who discovered a gotcha worth documenting?"
- "Any patterns emerging from your micro-iterations?"

## 🎯 Sprint Planning with Micro-Iterations

### Traditional Story Breakdown
```
Story: Implement Password Reset
Tasks:
- Backend API
- Email template  
- Frontend form
- Testing
```

### Smart Iterative Breakdown
```
Story: Implement Password Reset
Micro-Iterations:
1. Test: Password reset token generation (30 min)
2. Implement: Token generation with tests passing (45 min)
3. Test: Token expiration logic (20 min)
4. Implement: Expiration with edge cases (30 min)
5. Commit checkpoint: Basic token system working
6. Test: Email sending interface (20 min)
7. Implement: Email adapter pattern (30 min)
8. Test: Rate limiting for reset requests (30 min)
9. Implement: Rate limiter with Redis (45 min)
10. Integration test: Full flow (45 min)
... (continue until complete)
```

### Estimation in Micro-Steps
- Each micro-iteration: 15-45 minutes
- Commit after each green test
- Natural rollback points between iterations

## 🔄 Weekly Retrospective Templates

### The Smart Iterative Retro Format

#### 1. Celebration Round (10 min)
**Best Save of the Week**
```markdown
Nominee: Sarah's payment gateway rollback
What happened: New validation broke Stripe webhooks
Time to detect: 2 minutes (monitoring alert)
Time to fix: 3 minutes (git revert)
What we learned: Always test with real webhook payloads
Prize: "Fast Fingers" trophy for the week
```

**Smallest Valuable Commit**
```markdown
Winner: Tom's 1-line fix
Change: Added `.trim()` to username input
Impact: Eliminated 50% of login failures
Learning: Sometimes the smallest changes have huge impact
```

#### 2. Learning Harvest (15 min)
```markdown
## This Week's Gotchas
1. **JavaScript Date parsing** - Different browsers, different results
   - Solution: Always use date-fns or moment
   - Documented in: /docs/gotchas/date-handling.md

2. **Redis connection pooling** - Default pool too small for load
   - Solution: Set minimum pool size to 10
   - Pattern added to: /docs/patterns/redis-setup.md

3. **React useEffect cleanup** - Memory leaks in async calls
   - Solution: AbortController pattern
   - Test added to: /tests/helpers/react-testing.ts
```

#### 3. Process Improvement (10 min)
```markdown
## What's Working
- Pre-commit hooks caught 15 issues before PR
- Average commit size down to 50 lines (from 200)
- Rollback confidence up - 3 rollbacks, 0 incidents

## What Needs Tuning
- Test suite taking too long (5 min) - need parallel runs
- Commit messages inconsistent - need better templates
- Documentation lag - stories closed before docs updated
```

#### 4. Commitment Round (5 min)
Each person commits to one micro-improvement:
- "I'll add commit message templates to my git config"
- "I'll document my debugging process for the OAuth issue"
- "I'll write tests FIRST for my next feature"

## 📊 Monthly Metrics Review

### Dashboard Template
```markdown
# Smart Iterative Metrics - January 2024

## Velocity Metrics
- Commits per dev per day: 8.5 (↑ from 3.2)
- Average commit size: 45 lines (↓ from 150)
- Time between commits: 52 minutes (↓ from 4 hours)

## Quality Metrics  
- Tests written before code: 78% (↑ from 40%)
- Rollbacks needed: 12 (↓ from 25)
- Rollbacks causing incidents: 0 (↓ from 3)

## Learning Metrics
- Gotchas documented: 23
- Patterns extracted: 8
- Decisions recorded: 15

## Team Health
- Psychological safety score: 8.5/10
- "Fear of breaking things": 2/10 (↓ from 7/10)
- "Confidence in rollback": 9/10 (↑ from 4/10)
```

### Story Wall Visualization
```
┌─────────────────────────────────────────────────┐
│                  STORY WALL                     │
├─────────────┬─────────────┬────────────────────┤
│   PLANNED   │ IN PROGRESS │     COMPLETED      │
├─────────────┼─────────────┼────────────────────┤
│ □□□□□□□□□□ │ ■■■□□□□□□□ │ ■■■■■■■■■■        │
│ Payment API │ User Auth   │ Email Service      │
│ (10 micro)  │ (3/10 done) │ (12 micro-commits) │
├─────────────┼─────────────┼────────────────────┤
│             │ ■■■■■■□□□□ │ ■■■■■■■■■■        │
│             │ Search Fix  │ Cache Layer        │
│             │ (6/10 done) │ (10 micro-commits) │
└─────────────┴─────────────┴────────────────────┘

■ = Completed micro-iteration with passing tests
□ = Planned micro-iteration
```

## 🎪 Special Rituals

### The Rollback Party (Monthly)
**Purpose**: Remove stigma from rollbacks, celebrate fast recovery

```markdown
## Rollback Party Agenda

1. **Rollback Stories** (20 min)
   - Each rollback presenter gets 3 minutes
   - Focus on: Detection, Decision, Recovery, Learning
   - No blame, only learnings

2. **Pattern Mining** (15 min)
   - What types of changes get rolled back most?
   - Can we test for these patterns?
   - Should we add new pre-commit checks?

3. **Rollback Drills** (10 min)
   - Practice scenario: "Production is down!"
   - Time the team: Identify → Rollback → Verify
   - Goal: Under 5 minutes total

4. **Awards** (5 min)
   - Fastest Rollback Response
   - Most Valuable Learning
   - Best Rollback Documentation
```

### Documentation Sprint (Quarterly)
**Purpose**: Keep knowledge fresh and discoverable

```markdown
## Documentation Sprint Format

Morning Session (2 hours):
1. **Gotcha Gathering**
   - Everyone writes 3 gotchas from last quarter
   - Group into themes
   - Assign owners to document

2. **Pattern Extraction**
   - Review commit history for repeated solutions
   - Document as reusable patterns
   - Create code snippets/templates

Afternoon Session (2 hours):
1. **Decision Documentation**
   - Review architectural changes
   - Write/update ADRs
   - Link to relevant commits

2. **Knowledge Graph Update**
   - Add new relationships
   - Prune outdated information
   - Test search functionality
```

### Micro-Iteration Mob Programming
**Purpose**: Teach smart iterative habits through practice

```markdown
## Mob Programming Session

Setup:
- 1 driver, 3-4 navigators
- Rotate every 15 minutes
- Focus on micro-iterations

Rules:
1. Write the test first (mob decides test name)
2. Driver implements minimal code to pass
3. Commit immediately when green
4. Refactor only if all agree
5. Commit after refactor

Debrief Questions:
- How small can we make iterations?
- What was our commit frequency?
- Did we ever go backward? Why?
- What patterns emerged?
```

## 📝 Templates and Checklists

### Micro-Iteration Planning Template
```markdown
## Feature: [Name]
### Current State
- Tests: [passing/failing]
- Coverage: [%]
- Last Commit: [sha]

### Next Micro-Step
1. Objective: [What will change]
2. Test First: [Test to write]
3. Success Criteria: [How we know it works]
4. Estimated Time: [minutes]
5. Rollback Plan: [If it fails]
```

### Commit Message Templates
```bash
# .gitmessage
# <type>: <subject>
#
# <body>
#
# Learning: <what we learned>
# Rollback: <how to undo if needed>
# Related: <links to issues/docs>
```

### Daily Reflection Prompts
End each day with these questions:
1. What was my smallest useful commit today?
2. What edge case surprised me?
3. What pattern did I notice repeating?
4. What would I do differently tomorrow?

## 🚀 Getting Started Checklist

Week 1:
- [ ] Try new standup format
- [ ] Practice micro-commits on one feature
- [ ] Document one gotcha

Week 2:
- [ ] Run first rollback celebration
- [ ] Set up commit templates
- [ ] Measure commit frequency

Week 3:
- [ ] Full team on micro-iterations
- [ ] First documentation sprint
- [ ] Knowledge graph prototype

Week 4:
- [ ] Monthly metrics review
- [ ] Adjust process based on learnings
- [ ] Celebrate wins!

---

*Remember: Rituals create culture. Culture drives results. Start small, iterate often.*