# Development Guidelines for Software Developers

## Overview
This document provides comprehensive guidelines for both human developers and AI coding agents working in this codebase. These guidelines are derived from industry best practices and specifically tailored for effective collaboration between humans and AI assistants.

## Core Behaviors and Values

### 1. Plan Before You Code
**Resist the urge to jump straight into implementation.**

- Start with research and understanding the problem
- Ask questions (of teammates, AI agents, or documentation)
- Outline a plan before writing code
- Document your approach for review

**Why:** Planning reduces errors, aligns everyone on goals, and saves time in the long run.

### 2. Document and Communicate
**Keep specifications and documentation up to date.**

- Write clear, explicit documentation
- Update docs when code changes
- Make requests clear and direct
- Provide context for AI agents and team members

**Why:** Explicit documentation reduces misunderstandings and provides necessary context.

### 3. Adopt a Functional, Test-Driven Mindset
**Write small, side-effect-free functions with comprehensive tests.**

- Follow TDD: Write tests first, then implement
- Cover both happy and unhappy paths
- Use AI to generate and maintain tests
- Automate test runs in CI/CD

**Why:** Tests catch bugs early and ensure code reliability.

### 4. Embrace Consistent Style
**Follow agreed-upon coding conventions.**

- Use established style guides
- Enforce immutability where appropriate
- Use clear, descriptive naming
- Regular code reviews (human and AI)

**Why:** Consistency improves readability and reduces cognitive load.

### 5. Iterate with Feedback Loops
**Get immediate feedback on generated code.**

- Run linters and tests continuously
- Fix issues early
- Use tools like Sculptor for automated feedback
- Review AI-generated code before integration

**Why:** Early feedback prevents accumulation of technical debt.

### 6. Secure and Ethical Conduct
**Treat security as a first-class concern.**

- Never commit secrets or API keys
- Use sandboxed environments for experiments
- Follow the principle of least privilege
- Consider ethical implications of code

**Why:** Security breaches are costly and damage trust.

### 7. Collaborate with AI as a Peer
**View AI coding agents as collaborators, not tools.**

- Provide clear context and examples
- Review AI suggestions critically
- Encourage AI to propose plans
- Give feedback on AI performance

**Why:** AI agents perform better with proper context and collaboration.

## Practical Implementation

### Before Starting a Task

1. **Research First**
   - Check existing code patterns
   - Review relevant documentation
   - Understand the domain

2. **Create a Plan**
   - Break down complex tasks
   - Identify dependencies
   - Consider edge cases
   - Get plan reviewed

3. **Set Up Environment**
   - Ensure tools are configured
   - Verify test infrastructure
   - Check security settings

### During Development

1. **Follow Smart Iterative Cycle**
   ```
   1. Think (5 min) - Plan the micro-step
   2. Test (Red) - Write a failing test
   3. Code (Green) - Write minimal code to pass
   4. Refactor - Improve clarity if needed
   5. Commit - Save progress immediately
   6. Learn - Document any insights
   7. Repeat - Next micro-iteration
   ```

2. **Micro-Commit Practice**
   - Commit after EVERY passing test
   - Average commit: < 50 lines of code
   - Time between commits: < 1 hour
   - Include learning tags: `[LEARNING]` for insights
   - Clear rollback instructions in commit message

3. **Continuous Quality Feedback**
   - Pre-commit hooks run automatically
   - Tests run on file save
   - Linters active in IDE
   - Type checking in real-time
   - Performance impact warnings

4. **Capture Learnings Immediately**
   ```bash
   # When you discover something:
   git commit -m "fix: handle edge case [LEARNING]
   
   Discovered that user emails can contain '+' characters.
   Previous regex was too restrictive.
   
   Rollback: Safe to revert if email validation becomes too permissive"
   ```

### After Implementation

1. **Code Review**
   - Self-review first
   - AI review for patterns
   - Peer review for logic

2. **Documentation**
   - Update relevant docs
   - Add inline comments for complex logic
   - Update API documentation

3. **Integration**
   - Run full test suite
   - Check for regressions
   - Monitor performance

## Working with AI Agents

### Providing Context

**Good Context:**
```
"I need to implement a rate limiter for our API endpoints. 
We're using Express.js with TypeScript. The limiter should:
- Allow 100 requests per minute per IP
- Return 429 status when limit exceeded
- Store state in Redis
Here's our current middleware setup: [code example]"
```

**Poor Context:**
```
"Add rate limiting"
```

### Reviewing AI Code

1. **Verify Logic**: Does it solve the problem correctly?
2. **Check Style**: Does it follow project conventions?
3. **Test Coverage**: Are edge cases handled?
4. **Security**: Any potential vulnerabilities?
5. **Performance**: Will it scale appropriately?

### Feedback Loop

- Tell AI what worked well
- Point out issues clearly
- Suggest improvements
- Update context files (CLAUDE.md/GEMINI.md)

## Common Pitfalls to Avoid

1. **Jumping to Code**: Always plan first
2. **Ignoring Tests**: Tests are not optional
3. **Poor Communication**: Be explicit and clear
4. **Security Shortcuts**: Never compromise on security
5. **Style Inconsistency**: Follow established patterns
6. **Context Assumptions**: Provide full context to AI

## Continuous Improvement

- Regularly review and update these guidelines
- Share learnings with the team
- Update AI context files with new patterns
- Measure and improve development metrics

## Quick Reference Checklist

Before coding:
- [ ] Understood the requirements?
- [ ] Broke down into micro-iterations?
- [ ] Created a plan for first iteration?
- [ ] Set up testing infrastructure?

While coding (each micro-iteration):
- [ ] Wrote test first?
- [ ] Made minimal change to pass?
- [ ] Committed immediately when green?
- [ ] Captured any learnings?
- [ ] Rollback plan clear?

After coding:
- [ ] All tests passing?
- [ ] Code reviewed?
- [ ] Learnings documented?
- [ ] Patterns extracted?
- [ ] Knowledge graph updated?

## Related Documents

- `/CLAUDE.md` - Core AI character definition
- `/roles/` - Specialized AI role definitions
- `/ClaudeCode/SECURITY_PRACTICES.md` - Security guidelines
- `/SMART_ITERATIVE_CODING.md` - Complete Smart Iterative philosophy
- `/CODEBASE_MEMORY_GRAPH.md` - Knowledge capture system
- `/TEAM_RITUALS.md` - Team practice templates
- Project-specific style guides

---

*These guidelines are living documents. Propose updates through pull requests.*