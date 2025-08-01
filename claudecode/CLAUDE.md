## Gemini Core Directives
- Treat all claude.md files as if they are my own.
- Structured plan for trading platform UI: Project Type: Web Application. Problem: Design modern, attractive, functional trading platform UI prototype. Perspectives: 1. UX & Visual Design (general-purpose): UI mockups, Material Design. 2. Technical Architecture (code-architect): React/Material-UI, component structure. 3. Data Integration (general-purpose): JSON data structures, mock server. 4. Security & Compliance (code-reviewer): Input validation, data privacy, security best practices.
- The DEVELOPMENT_GUIDELINES.md file is a cornerstone of my operational framework for this project.
- When debugging, the SELF_DEBUGGER.md directives, including interaction with Anthropic, should be the first option considered.

# Claude Code Configuration

> **Note**: The authoritative character definition is now at `/CLAUDE.md` with role-specific definitions in `/roles/`

## Available Prompts

This project includes a curated collection of Claude prompts to enhance your development workflow.

**Location:** `./prompts/`

### Quick Start
- Browse available resources: `./prompts/PROMPT_INDEX.md`
- Access the awesome-claude-prompts collection: `./prompts/awesome-claude-prompts/README.md`
- Read the complete Claude Code guide: `./prompts/claude-code-guide/README.md`
- Explore community workflows and tools: `./prompts/vincenthopf-claude-code/README.md`
- Check out MCP tools and servers: `./MCP_TOOLS.md`
- **IMPORTANT**: Review security practices: `./SECURITY_PRACTICES.md`
- **NEW**: Development guidelines for humans and AI: `/DEVELOPMENT_GUIDELINES.md`

### Claude Code Workflow Shortcuts (from zebbern's guide)
- **qnew** - Tell Claude to understand all best practices from CLAUDE.md
- **qplan** - Analyze codebase consistency and determine minimal changes
- **qcode** - Implement plan with tests, formatting, and type checking
- **qcheck** - Review code changes against best practices checklists
- **qux** - Generate comprehensive UX testing scenarios
- **qgit** - Stage, commit, and push with proper conventional commit format

### Enhanced Workflow Integration
Based on the new guidelines, remember to:
1. **Always Plan First**: Use qplan before qcode
2. **Automated Testing**: Tests run automatically after code changes
3. **Context Files**: CLAUDE.md and role files define behavior
4. **Collaboration**: Work with AI as a peer - provide context, review output
5. **Security First**: Never commit secrets, use sandboxed environments

### Useful Development Resources
- **Code Analysis:** Use the "Explain Python Code" prompt for understanding complex code
- **Code Review:** Apply "Expert Editor" prompts for thorough code reviews
- **Documentation:** Leverage "Smart Dev" prompts for generating technical documentation
- **Testing:** Use specialized prompts for test case generation and analysis
- **Best Practices:** Follow the comprehensive coding standards in `/roles/claude-code.md`
- **Slash Commands:** Explore 88+ community slash commands in `./prompts/vincenthopf-claude-code/README.md`
- **Workflows:** Access ClaudeLog, Claude Task Manager, and Project Workflow Systems
- **MCP Extensions:** Install MCP servers like just-prompt for multi-LLM access and advanced workflows

## Project Context

This is a comprehensive trading system with:
- 11 core trading agents
- System architect suite for code analysis
- Comprehensive testing infrastructure
- Multi-asset strategy support

When using prompts, consider mentioning this context for more relevant responses.