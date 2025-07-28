# MCP Tools for Claude Code

Model Context Protocol (MCP) servers extend Claude Code's capabilities by providing additional tools and integrations.

## Available MCP Servers

### disler/just-prompt
**Repository:** https://github.com/disler/just-prompt  
**Purpose:** Unified interface to multiple LLM providers

**What it provides:**
- Access to OpenAI, Anthropic, Google Gemini, Groq, DeepSeek, and Ollama
- CEO & Board decision-making workflow (multi-model consensus with o3 as CEO)
- Side-by-side model comparison
- File-based prompt processing

**MCP Tools:**
- `prompt` - Send prompts to multiple LLM models
- `prompt_from_file` - Send prompts from files to multiple models
- `prompt_from_file_to_file` - Process files and save responses as markdown
- `ceo_and_board` - Multi-model decision making with CEO arbitration
- `list_providers` - List available LLM providers
- `list_models` - List models for specific providers

**Installation:**
```bash
# Clone and install
git clone https://github.com/disler/just-prompt.git
cd just-prompt
pip install -e .

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Add to Claude Code MCP configuration
# Add server configuration to your Claude Code settings
```

**Usage in Claude Code:**
Once configured, you can use commands like:
- Compare responses across multiple models
- Make decisions using the CEO & Board workflow
- Process files through multiple LLM providers
- Test prompts against different model capabilities

## How to Add MCP Servers

1. **Install the MCP server** (follow its installation instructions)
2. **Configure in Claude Code** by adding to your MCP settings
3. **Use the tools** through Claude Code's tool interface

## Other Notable MCP Servers

For more MCP servers, check:
- [Anthropic MCP Server Collection](https://github.com/modelcontextprotocol/servers)
- [Community MCP Servers](https://github.com/topics/mcp-server)

---

*Note: MCP servers are separate applications that extend Claude Code's functionality, unlike prompts which are text templates you can copy and use directly.*