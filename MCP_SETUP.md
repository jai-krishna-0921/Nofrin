# MCP Setup for Deep Research Agent

## Install all MCP servers (run once)

```bash
# 1. Context7 — live library documentation (no API key needed)
claude mcp add --transport http context7 https://mcp.context7.com/mcp

# 2. Sequential Thinking — structured reasoning for complex LangGraph wiring
claude mcp add sequential-thinking -- npx -y @modelcontextprotocol/server-sequential-thinking

# 3. Filesystem — scoped to project root only
claude mcp add filesystem -- npx -y @modelcontextprotocol/server-filesystem $(pwd)

# 4. Verify all installed
claude mcp list
```

## Alternative: claude_desktop_config.json (if using Claude Desktop)

```json
{
  "mcpServers": {
    "context7": {
      "type": "http",
      "url": "https://mcp.context7.com/mcp"
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/deep_research_agent"]
    }
  }
}
```

Config location:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

---

## How to use each MCP in Claude Code sessions

### Context7 (use for ALL library lookups)

Instead of asking "how do I use LangGraph's Send pattern?" — say:

```
use context7 to get the latest LangGraph docs on the Send pattern and deferred nodes
```

Always use context7 for:
- LangGraph: StateGraph, Send, conditional_edges, checkpointing
- Anthropic SDK: Messages API, streaming, tool_use, prompt caching
- Exa: search client, highlights, date filtering
- RAGAS: faithfulness, answer_relevancy metrics
- LangSmith: tracing, prompt hub, dataset management

### Sequential Thinking (use for complex planning)

Use when planning multi-step LangGraph wiring, async orchestration patterns,
or any change that touches more than 3 files. Say:

```
use sequential-thinking to plan how to wire the deferred coordinator node that
waits for all parallel workers before proceeding
```

### Filesystem (scoped to project)

```
use filesystem to read all files in agents/ and give me an overview
```

---

## MCP usage rules (in CLAUDE.md)

These are already in CLAUDE.md but repeated here for setup reference:
- Use context7 BEFORE answering any question about library APIs — never from training memory
- Use sequential-thinking for any change that touches more than 3 files
- Filesystem MCP is scoped to project root — do not attempt paths outside it
