# OpenMemory-enhanced

An enhanced fork of [CaviraOSS/OpenMemory](https://github.com/CaviraOSS/OpenMemory), purpose-built for **AI coding agents** that need persistent memory across sessions.

OpenMemory provides the engine — HSG vector search, salience decay, SimHash deduplication, temporal facts. This fork adds the features that coding agents actually need: **project scoping**, **session lifecycle**, **typed observations**, and an **agent protocol** (SKILL.md) that tells the agent exactly when and how to use memory.

## What This Fork Adds

| Feature | Upstream | This Fork |
|---|---|---|
| Project scoping | ❌ Flat namespace | ✅ `project` column — memories partitioned per-project |
| Session lifecycle | ❌ No session concept | ✅ `sessions` + `summaries` tables, `openmemory_summarize` tool |
| Observation types | ❌ Generic content | ✅ `observation_type` field (bugfix / decision / discovery / feature / gotcha / refactor) |
| Timeline queries | ❌ No chronological browsing | ✅ `openmemory_timeline` tool — get memories before/after an anchor |
| Session ID tracking | ❌ Not on memories | ✅ `session_id` on every memory — trace back to the conversation |
| Agent protocol | ❌ Generic MCP tools | ✅ SKILL.md with explicit session start/save/summarize protocol |

Everything from upstream is preserved: HSG engine, multi-provider embeddings (OpenAI / Gemini / Ollama / AWS / synthetic), SQLite + Postgres backends, temporal facts, waypoints, decay.

## MCP Tools

8 tools exposed via stdio MCP transport:

| Tool | Description |
|---|---|
| `openmemory_store` | Save a memory with project, session, and observation type |
| `openmemory_query` | Vector + keyword search, scoped to a project |
| `openmemory_list` | List recent memories, optionally filtered by project |
| `openmemory_get` | Fetch full details of a specific memory |
| `openmemory_delete` | Remove a memory by ID |
| `openmemory_reinforce` | Boost a memory's salience score |
| `openmemory_summarize` | Save a session summary (what was done, learned, next steps) |
| `openmemory_timeline` | Get chronological context around a specific memory |

## Setup

### Prerequisites

- Node.js 20+
- An MCP-compatible AI coding agent ([opencode](https://opencode.ai), Claude Code, etc.)

### Install

```bash
git clone https://github.com/ayixiayi/OpenMemory-enhanced.git
cd OpenMemory-enhanced/packages/openmemory-js
npm install
npm run build
```

### Register as MCP Server

For **opencode**, add to `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "opencode-mem": {
      "type": "local",
      "command": ["node", "/path/to/OpenMemory-enhanced/packages/openmemory-js/dist/ai/mcp.js"],
      "enabled": true,
      "timeout": 15000
    }
  }
}
```

For **Claude Code** or other MCP clients, point the stdio transport to:

```
node /path/to/OpenMemory-enhanced/packages/openmemory-js/dist/ai/mcp.js
```

No API keys required — defaults to `synthetic` embeddings that work fully offline. Set `OM_EMBEDDINGS=openai` and `OPENAI_API_KEY=...` for higher quality vector search.

### Add Agent Protocol (Optional)

Copy `packages/openmemory-js/SKILL.md` (or the version in this repo's root) into your agent's skill directory. This tells the agent:

1. **Session start** → query recent project memories silently
2. **During work** → save observations with type, project, session
3. **Session end** → save a summary of what was done and learned

## How It Works

```
Session 1 (project: my-app)
  Agent works on auth feature
  → openmemory_store(content: "Chose JWT over sessions because...",
      project: "my-app", observation_type: "decision")
  → openmemory_summarize(project: "my-app", completed: "JWT auth", learned: "...")

Session 2 (project: my-app, new conversation)
  → openmemory_query(query: "recent work", project: "my-app")
  ← "Last session: implemented JWT auth. Decision: chose JWT because..."
  Agent has full context without the user repeating anything.
```

Memories are stored in SQLite at `~/.openmemory-js/data/openmemory.sqlite` (configurable via `OM_DB_PATH`).

## Schema (Added Tables)

```sql
-- New columns on memories table
project text default 'default'
session_id text
observation_type text default 'observation'

-- New tables
sessions(id, project, started_at, ended_at, user_goal, user_id)
summaries(id, session_id, project, request, completed, learned, next_steps, files_modified, created_at)
```

## Configuration

All settings via environment variables (no `.env` file required):

| Variable | Default | Description |
|---|---|---|
| `OM_TIER` | `hybrid` | Performance tier: `fast` / `smart` / `deep` / `hybrid` |
| `OM_EMBEDDINGS` | `synthetic` | Embedding provider: `synthetic` / `openai` / `gemini` / `ollama` / `aws` |
| `OM_DB_PATH` | `~/.openmemory-js/data/openmemory.sqlite` | SQLite database path |
| `OM_VEC_DIM` | `1536` | Vector dimensions |
| `OPENAI_API_KEY` | — | Required only if `OM_EMBEDDINGS=openai` |

## Upstream

Forked from [CaviraOSS/OpenMemory](https://github.com/CaviraOSS/OpenMemory). See upstream for the full OpenMemory documentation, Python SDK, VS Code extension, and deployment options.

## License

[Apache-2.0](LICENSE) (same as upstream)
