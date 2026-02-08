# Houdini Graph Extractor

Data extraction agent for the **pixel_vision** pipeline. Extracts structured Houdini node graph knowledge from multiple sources and emits [HouGraph IR](../hougraph-ir) — the canonical interchange format for the system.

## Extraction Sources

| Source | Phase | Status |
|--------|-------|--------|
| Houdini node introspection (stock + Labs) | 1A, 1C | Planned |
| Houdini help documentation | 1B | Planned |
| Connection pattern mining | 2A | Planned |
| YouTube video tutorials | 3A | Implemented |

The extraction pipeline builds knowledge in layers: structural foundation (node definitions, parameter schemas) → relational patterns (common node combinations) → applied examples (video tutorials). Each layer validates against the one before it.

## Architecture

**Video extraction** uses dual-stream processing:
- **Visual stream:** VLM extracts node graph topology from keyframes
- **Transcript stream:** LLM/regex extracts action events from speech
- **Harmonizer:** Cross-validates both streams, resolves conflicts
- **State manager:** Merges incremental extractions with confidence tracking

Supports multiple VLM backends (Google Gemini, OpenRouter, vLLM, Ollama).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# With HouGraph IR support
pip install -e ".[hougraph]"

# With dev tools
pip install -e ".[dev]"
```

**External requirements:** `ffmpeg` and `ffprobe` must be on PATH.

## Usage

```bash
# Extract graph from a YouTube tutorial
houdini-extract https://youtube.com/watch?v=VIDEO_ID -o output.json

# With specific VLM backend
houdini-extract URL -o output.json --api-base http://localhost:11434/v1 --model qwen2-vl

# See all options
houdini-extract --help
```

## Related Components

- **[hougraph-ir](../hougraph-ir)** — Interchange format (shared data structures, validation, Houdini builder)
- **[houdini_mcp_master](../houdini_mcp_master)** — MCP Server that exposes extracted data to Claude
- **[pixel_vision_interface_contract.md](pixel_vision_interface_contract.md)** — Binding specification between extraction and MCP server

## Testing

```bash
pytest
```

## Project Documentation

- `CLAUDE.md` — Development conventions and architecture reference
- `EXTRACTION_PHASES.md` — Phased extraction strategy and coordination requirements
- `houdini-graph-extractor-spec.md` — Detailed technical specification
- `pixel_vision_interface_contract.md` — Interface contract (v1.0, locked)
