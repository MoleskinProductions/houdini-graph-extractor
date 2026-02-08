# CLAUDE.md — Houdini Graph Extractor

## What This Is

The **Data Extraction Agent** for the pixel_vision pipeline. This repo extracts structured Houdini knowledge from multiple sources and emits it as **HouGraph IR** — the universal interchange format shared across all pixel_vision components.

**Peer component:** `houdini_mcp_master` (MCP Server Agent) — exposes extracted data to Claude via Model Context Protocol.

**Shared format:** `hougraph-ir` — canonical IR for Houdini node networks. Lives at `/home/frank-martinelli/pixel_vision/hougraph-ir`.

## Binding Contract

`pixel_vision_interface_contract.md` (v1.0) is the locked interface specification between this repo and the MCP Server. Both agents implement it exactly. Breaking changes require a major version bump. Read it before modifying any data schemas or access patterns.

## Architecture

```
src/
├── ingestion/       # Source acquisition (YouTube, frame, help docs, node schemas)
├── analysis/        # Dual-stream processing
│   ├── transcript_parser.py        # Regex-based transcript event detection
│   ├── llm_transcript_analyzer.py  # LLM-enhanced entity extraction
│   ├── visual_extractor.py         # VLM-based graph topology extraction
│   └── harmonizer.py               # Cross-validates transcript vs visual
├── state/           # Incremental state management
│   ├── models.py       # NodeState, ConnectionState, provenance tracking
│   ├── graph_state.py  # Canonical graph state with confidence decay
│   └── merger.py       # Merge logic with conflict resolution
├── output/          # Formatters
│   ├── hou_data.py            # hou.data JSON format
│   └── hougraph_ir_export.py  # HouGraph IR export
├── config.py        # Centralized configuration
└── main.py          # Click CLI entry point
```

## Extraction Phases (see EXTRACTION_PHASES.md)

| Phase | What | Status |
|-------|------|--------|
| 1A | Node type introspection from Houdini | Implemented |
| 1B | Help documentation parsing | Implemented |
| 1C | Labs HDA internal graph extraction | Not started |
| 2A | Connection pattern mining | Not started |
| 2B | Intent-to-subgraph mapping | Not started |
| 3A | Video tutorial extraction | Implemented |

Phase 1 builds the structural foundation. Phase 2 extracts relational patterns. Phase 3 layers applied knowledge from video — validated against Phases 1-2.

## Conventions

- **HouGraph IR is the output format.** Every extraction path emits IR.
- **Node paths are always absolute**, starting with `/`.
- **Error codes** follow §4.4 of the interface contract.
- **Large data (>1MB)** uses file refs with TTL, not inline payloads. Shared temp: `/tmp/pixel_vision/extract/`.
- **Tests** go in `tests/`, fixtures in `tests/fixtures/`.
- **No implicit dependencies.** If code imports it, `pyproject.toml` declares it (core or optional).

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## CLI

```bash
# Video extraction (Phase 3A)
houdini-extract <youtube-url> -o output.json

# See all options
houdini-extract --help
```

## Key Design Decisions

1. **Dual-stream harmonization** — Visual and transcript processed independently, then cross-validated. Each stream can validate/enhance the other.
2. **Confidence + provenance tracking** — Every node/connection tracks where it came from, when, and how confident we are. Confidence decays if not re-observed.
3. **Incremental state merging** — Frames processed in order, each merged into canonical state. Supports streaming/online processing.
4. **Peer process architecture** — Extraction and MCP Server are peers, not plugin/host. Communicate via HTTP bridge + file refs.
