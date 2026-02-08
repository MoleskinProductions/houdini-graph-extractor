# Extraction Phases & Coordination Requirements

## Overview

Data extraction follows the same learning path a Houdini artist takes:
formal definitions first, then patterns from expert implementations,
then diverse real-world examples. Each phase produces knowledge that
the agent layer can consume incrementally — and each phase validates
against the one before it.

---

## Phase 1: Structural Foundation

**Goal:** Build the ground-truth knowledge base from Houdini itself.

### 1A — Node Type Introspection (Schema Generator) ✅

**Source:** Houdini via hython subprocess
**Mechanism:** `src/ingestion/node_schema/` (implemented)

Extract for every stock node + Labs node:
- Node type name + category (SOP, DOP, VOP, etc.)
- Input/output port definitions (names, types, variadic flags)
- Parameter templates (name, type, default, range, menu items)
- Parameter dependencies and conditionals
- Node help summary (one-line description)
- Internal operator namespace (e.g., `SideFX::scatter::2.0`)

**Output:** `node_schema.json` — the canonical node type registry.
Stored in a format queryable by the orchestrator via MCP tools.

**Coordination:**
- Extraction owns the introspection pipeline
- HouGraph IR owns the schema data structures
- Orchestrator consumes via MCP tool: `query_node_schema(node_type) → definition`

### 1B — Help Documentation Extraction ✅

**Source:** Houdini help corpus (HTML shipped with Houdini install)
**Location:** `$HFS/houdini/help/nodes/`

Extract per node:
- Full parameter descriptions (what each parm does, not just its type)
- Usage notes and caveats
- "Since" version (when the node was introduced)
- Related nodes (explicitly listed in help pages)
- Embedded example descriptions

Extract globally:
- Concept pages (e.g., "How attributes work", "VEX overview")
- Workflow guides (e.g., "Scattering points on surfaces")

**Output:** Structured help corpus linked to node schema by type name.

**Coordination:**
- Extraction parses and structures the help content
- Links to 1A schema by node type key
- Orchestrator consumes via MCP tool: `query_node_help(node_type) → help_text`
- Concept pages available via: `query_concept(topic) → explanation`

### 1C — Labs Toolset Extraction ✅

**Source:** SideFX Labs HDAs (installed as a Houdini package)
**Mechanism:** `src/ingestion/labs_hda/` (implemented)

Labs nodes are HDAs (Houdini Digital Assets) wrapping subnet logic:
- Extract the same schema as 1A (external interface)
- Additionally extract the internal network (the subnet graph)
- These internal graphs are expert implementations — they demonstrate
  canonical patterns for combining stock nodes

**Output:** Labs schema + internal graph examples in HouGraph IR format.

**Coordination:**
- Labs internal graphs become the first entries in the pattern library
- They validate the HouGraph IR pipeline end-to-end (extract → serialize → rebuild)
- Orchestrator can reference them: `query_example_implementation(task_description)`

---

## Phase 2: Relational Knowledge

**Goal:** Extract how nodes combine — the navigational semantics that
make the search space tractable.

### 2A — Connection Pattern Mining ✅

**Source:** Phase 1C Labs graphs (462 HDAs, 17,248 internal nodes, 19,779 connections)
**Mechanism:** `src/analysis/pattern_mining/` (implemented)

Five mining passes over the HDA graph corpus:
1. **Connection frequency** — directed edge types deduplicated per graph
2. **Downstream/upstream suggestions** — ranked neighbor lists per node type
3. **Node co-occurrence** — Jaccard similarity between type pairs
4. **Port usage statistics** — which ports are actually connected, with usage ratios
5. **Chain patterns** — sequential 2-chains and 3-chains

Optional schema enrichment resolves port indices to human-readable names
using the Phase 1A SchemaCorpus.

**Results** (from 462 Labs HDAs):
- 4,005 connection patterns
- 468 node suggestion sets
- 14,174 co-occurrence pairs
- 479 port usage entries
- 3,384 two-chains, 23,730 three-chains

**Output:** `patterns.json` — PatternCorpus with `save_json`/`load_json`, same API as other corpora.

**CLI:** `houdini-pattern-mine --corpus labs_graphs.json --schema node_schema.json -o patterns.json`

**Coordination:**
- Extraction produces the pattern data
- Orchestrator uses it to narrow search: given current graph state,
  what nodes/connections are most likely next?
- MCP tool: `suggest_next_node(current_graph_state) → ranked_suggestions`
- `PatternCorpus.get_downstream(node_type)` / `get_upstream(node_type)` for fast lookups

### 2B — Intent-to-Subgraph Mapping ✅

**Source:** Phase 1C Labs HDA labels + internal graphs
**Mechanism:** `src/analysis/intent_mapping/` (implemented)

Deterministic string analysis of HDA labels to map high-level intents
to subgraph templates:

1. **Label normalization** — strip "Labs"/"vendor" prefixes, version suffixes,
   stopwords; tokenize into keywords
2. **Prefix-based clustering** — HDAs sharing keyword tokens cluster into
   one intent (e.g. all "Tree *" HDAs → tree_* intents)
3. **Template extraction** — build SubgraphTemplate per HDA with node types,
   counts, and connections; sort richest implementation first

**Output:** `intent_library.json` — IntentLibrary with `save_json`/`load_json`,
same API as other corpora. Supports keyword `search()` and `get_by_category()`.

**CLI:** `houdini-intent-map --corpus labs_graphs.json -o intent_library.json`

**Coordination:**
- Orchestrator decomposes user requests into intents
- Queries template library for starting points
- MCP tool: `query_workflow_template(intent) → HouGraphIR_template`
- Templates are approximate — the orchestrator adapts them to specifics

---

## Phase 3: Applied Knowledge (Video + Research)

**Goal:** Layer rich, diverse, real-world knowledge on top of the
structural and relational foundation.

### 3A — Video Extraction Pipeline ✅

**Source:** YouTube tutorials, official SideFX webinars, Houdini streams
**Mechanism:** Existing dual-stream pipeline (visual + transcript)

Now that Phases 1-2 provide a validated knowledge base:
- Visual extractions validate against node schema (no hallucinated types)
- Transcript entity extraction uses canonical node names from schema
- Extracted graphs validate against known connection patterns
- Parameter values validate against parameter definitions

**Output:** HouGraph IR + annotated transcript events + confidence scores.

**Coordination:**
- Extraction validates against Phase 1 schema before emitting IR
- Orchestrator assigns extraction tasks to specialized agents
- Feedback loop: orchestrator reports which extractions were useful,
  informing extraction priority

### 3B — Specialized Extraction Agents

**Not built yet — topology emerges from experience.**

As video extraction matures, patterns will emerge:
- Some tutorials focus on specific contexts (DOPs, VOPs, SOPs)
- Some require OCR for parameter panels
- Some involve VEX/Python code that needs separate extraction
- Some demonstrate problem-solving strategies, not just node graphs

Agent specialization follows the data:
- Performance analysis reveals which tasks benefit from dedicated agents
- Workloads get subdivided or consolidated based on actual results
- Tool interfaces evolve: overly broad tools get split,
  redundant tools get merged

**Coordination:**
- Orchestrator tracks agent performance per task type
- Extraction pipeline exposes modular stages as independent MCP tools
- Agents compose tools rather than calling monolithic pipelines

---

## Coordination Architecture

```
                    ┌─────────────────────┐
                    │    Orchestrator      │
                    │  (Frontier Model)    │
                    └──────┬──────────────┘
                           │ MCP
              ┌────────────┼────────────────┐
              ▼            ▼                ▼
     ┌────────────┐ ┌────────────┐  ┌─────────────┐
     │ Extraction  │ │ Knowledge  │  │  Houdini    │
     │  Tools      │ │  Store     │  │  Session    │
     └──────┬─────┘ └─────┬──────┘  └──────┬──────┘
            │              │                │
            ▼              ▼                ▼
      HouGraph IR ◄───► Node Schema    hou module
      (instances)       Help Corpus     (live ops)
                        Patterns
                        Templates
```

### MCP Tool Surface (Initial)

**Extraction Tools** (this repo provides):
- `extract_from_video(url, options)` → HouGraph IR
- `extract_help_docs(node_type)` → structured help
- `introspect_node_type(type_name)` → schema definition
- `extract_hip_graph(hip_path, network_path)` → HouGraph IR

**Knowledge Tools** (knowledge store provides):
- `query_node_schema(type_name)` → definition + help
- `query_workflow_template(intent)` → HouGraph IR template
- `suggest_next_node(context)` → ranked suggestions
- `query_concept(topic)` → explanation

**Houdini Session Tools** (already working via MCP):
- `create_node(type, parent, name)` → node path
- `connect_nodes(source, dest, ports)` → connection
- `set_parameter(node, parm, value)` → confirmation
- `get_network_state(path)` → current graph

### Data Flow Between Components

```
Phase 1: Houdini Install ──► Extraction ──► Knowledge Store
                                              (schema, help, Labs graphs)

Phase 2: Knowledge Store ──► Extraction ──► Knowledge Store
         (raw graphs)        (analysis)      (patterns, templates)

Phase 3: Video ──► Extraction ──► Knowledge Store
                   (validated      (examples, strategies)
                    against
                    Phase 1-2)

Runtime: User Request ──► Orchestrator ──► Knowledge + Houdini
                              │                 │
                              ◄─── feedback ────┘
```

### Key Interface: HouGraph IR as Universal Currency

Every component speaks HouGraph IR:
- Extraction emits it
- Knowledge store indexes it
- Orchestrator reasons about it
- Builder consumes it into Houdini

This means:
- Adding a new extraction source = new adapter that emits IR
- Adding a new agent capability = new MCP tool that accepts/returns IR
- The IR format evolves deliberately, not per-component

---

## Immediate Next Steps

### For Extraction (this repo):

1. **Validation integration**
   - Wire Phase 1 schema into video extraction pipeline
   - Extracted node types checked against known definitions
   - Unknown types flagged, not silently accepted
   - Connection patterns validated against Phase 2A data

3. **Define MCP tool interfaces**
   - Specify the extraction tool contracts
   - Build thin MCP wrappers around existing pipeline stages
   - Start with `introspect_node_type`, `extract_help_docs`, `suggest_next_node`

### For HouGraph IR:

4. **Extend format for knowledge metadata**
   - Pattern annotations (frequency, co-occurrence scores)
   - Intent tags on subgraph templates
   - Provenance chain (which phase produced this instance)

### For Orchestrator:

5. **Define knowledge query protocol**
   - How the orchestrator asks for node info, patterns, templates
   - Response format that fits context windows efficiently
   - Progressive detail (summary → full definition on demand)
