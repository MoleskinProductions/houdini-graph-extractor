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

### 1B — Help Documentation Extraction

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

### 1C — Labs Toolset Extraction

**Source:** SideFX Labs HDAs (installed as a Houdini package)
**Mechanism:** HDA introspection via hou module

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

### 2A — Connection Pattern Mining

**Source:** Phase 1C Labs graphs + Houdini example .hip files
**Mechanism:** Graph analysis on HouGraph IR instances

Extract:
- Common upstream/downstream relationships (e.g., scatter → copy_to_points input 1)
- Typical subgraph patterns (e.g., the "scatter + copy + instance" idiom)
- Port usage statistics (which inputs are most commonly connected)
- Parameter co-occurrence (when node X has param A set, node Y often has param B)

**Output:** Pattern graph / co-occurrence matrix queryable by context.

**Coordination:**
- Extraction produces the pattern data
- Orchestrator uses it to narrow search: given current graph state,
  what nodes/connections are most likely next?
- MCP tool: `suggest_next_node(current_graph_state) → ranked_suggestions`

### 2B — Context-to-Subgraph Mapping

**Source:** Help examples, Labs internals, example hip files
**Mechanism:** Clustering and tagging extracted graphs by task/domain

Map high-level intents to subgraph templates:
- "Scatter points on a surface" → scatter + attribnoise + copy_to_points
- "Simulate cloth" → cloth_object + cloth_solver + constraint setup
- "Create procedural rocks" → sphere + mountain + remesh + material

**Output:** Intent-indexed template library.

**Coordination:**
- Orchestrator decomposes user requests into intents
- Queries template library for starting points
- MCP tool: `query_workflow_template(intent) → HouGraphIR_template`
- Templates are approximate — the orchestrator adapts them to specifics

---

## Phase 3: Applied Knowledge (Video + Research)

**Goal:** Layer rich, diverse, real-world knowledge on top of the
structural and relational foundation.

### 3A — Video Extraction Pipeline (Current Repo)

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

1. **Build help doc parser** (Phase 1B)
   - Parse $HFS/houdini/help/nodes/ HTML
   - Structure per-node: params, notes, related nodes, examples
   - Link to schema by node type

2. **Extend schema generator** (Phase 1A)
   - Add Labs HDA introspection (internal networks)
   - Export internal graphs as HouGraph IR instances
   - Add help summary extraction from hou module

3. **Define MCP tool interfaces**
   - Specify the extraction tool contracts
   - Build thin MCP wrappers around existing pipeline stages
   - Start with `introspect_node_type` and `extract_help_docs`

4. **Validation integration**
   - Wire Phase 1 schema into extraction pipeline
   - Extracted node types checked against known definitions
   - Unknown types flagged, not silently accepted

### For HouGraph IR:

5. **Extend format for knowledge metadata**
   - Pattern annotations (frequency, co-occurrence scores)
   - Intent tags on subgraph templates
   - Provenance chain (which phase produced this instance)

### For Orchestrator:

6. **Define knowledge query protocol**
   - How the orchestrator asks for node info, patterns, templates
   - Response format that fits context windows efficiently
   - Progressive detail (summary → full definition on demand)
