# Houdini Node Graph Extractor

## Project Overview

A video-to-node-graph pipeline that extracts Houdini network structures from YouTube tutorial videos by harmonizing transcript analysis with visual frame extraction. Outputs `hou.data` compatible JSON for direct injection into Houdini via hrpyc/MCP.

### Core Premise

YouTube tutorials narrate actions ("I'm adding an attribute wrangle here", "connect the output to scatter") while showing the node graph. By synchronizing transcript events with visual frame analysis, we can:

1. Use transcript as a guide for *when* to sample frames
2. Use transcript semantics to *validate* visual extraction
3. Resolve visual ambiguity with spoken context
4. Track incremental graph construction over time

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ORCHESTRATOR AGENT                               │
│  (coordinates extraction loop, maintains graph state, resolves conflicts)   │
└─────────────────────────────────────────────────────────────────────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────────┐
│  TRANSCRIPT   │         │  FRAME          │         │  GRAPH STATE        │
│  ANALYZER     │         │  EXTRACTOR      │         │  MANAGER            │
│               │         │                 │         │                     │
│  - fetch YT   │         │  - keyframe     │         │  - merge deltas     │
│    transcript │         │    extraction   │         │  - validate conns   │
│  - detect     │         │  - graph detect │         │  - resolve conflicts│
│    action     │         │  - node OCR     │         │  - emit hou.data    │
│    events     │         │  - topology     │         │                     │
│  - timestamp  │         │    extraction   │         │                     │
│    alignment  │         │                 │         │                     │
└───────────────┘         └─────────────────┘         └─────────────────────┘
        │                           │                           │
        └───────────────────────────┴───────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │  HOUDINI NODE         │
                        │  REFERENCE DATABASE   │
                        │                       │
                        │  - node type registry │
                        │  - icon embeddings    │
                        │  - port definitions   │
                        │  - param schemas      │
                        └───────────────────────┘
```

---

## Components

### 1. YouTube Ingestion Module

**Responsibilities:**
- Accept YouTube URL as input
- Download video at optimal resolution for OCR (1080p preferred)
- Fetch existing transcript via `yt-dlp` or YouTube API
- Parse transcript into timestamped segments

**Output:**
```python
{
    "video_id": "dQw4w9WgXcQ",
    "title": "Houdini Tutorial - Procedural Rocks",
    "duration_seconds": 1847,
    "video_path": "/tmp/extracted/video.mp4",
    "transcript": [
        {"start": 0.0, "end": 4.2, "text": "Hey everyone, today we're building a rock generator"},
        {"start": 4.2, "end": 8.1, "text": "Let's start by dropping down a sphere"},
        ...
    ]
}
```

**Dependencies:**
- `yt-dlp` for video/transcript download
- `youtube-transcript-api` as fallback for transcripts

---

### 2. Transcript Analyzer

**Responsibilities:**
- Process transcript segments to detect "action events"
- Classify events by type: node_create, node_connect, param_change, context_switch
- Extract entity mentions: node names, node types, parameter names, values
- Generate temporal markers for frame sampling

**Action Event Schema:**
```python
{
    "timestamp": 45.2,
    "event_type": "node_create",
    "confidence": 0.85,
    "entities": {
        "node_type": "attribwrangle",
        "node_name": null,  # often not mentioned
        "context": "geometry network"
    },
    "raw_text": "and I'll drop down an attribute wrangle"
}
```

**Event Types:**
| Type | Trigger Phrases |
|------|-----------------|
| `node_create` | "add a", "drop down", "create", "put in" |
| `node_connect` | "connect", "wire", "plug into", "feed into" |
| `param_change` | "set this to", "change the", "adjust", "type in" |
| `node_delete` | "delete", "remove", "get rid of" |
| `node_rename` | "rename", "call this" |
| `context_switch` | "go into", "dive into", "jump to", "back to" |
| `selection` | "select", "click on", "grab" |

**Implementation Notes:**
- Use an LLM (local or API) for entity extraction
- Build a Houdini vocabulary list for fuzzy matching node types
- Handle common speech patterns: "scatter SOP" → "scatter", "attrib wrangle" → "attribwrangle"

---

### 3. Frame Extractor

**Responsibilities:**
- Extract frames at action event timestamps (± buffer window)
- Detect which frames contain visible node graphs
- Perform node-level segmentation
- Extract topology via Qwen3-VL

**Frame Sampling Strategy:**
```python
def get_sample_timestamps(action_events, buffer_sec=2.0, interval=0.5):
    """
    For each action event, sample frames in a window around it.
    The graph state is often visible *after* the narrator describes the action.
    """
    timestamps = []
    for event in action_events:
        # Sample more heavily after the spoken action
        start = event.timestamp - buffer_sec * 0.25
        end = event.timestamp + buffer_sec
        t = start
        while t <= end:
            timestamps.append({
                "time": t,
                "source_event": event,
                "priority": "high" if t > event.timestamp else "low"
            })
            t += interval
    return dedupe_and_sort(timestamps)
```

**Graph Detection Prompt:**
```
Analyze this frame from a Houdini tutorial video.

1. Is a node graph/network editor visible? (yes/no)
2. If yes, what percentage of the frame does it occupy?
3. Is the graph clearly readable (good resolution, not motion-blurred)?
4. What network context is this? (SOP, DOP, VOP, OBJ, SHOP, COP, TOP, LOP)

Respond as JSON:
{
    "graph_visible": true,
    "graph_area_percent": 65,
    "readability": "high",
    "network_context": "SOP",
    "notes": "Parameter panel visible on right side"
}
```

**Topology Extraction Prompt:**
```
Extract the complete node graph structure from this Houdini screenshot.

For each node, identify:
- The node name (text label below/on the node)
- The node type (from icon/shape/color)
- Approximate grid position (normalize to 0-100 coordinate space)
- Visible input/output connections

For each connection, identify:
- Source node name and output index
- Destination node name and input index

Pay careful attention to:
- Wire colors (different data types)
- Dotted vs solid lines (some connections may be references)
- Nodes that are bypassed (typically shown with strikethrough or different color)
- The display/render flag indicators (blue/purple badges on rightmost nodes)

Output as JSON:
{
    "network_context": "SOP",
    "parent_path": "/obj/geo1",  // if visible in breadcrumb
    "nodes": [
        {
            "name": "sphere1",
            "type": "sphere",
            "position": [20, 15],
            "flags": {"display": false, "render": false, "bypass": false},
            "inputs_connected": [false],
            "outputs_connected": [true]
        },
        {
            "name": "mountain1", 
            "type": "mountain",
            "position": [20, 35],
            "flags": {"display": true, "render": true, "bypass": false},
            "inputs_connected": [true],
            "outputs_connected": [false]
        }
    ],
    "connections": [
        {"from_node": "sphere1", "from_output": 0, "to_node": "mountain1", "to_input": 0}
    ],
    "extraction_confidence": 0.9,
    "uncertain_elements": ["wire routing unclear between nodes X and Y"]
}
```

---

### 4. Houdini Node Reference Database

**Purpose:**
Pre-built knowledge base of Houdini node types for validation and disambiguation.

**Structure:**
```
/reference_db/
├── node_registry.json      # All node types with metadata
├── icons/                   # Reference screenshots of each node
│   ├── sop/
│   │   ├── sphere.png
│   │   ├── attribwrangle.png
│   │   └── ...
│   ├── dop/
│   └── ...
├── embeddings/              # Pre-computed Qwen3-VL-Embedding vectors
│   └── node_icons.npy
└── port_definitions.json    # Input/output port specs per node type
```

**Node Registry Schema:**
```python
{
    "attribwrangle": {
        "category": "SOP",
        "display_name": "Attribute Wrangle",
        "aliases": ["attrib wrangle", "wrangle", "vex wrangle"],
        "icon_color": "#5c7a3d",  # olive green
        "default_inputs": 4,
        "default_outputs": 1,
        "common_params": ["snippet", "class", "vex_exportlist"],
        "param_schema": {
            "snippet": {"type": "string", "default": ""},
            "class": {"type": "menu", "options": ["point", "vertex", "primitive", "detail"]}
        }
    },
    ...
}
```

**Embedding Index Usage:**
```python
def validate_node_type(cropped_node_image, claimed_type, threshold=0.75):
    """
    Verify VL model's node type identification against reference embeddings.
    """
    query_embedding = embedding_model.encode(cropped_node_image)
    reference_embedding = load_embedding(f"icons/{claimed_type}.png")
    similarity = cosine_similarity(query_embedding, reference_embedding)
    
    if similarity < threshold:
        # Find actual closest match
        all_similarities = compare_against_all(query_embedding)
        best_match = all_similarities.argmax()
        return {
            "claimed": claimed_type,
            "corrected": best_match.name,
            "confidence": best_match.similarity
        }
    return {"claimed": claimed_type, "corrected": None, "confidence": similarity}
```

---

### 5. Graph State Manager

**Responsibilities:**
- Maintain canonical graph state across video timeline
- Merge incremental extractions from sequential frames
- Detect and resolve conflicts between transcript hints and visual extraction
- Handle node renames, deletes, and reconnections
- Emit final hou.data JSON

**State Schema:**
```python
{
    "meta": {
        "source_video": "https://youtube.com/watch?v=...",
        "extraction_timestamp": "2025-01-17T...",
        "houdini_version_hint": "20.5",  # if mentioned in video
        "network_contexts": ["SOP", "MAT"]
    },
    "networks": {
        "/obj/geo1": {
            "type": "geo",
            "children": {
                "sphere1": {
                    "type": "sphere",
                    "position": [0, 0],
                    "params": {"type": "polymesh", "rows": 24, "cols": 24},
                    "first_seen": 45.2,
                    "last_seen": 312.0,
                    "confidence": 0.95
                },
                ...
            },
            "connections": [
                {"src": "sphere1", "src_out": 0, "dst": "mountain1", "dst_in": 0, "first_seen": 48.0}
            ]
        }
    },
    "timeline": [
        {"time": 45.2, "action": "node_create", "target": "/obj/geo1/sphere1"},
        {"time": 48.0, "action": "node_connect", "src": "sphere1", "dst": "mountain1"},
        ...
    ]
}
```

**Merge Strategy:**
```python
def merge_extraction(current_state, new_extraction, timestamp, transcript_hint=None):
    """
    Merge a new frame extraction into canonical state.
    
    Rules:
    1. New nodes are added if not present
    2. Existing nodes update position only if significantly different
    3. Connections are additive unless explicitly deleted
    4. Transcript hints can override low-confidence visual extractions
    5. Track provenance: which frame/timestamp contributed each element
    """
    for node in new_extraction.nodes:
        if node.name in current_state.nodes:
            existing = current_state.nodes[node.name]
            # Update if positions drifted (user reorganized)
            if position_distance(existing.position, node.position) > THRESHOLD:
                existing.position = node.position
                existing.last_seen = timestamp
        else:
            # New node
            if transcript_hint and transcript_hint.event_type == "node_create":
                # Boost confidence if transcript corroborates
                node.confidence = min(1.0, node.confidence + 0.15)
            current_state.nodes[node.name] = node
            current_state.nodes[node.name].first_seen = timestamp
    
    # Similar logic for connections...
    return current_state
```

---

### 6. Orchestrator Agent

**Responsibilities:**
- Coordinate the extraction loop
- Decide when to sample more frames vs. proceed
- Resolve conflicts between transcript and visual analysis
- Handle edge cases: zoomed views, parameter panels, viewport captures

**Agent Loop:**
```
INITIALIZE:
    - Fetch video and transcript
    - Parse transcript into action events
    - Initialize empty graph state
    - Load node reference database

PHASE 1 - COARSE EXTRACTION:
    For each action_event in transcript:
        - Sample frames around event timestamp
        - Detect frames with visible graphs
        - Run topology extraction on best frame
        - Merge into graph state with transcript context
        
PHASE 2 - REFINEMENT:
    - Identify low-confidence nodes/connections
    - For each uncertain element:
        - Sample additional frames from nearby timestamps
        - Cross-reference against node reference DB
        - Use transcript context to disambiguate
        - If still uncertain, flag for manual review
        
PHASE 3 - VALIDATION:
    - Check graph connectivity (orphan nodes?)
    - Validate node types against Houdini registry
    - Verify connection compatibility (port types)
    - Check for common patterns (feedback loops, switches)
    
PHASE 4 - OUTPUT:
    - Convert canonical state to hou.data format
    - Generate reconstruction script
    - Optionally: generate diff from any provided base graph
```

**Conflict Resolution:**
```python
def resolve_conflict(visual_extraction, transcript_hint, context):
    """
    When visual and transcript disagree, use heuristics to pick winner.
    """
    conflict_type = detect_conflict_type(visual_extraction, transcript_hint)
    
    if conflict_type == "node_type_mismatch":
        # Transcript says "scatter" but visual shows "copy to points"
        # Check if transcript might be using informal name
        if is_alias(transcript_hint.entity, visual_extraction.node_type):
            return visual_extraction  # Visual is canonical
        # Check visual confidence
        if visual_extraction.confidence < 0.7:
            return transcript_hint  # Trust transcript for unclear visuals
        # Otherwise, flag for review
        return flag_for_review(visual_extraction, transcript_hint)
    
    if conflict_type == "node_missing":
        # Transcript mentions node not visible in frame
        # Might be off-screen, check adjacent frames
        return expand_search(context.timestamp, context.node_name)
    
    if conflict_type == "extra_node":
        # Visual shows node not mentioned in transcript
        # Could be pre-existing or undocumented
        return accept_visual(visual_extraction, confidence_penalty=0.1)
```

---

## Output Format

### Primary Output: hou.data JSON

```json
{
    "version": "1.0",
    "source": {
        "video_url": "https://youtube.com/watch?v=xyz",
        "video_title": "Houdini Procedural Rocks Tutorial",
        "extracted_at": "2025-01-17T14:30:00Z"
    },
    "networks": [
        {
            "path": "/obj/geo1",
            "type": "geo",
            "nodes": [
                {
                    "name": "sphere1",
                    "type": "sphere",
                    "position": [0, 0],
                    "params": {
                        "type": "polymesh",
                        "rows": 24,
                        "cols": 24
                    }
                },
                {
                    "name": "mountain1",
                    "type": "mountain",
                    "position": [0, -1.5],
                    "params": {
                        "height": 0.3,
                        "elementsize": 0.1
                    }
                },
                {
                    "name": "attribwrangle1",
                    "type": "attribwrangle",
                    "position": [0, -3.0],
                    "params": {
                        "snippet": "@P *= fit01(rand(@ptnum), 0.8, 1.2);",
                        "class": "point"
                    }
                }
            ],
            "connections": [
                ["sphere1", 0, "mountain1", 0],
                ["mountain1", 0, "attribwrangle1", 0]
            ],
            "display_node": "attribwrangle1",
            "render_node": "attribwrangle1"
        }
    ],
    "extraction_metadata": {
        "total_frames_analyzed": 145,
        "action_events_detected": 23,
        "confidence_score": 0.87,
        "flagged_for_review": [
            {
                "element": "connection mountain1 -> attribwrangle1",
                "reason": "wire partially obscured",
                "timestamp": 127.4
            }
        ]
    }
}
```

### Secondary Output: Reconstruction Script

```python
# Auto-generated Houdini reconstruction script
# Source: https://youtube.com/watch?v=xyz
# Generated: 2025-01-17

import hou

def build_network(parent_path="/obj"):
    # Create geo container
    obj = hou.node(parent_path)
    geo = obj.createNode("geo", "geo1")
    geo.moveToGoodPosition()
    
    # Create nodes
    sphere1 = geo.createNode("sphere", "sphere1")
    sphere1.parm("type").set("polymesh")
    sphere1.parm("rows").set(24)
    sphere1.parm("cols").set(24)
    sphere1.setPosition(hou.Vector2(0, 0))
    
    mountain1 = geo.createNode("mountain", "mountain1")
    mountain1.parm("height").set(0.3)
    mountain1.parm("elementsize").set(0.1)
    mountain1.setPosition(hou.Vector2(0, -1.5))
    
    attribwrangle1 = geo.createNode("attribwrangle", "attribwrangle1")
    attribwrangle1.parm("snippet").set("@P *= fit01(rand(@ptnum), 0.8, 1.2);")
    attribwrangle1.parm("class").set("point")
    attribwrangle1.setPosition(hou.Vector2(0, -3.0))
    
    # Create connections
    mountain1.setInput(0, sphere1, 0)
    attribwrangle1.setInput(0, mountain1, 0)
    
    # Set flags
    attribwrangle1.setDisplayFlag(True)
    attribwrangle1.setRenderFlag(True)
    
    return geo

if __name__ == "__main__":
    build_network()
```

---

## Technology Stack

### Required

| Component | Recommended | Alternatives |
|-----------|-------------|--------------|
| Video Download | `yt-dlp` | youtube-dl |
| Transcript Fetch | `youtube-transcript-api` | yt-dlp (built-in) |
| Frame Extraction | `ffmpeg` via subprocess | OpenCV |
| Vision-Language Model | Qwen3-VL-8B (local) | Qwen3-VL-32B, API providers |
| Embedding Model | Qwen3-VL-Embedding-2B | Qwen3-VL-Embedding-8B |
| LLM Inference | vLLM, Ollama | llama.cpp, SGLang |
| Transcript Analysis | Claude API, local LLM | Qwen3 text model |
| Vector Store | FAISS, ChromaDB | Qdrant, Milvus |

### Python Dependencies

```
yt-dlp>=2024.0.0
youtube-transcript-api>=0.6.0
ffmpeg-python>=0.2.0
torch>=2.0.0
transformers>=4.57.0
vllm>=0.11.0  # if using vLLM
qwen-vl-utils>=0.0.14
faiss-cpu>=1.7.0  # or faiss-gpu
numpy>=1.24.0
pydantic>=2.0.0  # for schema validation
```

---

## Implementation Phases

### Phase 1: Foundation (MVP)

**Goal:** Extract a simple linear node chain from a single video.

- [ ] YouTube video download + transcript fetch
- [ ] Basic transcript parsing (regex-based action detection)
- [ ] Frame extraction at fixed intervals
- [ ] Single-frame graph extraction with Qwen3-VL
- [ ] Basic JSON output (no validation)

**Test Case:** Simple tutorial with 5-10 nodes, linear connections.

### Phase 2: Harmonization

**Goal:** Synchronize transcript with visual extraction.

- [ ] LLM-based transcript analysis for action events
- [ ] Timestamp-guided frame sampling
- [ ] Multi-frame extraction with state merging
- [ ] Transcript-visual conflict detection
- [ ] Confidence scoring

**Test Case:** Tutorial where narrator describes actions before/after they appear.

### Phase 3: Validation & Refinement

**Goal:** Production-quality extraction with error handling.

- [ ] Node reference database (top 100 SOPs)
- [ ] Embedding-based node type validation
- [ ] Connection compatibility checking
- [ ] Parameter extraction (from visible panels)
- [ ] Reconstruction script generation

**Test Case:** Complex tutorial with 20+ nodes, branches, and parameter changes.

### Phase 4: Agent Loop

**Goal:** Autonomous extraction with self-correction.

- [ ] Orchestrator agent implementation
- [ ] Adaptive frame sampling (focus on uncertain areas)
- [ ] Multi-pass refinement
- [ ] Flagging system for manual review
- [ ] Support for multiple network contexts (dive into subnetworks)

**Test Case:** Full-length tutorial (30+ minutes) with multiple network types.

### Phase 5: Integration

**Goal:** Direct Houdini integration via MCP.

- [ ] hrpyc MCP server integration
- [ ] Live reconstruction preview
- [ ] Diff-based updates (incremental reconstruction)
- [ ] User feedback loop (correct extraction errors)
- [ ] Batch processing for playlists

---

## Directory Structure

```
houdini-graph-extractor/
├── README.md
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── main.py                    # CLI entry point
│   ├── config.py                  # Configuration management
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── youtube.py             # Video/transcript download
│   │   └── frame_extractor.py     # ffmpeg frame extraction
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── transcript_parser.py   # Action event detection
│   │   ├── visual_extractor.py    # Qwen3-VL integration
│   │   └── harmonizer.py          # Transcript-visual alignment
│   ├── state/
│   │   ├── __init__.py
│   │   ├── graph_state.py         # Canonical state management
│   │   ├── merger.py              # Extraction merging
│   │   └── validator.py           # Graph validation
│   ├── reference/
│   │   ├── __init__.py
│   │   ├── node_registry.py       # Houdini node database
│   │   └── embedding_index.py     # Node type embeddings
│   ├── output/
│   │   ├── __init__.py
│   │   ├── hou_data.py            # JSON formatter
│   │   └── script_generator.py    # Python script generation
│   └── agent/
│       ├── __init__.py
│       └── orchestrator.py        # Main agent loop
├── reference_db/
│   ├── node_registry.json
│   ├── icons/
│   └── embeddings/
├── tests/
│   ├── test_transcript.py
│   ├── test_extraction.py
│   └── fixtures/
└── examples/
    └── sample_outputs/
```

---

## Configuration

```yaml
# config.yaml

extraction:
  frame_buffer_seconds: 2.0
  frame_interval_seconds: 0.5
  min_graph_area_percent: 30
  min_readability: "medium"

models:
  vision_language:
    model_name: "Qwen/Qwen3-VL-8B-Instruct"
    backend: "vllm"  # or "ollama", "transformers"
    device: "cuda"
    max_tokens: 4096
  
  embedding:
    model_name: "Qwen/Qwen3-VL-Embedding-2B"
    device: "cuda"
  
  transcript_analysis:
    provider: "anthropic"  # or "local"
    model: "claude-sonnet-4-20250514"

validation:
  node_type_confidence_threshold: 0.75
  connection_confidence_threshold: 0.8
  require_transcript_corroboration: false

output:
  format: "hou_data"  # or "script", "both"
  include_metadata: true
  include_timeline: true
```

---

## CLI Interface

```bash
# Basic extraction
houdini-extract https://youtube.com/watch?v=xyz -o output.json

# With options
houdini-extract https://youtube.com/watch?v=xyz \
    --output output.json \
    --format both \
    --confidence-threshold 0.8 \
    --model qwen3-vl-32b \
    --verbose

# Batch processing
houdini-extract --playlist https://youtube.com/playlist?list=xyz \
    --output-dir ./extracted/

# With MCP integration (live preview)
houdini-extract https://youtube.com/watch?v=xyz \
    --mcp-server localhost:9001 \
    --live-preview
```

---

## Error Handling

| Error Type | Handling Strategy |
|------------|-------------------|
| No transcript available | Fall back to frame-only extraction with whisper |
| Graph not visible | Skip segment, interpolate from adjacent frames |
| OCR failure on node names | Use embedding similarity to infer type, flag name as unknown |
| Conflicting extractions | Use confidence weighting, flag for review if tied |
| Unknown node type | Add to output with `"type": "unknown"`, include screenshot |
| Connection ambiguity | Sample more frames, use transcript hints, flag if unresolved |

---

## Future Enhancements

- **Whisper integration**: Generate transcripts for videos without captions
- **Parameter inference**: OCR visible parameter panels, track changes over time
- **Code extraction**: Capture VEX/Python snippets shown in editor panels
- **Multi-video learning**: Build node patterns across tutorial corpus
- **Interactive correction**: UI for human-in-the-loop refinement
- **HDA detection**: Identify when tutorials use custom HDAs
- **Version detection**: Infer Houdini version from UI elements

---

## References

- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL-Embedding: https://github.com/QwenLM/Qwen3-VL-Embedding
- hou.data format: https://www.sidefx.com/docs/houdini/hom/hou/Node.html
- hrpyc: https://github.com/teared/hrpyc
