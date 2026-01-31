"""Visual extraction using Qwen3-VL via vLLM OpenAI-compatible API."""

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from ..config import get_config


@dataclass
class NodeExtraction:
    """Extracted node information."""

    name: str
    type: str
    position: list[float]
    flags: dict = field(default_factory=lambda: {"display": False, "render": False, "bypass": False})


@dataclass
class ConnectionExtraction:
    """Extracted connection information."""

    from_node: str
    from_output: int
    to_node: str
    to_input: int


@dataclass
class GraphExtraction:
    """Complete graph extraction from a frame."""

    network_context: str
    parent_path: str | None
    nodes: list[NodeExtraction]
    connections: list[ConnectionExtraction]
    extraction_confidence: float
    graph_visible: bool
    graph_area_percent: int
    readability: str
    uncertain_elements: list[str] = field(default_factory=list)
    timestamp: float = 0.0


# Prompt for detecting if a graph is visible
GRAPH_DETECTION_PROMPT = """Analyze this frame from a Houdini tutorial video.

1. Is a node graph/network editor visible? (yes/no)
2. If yes, what percentage of the frame does it occupy?
3. Is the graph clearly readable (good resolution, not motion-blurred)?
4. What network context is this? (SOP, DOP, VOP, OBJ, SHOP, COP, TOP, LOP, or unknown)

Respond ONLY with valid JSON:
{
    "graph_visible": true,
    "graph_area_percent": 65,
    "readability": "high",
    "network_context": "SOP",
    "notes": "Parameter panel visible on right side"
}"""


# Prompt for full topology extraction
TOPOLOGY_EXTRACTION_PROMPT = """Extract the complete node graph structure from this Houdini screenshot.

For each node, identify:
- The node name (text label below or on the node)
- The node type (from icon/shape/color - common types: sphere, box, grid, merge, null, attribwrangle, scatter, copytopoints, mountain, transform, etc.)
- Approximate grid position (normalize to 0-100 coordinate space where 0,0 is top-left)
- Display/render flags if visible (blue=display, purple=render)

For each connection, identify:
- Source node name and output index (usually 0)
- Destination node name and input index (usually 0)

Pay attention to:
- Wire colors indicate data types
- Dotted vs solid lines
- Bypassed nodes (strikethrough or faded)
- The display/render flag badges on nodes

Output ONLY valid JSON in this exact format:
{
    "network_context": "SOP",
    "parent_path": "/obj/geo1",
    "nodes": [
        {
            "name": "sphere1",
            "type": "sphere",
            "position": [20, 15],
            "flags": {"display": false, "render": false, "bypass": false}
        }
    ],
    "connections": [
        {"from_node": "sphere1", "from_output": 0, "to_node": "mountain1", "to_input": 0}
    ],
    "extraction_confidence": 0.9,
    "uncertain_elements": ["wire routing unclear between X and Y"]
}

If you cannot clearly read a node name, use a descriptive placeholder like "node_1".
If you cannot determine a node type, use "unknown"."""


class VisualExtractor:
    """Extract node graph information from video frames using vision-language models."""

    def __init__(self, api_base: str | None = None, model: str | None = None, api_key: str | None = None):
        import os
        config = get_config()
        self.api_base = api_base or config.vlm.api_base
        self.model = model or config.vlm.model_name
        self.max_tokens = config.vlm.max_tokens
        self.temperature = config.vlm.temperature

        # Get API key from parameter, config, or environment
        self.api_key = (
            api_key
            or config.vlm.api_key
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or "not-needed"
        )

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for API."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_vlm(self, image_path: Path, prompt: str, max_retries: int = 3) -> str:
        """Make a vision-language model call with retry logic for rate limits."""
        import time
        image_data = self._encode_image(image_path)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_data}",
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content

            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait_time = (attempt + 1) * 15  # 15s, 30s, 45s
                    print(f"Rate limited, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed after {max_retries} retries")

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from model response, handling markdown code blocks."""
        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            response = json_match.group(1)

        # Clean up common issues
        response = response.strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse JSON response: {e}")
            print(f"Response was: {response[:500]}")
            return {}

    def detect_graph(self, image_path: Path) -> dict:
        """
        Detect if a node graph is visible in the frame.

        Returns:
            Dict with graph_visible, graph_area_percent, readability, network_context
        """
        response = self._call_vlm(image_path, GRAPH_DETECTION_PROMPT)
        result = self._parse_json_response(response)

        # Set defaults for missing fields
        result.setdefault("graph_visible", False)
        result.setdefault("graph_area_percent", 0)
        result.setdefault("readability", "low")
        result.setdefault("network_context", "unknown")

        return result

    def extract_topology(self, image_path: Path, timestamp: float = 0.0) -> GraphExtraction:
        """
        Extract full node graph topology from a frame.

        Args:
            image_path: Path to the frame image
            timestamp: Timestamp of the frame in the video

        Returns:
            GraphExtraction with nodes and connections
        """
        # First detect if graph is visible
        detection = self.detect_graph(image_path)

        if not detection.get("graph_visible", False):
            return GraphExtraction(
                network_context="unknown",
                parent_path=None,
                nodes=[],
                connections=[],
                extraction_confidence=0.0,
                graph_visible=False,
                graph_area_percent=0,
                readability="none",
                timestamp=timestamp,
            )

        # Check readability threshold
        readability = detection.get("readability", "low")
        if readability == "low":
            return GraphExtraction(
                network_context=detection.get("network_context", "unknown"),
                parent_path=None,
                nodes=[],
                connections=[],
                extraction_confidence=0.3,
                graph_visible=True,
                graph_area_percent=detection.get("graph_area_percent", 0),
                readability=readability,
                uncertain_elements=["Graph visible but not readable enough for extraction"],
                timestamp=timestamp,
            )

        # Extract full topology
        response = self._call_vlm(image_path, TOPOLOGY_EXTRACTION_PROMPT)
        data = self._parse_json_response(response)

        if not data:
            return GraphExtraction(
                network_context=detection.get("network_context", "unknown"),
                parent_path=None,
                nodes=[],
                connections=[],
                extraction_confidence=0.2,
                graph_visible=True,
                graph_area_percent=detection.get("graph_area_percent", 0),
                readability=readability,
                uncertain_elements=["Failed to parse extraction response"],
                timestamp=timestamp,
            )

        # Parse nodes
        nodes = []
        for node_data in data.get("nodes", []):
            nodes.append(
                NodeExtraction(
                    name=node_data.get("name", "unknown"),
                    type=node_data.get("type", "unknown"),
                    position=node_data.get("position", [0, 0]),
                    flags=node_data.get(
                        "flags", {"display": False, "render": False, "bypass": False}
                    ),
                )
            )

        # Parse connections
        connections = []
        for conn_data in data.get("connections", []):
            connections.append(
                ConnectionExtraction(
                    from_node=conn_data.get("from_node", ""),
                    from_output=conn_data.get("from_output", 0),
                    to_node=conn_data.get("to_node", ""),
                    to_input=conn_data.get("to_input", 0),
                )
            )

        return GraphExtraction(
            network_context=data.get("network_context", detection.get("network_context", "unknown")),
            parent_path=data.get("parent_path"),
            nodes=nodes,
            connections=connections,
            extraction_confidence=data.get("extraction_confidence", 0.7),
            graph_visible=True,
            graph_area_percent=detection.get("graph_area_percent", 0),
            readability=readability,
            uncertain_elements=data.get("uncertain_elements", []),
            timestamp=timestamp,
        )

    def extract_from_frames(
        self,
        frames: list,
        skip_low_priority: bool = False,
    ) -> list[GraphExtraction]:
        """
        Extract graph information from multiple frames.

        Args:
            frames: List of ExtractedFrame objects
            skip_low_priority: If True, skip frames marked as low priority

        Returns:
            List of GraphExtraction objects
        """
        extractions = []

        for frame in frames:
            if skip_low_priority and frame.priority == "low":
                continue

            print(f"Extracting from frame at {frame.timestamp}s...")
            extraction = self.extract_topology(frame.path, frame.timestamp)
            extractions.append(extraction)

        return extractions
