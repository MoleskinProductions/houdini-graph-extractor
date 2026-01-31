"""Output formatter for hou.data JSON format."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..analysis.visual_extractor import GraphExtraction, NodeExtraction, ConnectionExtraction
from ..analysis.transcript_parser import ActionEvent


@dataclass
class HouDataOutput:
    """Complete hou.data output structure."""

    version: str = "1.0"
    source: dict = field(default_factory=dict)
    networks: list = field(default_factory=list)
    extraction_metadata: dict = field(default_factory=dict)


class HouDataFormatter:
    """Format extraction results into hou.data JSON."""

    def __init__(self):
        self.node_grid_spacing = 1.5  # Houdini node grid spacing

    def _normalize_position(self, pos: list[float]) -> list[float]:
        """
        Convert normalized 0-100 position to Houdini coordinates.

        In Houdini, Y increases downward in the network editor, and
        nodes are typically spaced about 1.5 units apart.
        """
        # Convert from 0-100 to reasonable Houdini coordinates
        # Assuming a typical visible area of about 20x20 units
        x = (pos[0] / 100.0) * 20 - 10  # Center around 0
        y = -(pos[1] / 100.0) * 20 + 10  # Flip Y, center around 0

        return [round(x, 2), round(y, 2)]

    def _deduplicate_nodes(
        self,
        extractions: list[GraphExtraction],
    ) -> tuple[list[NodeExtraction], dict[str, float]]:
        """
        Deduplicate nodes across multiple frame extractions.

        Returns nodes with highest confidence and tracks first-seen timestamps.
        """
        node_map: dict[str, tuple[NodeExtraction, float, float]] = {}  # name -> (node, confidence, first_seen)

        for extraction in extractions:
            for node in extraction.nodes:
                existing = node_map.get(node.name)

                if existing is None:
                    node_map[node.name] = (node, extraction.extraction_confidence, extraction.timestamp)
                else:
                    # Keep the one with higher confidence or update flags
                    _, existing_conf, first_seen = existing
                    if extraction.extraction_confidence > existing_conf:
                        node_map[node.name] = (node, extraction.extraction_confidence, first_seen)
                    elif node.flags.get("display") or node.flags.get("render"):
                        # Update flags if we see them later
                        existing[0].flags.update(node.flags)

        nodes = [item[0] for item in node_map.values()]
        first_seen = {name: item[2] for name, item in node_map.items()}

        return nodes, first_seen

    def _deduplicate_connections(
        self,
        extractions: list[GraphExtraction],
    ) -> list[ConnectionExtraction]:
        """Deduplicate connections across extractions."""
        seen = set()
        connections = []

        for extraction in extractions:
            for conn in extraction.connections:
                key = (conn.from_node, conn.from_output, conn.to_node, conn.to_input)
                if key not in seen:
                    seen.add(key)
                    connections.append(conn)

        return connections

    def _find_display_render_nodes(
        self,
        nodes: list[NodeExtraction],
    ) -> tuple[str | None, str | None]:
        """Find which nodes have display and render flags."""
        display_node = None
        render_node = None

        for node in nodes:
            if node.flags.get("display"):
                display_node = node.name
            if node.flags.get("render"):
                render_node = node.name

        return display_node, render_node

    def format(
        self,
        extractions: list[GraphExtraction],
        video_url: str,
        video_title: str,
        action_events: list[ActionEvent] | None = None,
    ) -> dict:
        """
        Format extraction results into hou.data JSON structure.

        Args:
            extractions: List of graph extractions from frames
            video_url: Source YouTube URL
            video_title: Video title
            action_events: Optional list of detected action events

        Returns:
            Dictionary in hou.data format
        """
        # Filter to extractions with actual content
        valid_extractions = [e for e in extractions if e.nodes]

        if not valid_extractions:
            return {
                "version": "1.0",
                "source": {
                    "video_url": video_url,
                    "video_title": video_title,
                    "extracted_at": datetime.utcnow().isoformat() + "Z",
                },
                "networks": [],
                "extraction_metadata": {
                    "total_frames_analyzed": len(extractions),
                    "action_events_detected": len(action_events) if action_events else 0,
                    "confidence_score": 0.0,
                    "flagged_for_review": [{"reason": "No valid graph extractions found"}],
                },
            }

        # Deduplicate nodes and connections
        nodes, first_seen = self._deduplicate_nodes(valid_extractions)
        connections = self._deduplicate_connections(valid_extractions)

        # Determine network context (use most common)
        contexts = [e.network_context for e in valid_extractions if e.network_context != "unknown"]
        network_context = max(set(contexts), key=contexts.count) if contexts else "SOP"

        # Find display/render nodes
        display_node, render_node = self._find_display_render_nodes(nodes)

        # If no flags found, assume last node in chain
        if not display_node and nodes:
            display_node = nodes[-1].name
        if not render_node and nodes:
            render_node = nodes[-1].name

        # Build network structure
        network = {
            "path": "/obj/geo1",  # Default path
            "type": "geo",
            "nodes": [],
            "connections": [],
            "display_node": display_node,
            "render_node": render_node,
        }

        # Add nodes
        for node in nodes:
            network["nodes"].append({
                "name": node.name,
                "type": node.type,
                "position": self._normalize_position(node.position),
                "params": {},  # Phase 1 doesn't extract parameters
            })

        # Add connections
        for conn in connections:
            network["connections"].append([
                conn.from_node,
                conn.from_output,
                conn.to_node,
                conn.to_input,
            ])

        # Calculate overall confidence
        confidences = [e.extraction_confidence for e in valid_extractions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Collect flagged items
        flagged = []
        for extraction in valid_extractions:
            for uncertain in extraction.uncertain_elements:
                flagged.append({
                    "element": uncertain,
                    "timestamp": extraction.timestamp,
                })

        # Build timeline if action events provided
        timeline = []
        if action_events:
            for event in action_events:
                timeline.append({
                    "time": event.timestamp,
                    "action": event.event_type.value,
                    "raw_text": event.raw_text,
                })

        return {
            "version": "1.0",
            "source": {
                "video_url": video_url,
                "video_title": video_title,
                "extracted_at": datetime.utcnow().isoformat() + "Z",
            },
            "networks": [network],
            "extraction_metadata": {
                "total_frames_analyzed": len(extractions),
                "valid_extractions": len(valid_extractions),
                "action_events_detected": len(action_events) if action_events else 0,
                "confidence_score": round(avg_confidence, 2),
                "flagged_for_review": flagged,
                "timeline": timeline if timeline else None,
            },
        }

    def save(self, data: dict, output_path: Path) -> None:
        """Save hou.data to JSON file."""
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def to_json(self, data: dict) -> str:
        """Convert to JSON string."""
        return json.dumps(data, indent=2)
