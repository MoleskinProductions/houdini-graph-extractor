"""HouGraph IR export adapter for the video extraction pipeline.

This module converts extraction results from the visual extractor into
the canonical HouGraph IR format for interchange with other tools.
"""

from datetime import datetime, timezone
from pathlib import Path

from ..analysis.visual_extractor import GraphExtraction, NodeExtraction, ConnectionExtraction
from ..analysis.transcript_parser import ActionEvent

# Import HouGraph IR - handle case where it's not installed
try:
    from hougraph_ir import (
        HouGraphIR,
        NodeDefinition,
        Connection,
        Position,
        NetworkContext,
        ParameterValue,
    )
    HOUGRAPH_IR_AVAILABLE = True
except ImportError:
    HOUGRAPH_IR_AVAILABLE = False
    HouGraphIR = None
    NodeDefinition = None
    Connection = None
    Position = None
    NetworkContext = None
    ParameterValue = None


def _check_hougraph_ir():
    """Raise an error if hougraph-ir is not installed."""
    if not HOUGRAPH_IR_AVAILABLE:
        raise ImportError(
            "hougraph-ir package is not installed. "
            "Install it with: pip install hougraph-ir"
        )


def _map_network_context(context_str: str) -> "NetworkContext":
    """Map extraction context string to NetworkContext enum."""
    context_map = {
        "SOP": NetworkContext.SOP,
        "sop": NetworkContext.SOP,
        "DOP": NetworkContext.DOP,
        "dop": NetworkContext.DOP,
        "VOP": NetworkContext.VOP,
        "vop": NetworkContext.VOP,
        "OBJ": NetworkContext.OBJ,
        "obj": NetworkContext.OBJ,
        "SHOP": NetworkContext.SHOP,
        "shop": NetworkContext.SHOP,
        "COP": NetworkContext.COP,
        "cop": NetworkContext.COP,
        "TOP": NetworkContext.TOP,
        "top": NetworkContext.TOP,
        "LOP": NetworkContext.LOP,
        "lop": NetworkContext.LOP,
        "CHOP": NetworkContext.CHOP,
        "chop": NetworkContext.CHOP,
    }
    return context_map.get(context_str, NetworkContext.SOP)


class HouGraphIRExporter:
    """Export extraction results to HouGraph IR format."""

    def __init__(self):
        _check_hougraph_ir()
        self.node_grid_spacing = 1.5

    def _normalize_position(self, pos: list[float]) -> Position:
        """
        Convert normalized 0-100 position to Houdini coordinates.

        In Houdini, Y increases downward in the network editor, and
        nodes are typically spaced about 1.5 units apart.
        """
        x = (pos[0] / 100.0) * 20 - 10
        y = -(pos[1] / 100.0) * 20 + 10
        return Position(x=round(x, 2), y=round(y, 2))

    def _deduplicate_nodes(
        self,
        extractions: list[GraphExtraction],
    ) -> tuple[list[NodeExtraction], dict[str, tuple[float, float]]]:
        """
        Deduplicate nodes across multiple frame extractions.

        Returns:
            Tuple of (deduplicated nodes, dict mapping name -> (confidence, first_seen_timestamp))
        """
        node_map: dict[str, tuple[NodeExtraction, float, float]] = {}

        for extraction in extractions:
            for node in extraction.nodes:
                existing = node_map.get(node.name)

                if existing is None:
                    node_map[node.name] = (node, extraction.extraction_confidence, extraction.timestamp)
                else:
                    _, existing_conf, first_seen = existing
                    if extraction.extraction_confidence > existing_conf:
                        node_map[node.name] = (node, extraction.extraction_confidence, first_seen)
                    elif node.flags.get("display") or node.flags.get("render"):
                        existing[0].flags.update(node.flags)

        nodes = [item[0] for item in node_map.values()]
        metadata = {name: (item[1], item[2]) for name, item in node_map.items()}
        return nodes, metadata

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

    def _convert_node(
        self,
        node: NodeExtraction,
        confidence: float,
        first_seen: float,
    ) -> NodeDefinition:
        """Convert a NodeExtraction to NodeDefinition."""
        node_def = NodeDefinition(
            name=node.name,
            type=node.type,
            position=self._normalize_position(node.position),
            display_flag=node.flags.get("display", False),
            render_flag=node.flags.get("render", False),
            bypass_flag=node.flags.get("bypass", False),
        )
        # Store extraction metadata as private attributes
        node_def._extraction_confidence = confidence
        node_def._source_timestamp = first_seen
        return node_def

    def _convert_connection(self, conn: ConnectionExtraction) -> Connection:
        """Convert a ConnectionExtraction to Connection."""
        return Connection(
            source_node=conn.from_node,
            source_output=conn.from_output,
            dest_node=conn.to_node,
            dest_input=conn.to_input,
        )

    def export(
        self,
        extractions: list[GraphExtraction],
        video_url: str | None = None,
        video_title: str | None = None,
        action_events: list[ActionEvent] | None = None,
        parent_path: str = "/obj/geo1",
    ) -> HouGraphIR:
        """
        Export extraction results to HouGraph IR format.

        Args:
            extractions: List of graph extractions from frames
            video_url: Source YouTube URL
            video_title: Video title
            action_events: Optional list of detected action events
            parent_path: Houdini parent path for the network

        Returns:
            HouGraphIR object representing the extracted network
        """
        # Filter to extractions with actual content
        valid_extractions = [e for e in extractions if e.nodes]

        if not valid_extractions:
            return HouGraphIR(
                parent_path=parent_path,
                context=NetworkContext.SOP,
                nodes=[],
                connections=[],
                source={
                    "type": "youtube",
                    "url": video_url,
                    "title": video_title,
                } if video_url else None,
                extraction_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

        # Deduplicate
        nodes, node_metadata = self._deduplicate_nodes(valid_extractions)
        connections = self._deduplicate_connections(valid_extractions)

        # Determine network context
        contexts = [e.network_context for e in valid_extractions if e.network_context != "unknown"]
        network_context = max(set(contexts), key=contexts.count) if contexts else "SOP"

        # Convert nodes
        ir_nodes = []
        for node in nodes:
            confidence, first_seen = node_metadata.get(node.name, (0.5, 0.0))
            ir_nodes.append(self._convert_node(node, confidence, first_seen))

        # Convert connections
        ir_connections = [self._convert_connection(c) for c in connections]

        # Calculate overall confidence
        confidences = [e.extraction_confidence for e in valid_extractions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Build source metadata (includes extraction stats)
        source_metadata = None
        if video_url:
            source_metadata = {
                "type": "youtube",
                "url": video_url,
                "title": video_title,
                "total_frames_analyzed": len(extractions),
                "valid_extractions": len(valid_extractions),
                "extraction_confidence": round(avg_confidence, 2),
            }

        # Build transcript events from action events
        transcript_events = None
        if action_events:
            transcript_events = [
                {
                    "timestamp": event.timestamp,
                    "event_type": event.event_type.value,
                    "raw_text": event.raw_text,
                }
                for event in action_events
            ]

        # Build HouGraph IR
        return HouGraphIR(
            schema_version="1.0.0",
            parent_path=parent_path,
            context=_map_network_context(network_context),
            nodes=ir_nodes,
            connections=ir_connections,
            source=source_metadata,
            extraction_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            transcript_events=transcript_events,
        )

    def export_to_dict(
        self,
        extractions: list[GraphExtraction],
        video_url: str | None = None,
        video_title: str | None = None,
        action_events: list[ActionEvent] | None = None,
        parent_path: str = "/obj/geo1",
    ) -> dict:
        """Export to dictionary format."""
        ir = self.export(extractions, video_url, video_title, action_events, parent_path)
        return ir.to_dict()

    def export_to_json(
        self,
        extractions: list[GraphExtraction],
        video_url: str | None = None,
        video_title: str | None = None,
        action_events: list[ActionEvent] | None = None,
        parent_path: str = "/obj/geo1",
        indent: int = 2,
    ) -> str:
        """Export to JSON string."""
        ir = self.export(extractions, video_url, video_title, action_events, parent_path)
        return ir.to_json(indent=indent)

    def save(
        self,
        extractions: list[GraphExtraction],
        output_path: Path,
        video_url: str | None = None,
        video_title: str | None = None,
        action_events: list[ActionEvent] | None = None,
        parent_path: str = "/obj/geo1",
    ) -> None:
        """Save HouGraph IR to JSON file."""
        ir = self.export(extractions, video_url, video_title, action_events, parent_path)
        ir.save(output_path)


def export_extractions_to_hougraph_ir(
    extractions: list[GraphExtraction],
    video_url: str | None = None,
    video_title: str | None = None,
) -> "HouGraphIR":
    """
    Convenience function to export extractions to HouGraph IR.

    Args:
        extractions: List of graph extractions
        video_url: Optional source URL
        video_title: Optional video title

    Returns:
        HouGraphIR object
    """
    exporter = HouGraphIRExporter()
    return exporter.export(extractions, video_url, video_title)


def export_extractions_to_dict(
    extractions: list[GraphExtraction],
    video_url: str | None = None,
    video_title: str | None = None,
) -> dict:
    """
    Convenience function to export extractions to HouGraph IR dict format.

    Args:
        extractions: List of graph extractions
        video_url: Optional source URL
        video_title: Optional video title

    Returns:
        Dictionary in HouGraph IR format
    """
    exporter = HouGraphIRExporter()
    return exporter.export_to_dict(extractions, video_url, video_title)
