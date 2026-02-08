"""Graph state manager for maintaining canonical graph state with temporal metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator

from .models import (
    NodeState,
    ConnectionState,
    ConflictRecord,
    EnhancedActionEvent,
    SourceType,
    SourceRecord,
)

if TYPE_CHECKING:
    from ..analysis.validator import StructuralValidator


@dataclass
class GraphStateManager:
    """
    Maintains canonical graph state with temporal metadata.

    Tracks nodes and connections over time, merging extractions from
    visual and transcript sources with confidence tracking.
    """

    # Core state
    nodes: dict[str, NodeState] = field(default_factory=dict)
    connections: dict[tuple, ConnectionState] = field(default_factory=dict)

    # Network metadata
    network_context: str = "SOP"
    parent_path: str | None = None

    # Conflict tracking
    conflicts: list[ConflictRecord] = field(default_factory=list)

    # Timeline tracking
    processed_timestamps: list[float] = field(default_factory=list)
    transcript_events: list[EnhancedActionEvent] = field(default_factory=list)

    # Processing state
    current_timestamp: float = 0.0

    # Structural validator (Phase 1A/2A) — optional
    validator: StructuralValidator | None = None

    # Context tracking for parameter association
    # Tracks which node is currently being discussed/manipulated
    current_context_node: str | None = None
    context_node_timestamp: float = 0.0
    context_decay_seconds: float = 10.0  # How long context stays valid

    def add_visual_extraction(
        self,
        nodes: list[dict],
        connections: list[dict],
        timestamp: float,
        extraction_confidence: float = 0.7,
        network_context: str | None = None,
        parent_path: str | None = None,
    ) -> dict:
        """
        Add a visual extraction to the graph state.

        Args:
            nodes: List of node dictionaries from visual extraction
            connections: List of connection dictionaries
            timestamp: Timestamp of the extraction
            extraction_confidence: Overall extraction confidence
            network_context: Network context (SOP, DOP, etc.)
            parent_path: Parent network path

        Returns:
            Dict with counts of new/updated elements
        """
        self.current_timestamp = timestamp
        self.processed_timestamps.append(timestamp)

        if network_context:
            self.network_context = network_context
        if parent_path:
            self.parent_path = parent_path

        stats = {
            "new_nodes": 0,
            "updated_nodes": 0,
            "new_connections": 0,
            "updated_connections": 0,
        }

        # Decay confidence for nodes not seen recently
        for node in self.nodes.values():
            node.decay_confidence()

        # Process nodes
        for node_data in nodes:
            name = node_data.get("name", "unknown")
            node_type = node_data.get("type", "unknown")
            position = node_data.get("position", [0, 0])
            flags = node_data.get("flags", {})

            # Validate / resolve node type via structural validator
            validation_status = ""
            resolved_schema_key = None
            if self.validator:
                vr = self.validator.validate_node_type(
                    node_type, context_hint=self.network_context,
                )
                node_type = vr.resolved_type
                extraction_confidence = max(0.0, min(1.0, extraction_confidence + vr.confidence_adjustment))
                validation_status = vr.status.value
                resolved_schema_key = vr.resolved_key

            if name in self.nodes:
                # Update existing node
                self.nodes[name].update_from_visual(
                    timestamp=timestamp,
                    confidence=extraction_confidence,
                    position=position,
                    flags=flags,
                )
                if validation_status:
                    self.nodes[name].validation_status = validation_status
                if resolved_schema_key:
                    self.nodes[name].resolved_schema_key = resolved_schema_key
                stats["updated_nodes"] += 1
            else:
                # Create new node
                self.nodes[name] = NodeState(
                    name=name,
                    type=node_type,
                    position=position,
                    flags=flags,
                    confidence=extraction_confidence,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    sources=[SourceRecord(
                        source_type=SourceType.VISUAL,
                        timestamp=timestamp,
                        confidence=extraction_confidence,
                    )],
                    validation_status=validation_status,
                    resolved_schema_key=resolved_schema_key,
                )
                stats["new_nodes"] += 1

        # Process connections
        for conn_data in connections:
            from_node = conn_data.get("from_node", "")
            from_output = conn_data.get("from_output", 0)
            to_node = conn_data.get("to_node", "")
            to_input = conn_data.get("to_input", 0)

            # Validate connection against pattern corpus
            conn_pattern_count = 0
            conn_validation_status = ""
            conn_confidence = extraction_confidence
            if self.validator:
                from_type = self.nodes[from_node].type if from_node in self.nodes else ""
                to_type = self.nodes[to_node].type if to_node in self.nodes else ""
                if from_type and to_type:
                    cr = self.validator.validate_connection(
                        from_type, to_type, from_output, to_input,
                    )
                    conn_confidence = max(0.0, min(1.0, conn_confidence + cr.confidence_adjustment))
                    conn_pattern_count = cr.pattern_count
                    conn_validation_status = "known" if cr.known_pattern else "unknown"

            key = (from_node, from_output, to_node, to_input)

            if key in self.connections:
                # Update existing connection
                self.connections[key].update_observation(timestamp, conn_confidence)
                if conn_pattern_count:
                    self.connections[key].pattern_count = conn_pattern_count
                if conn_validation_status:
                    self.connections[key].validation_status = conn_validation_status
                stats["updated_connections"] += 1
            else:
                # Create new connection
                self.connections[key] = ConnectionState(
                    from_node=from_node,
                    from_output=from_output,
                    to_node=to_node,
                    to_input=to_input,
                    confidence=conn_confidence,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    sources=[SourceRecord(
                        source_type=SourceType.VISUAL,
                        timestamp=timestamp,
                        confidence=conn_confidence,
                    )],
                    pattern_count=conn_pattern_count,
                    validation_status=conn_validation_status,
                )
                stats["new_connections"] += 1

        return stats

    def add_transcript_hint(
        self,
        event: EnhancedActionEvent,
        time_window: float = 3.0,
    ) -> bool:
        """
        Add a transcript hint and corroborate with visual extractions.

        Args:
            event: Enhanced action event from transcript analysis
            time_window: Time window (seconds) for matching visual extractions

        Returns:
            True if corroboration was found
        """
        self.transcript_events.append(event)

        corroborated = False

        # Try to match with existing nodes
        if event.node_type or event.node_name:
            search_term = event.node_name or event.node_type

            for name, node in self.nodes.items():
                # Check if node was seen within time window
                if abs(node.last_seen - event.timestamp) > time_window:
                    continue

                # Match by name or type
                if self._fuzzy_match(search_term, name) or self._fuzzy_match(search_term, node.type):
                    node.corroborate_with_transcript(event.timestamp)
                    event.linked_visual_timestamp = node.last_seen
                    event.visual_match_confidence = 0.8
                    corroborated = True
                    break

        return corroborated

    def update_context_node(self, node_name: str | None, timestamp: float) -> None:
        """
        Update the current context node being discussed.

        This tracks which node is currently being manipulated so that
        param_change events can be associated with the correct node.

        Args:
            node_name: Name of the node being discussed (or None to clear)
            timestamp: When this context was established
        """
        if node_name:
            self.current_context_node = node_name
            self.context_node_timestamp = timestamp
        else:
            self.current_context_node = None

    def get_context_node(self, timestamp: float) -> str | None:
        """
        Get the current context node if still valid.

        Args:
            timestamp: Current timestamp

        Returns:
            Node name if context is still valid, else None
        """
        if not self.current_context_node:
            return None

        # Check if context has decayed
        if timestamp - self.context_node_timestamp > self.context_decay_seconds:
            return None

        return self.current_context_node

    def apply_transcript_params(
        self,
        events: list[EnhancedActionEvent],
        time_window: float = 5.0,
    ) -> int:
        """
        Apply parameter changes from transcript events to nodes.

        Uses context tracking to associate param_change events with the
        node that is currently being discussed/manipulated.

        Args:
            events: Transcript events to process
            time_window: Time window for matching nodes

        Returns:
            Number of parameters applied
        """
        params_applied = 0

        for event in events:
            # Update context when nodes are mentioned
            if event.event_type in ["node_create", "selection", "context_switch"]:
                # Determine the context node
                context_node = event.node_name or self._find_node_by_type(event.node_type)
                if context_node:
                    self.update_context_node(context_node, event.timestamp)

            # Apply parameters to context node
            if event.event_type == "param_change" and event.param_name:
                # Try to find the target node
                target_node = self._determine_param_target(event, time_window)

                if target_node and target_node in self.nodes:
                    self.nodes[target_node].apply_param(
                        event.param_name,
                        event.param_value,
                        event.timestamp,
                    )
                    params_applied += 1

                    # Update context to this node
                    self.update_context_node(target_node, event.timestamp)

        return params_applied

    def _determine_param_target(
        self,
        event: EnhancedActionEvent,
        time_window: float,
    ) -> str | None:
        """
        Determine which node a param_change event should target.

        Priority:
        1. Explicit node_name in event
        2. Match node by type if specified
        3. Current context node
        4. Most recently seen node matching the type

        Args:
            event: The param_change event
            time_window: Time window for matching

        Returns:
            Node name or None if no match found
        """
        # 1. Explicit node name
        if event.node_name and event.node_name in self.nodes:
            return event.node_name

        # 2. Find by type
        if event.node_type:
            match = self._find_node_by_type(event.node_type, event.timestamp, time_window)
            if match:
                return match

        # 3. Current context node
        context = self.get_context_node(event.timestamp)
        if context:
            return context

        # 4. Most recently active node
        return self._get_most_recent_node(event.timestamp, time_window)

    def _find_node_by_type(
        self,
        node_type: str | None,
        timestamp: float | None = None,
        time_window: float = 5.0,
    ) -> str | None:
        """Find a node by type, preferring recently seen nodes."""
        if not node_type:
            return None

        node_type_lower = node_type.lower()
        candidates = []

        for name, node in self.nodes.items():
            if self._fuzzy_match(node_type_lower, node.type):
                # Score by recency if timestamp provided
                if timestamp is not None:
                    time_diff = abs(node.last_seen - timestamp)
                    if time_diff <= time_window:
                        candidates.append((name, time_diff))
                else:
                    candidates.append((name, 0))

        if candidates:
            # Return the most recently seen match
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

        return None

    def _get_most_recent_node(self, timestamp: float, time_window: float) -> str | None:
        """Get the most recently seen node within time window."""
        best_node = None
        best_time_diff = time_window + 1

        for name, node in self.nodes.items():
            time_diff = abs(node.last_seen - timestamp)
            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_node = name

        return best_node if best_time_diff <= time_window else None

    def _fuzzy_match(self, search: str | None, target: str) -> bool:
        """Fuzzy string matching for node names/types."""
        if not search or not target:
            return False

        # Delegate to structural validator when available
        if self.validator:
            if self.validator.types_match(search, target, self.network_context):
                return True

        search = search.lower().strip()
        target = target.lower().strip()

        # Direct match
        if search == target:
            return True

        # Substring match
        if search in target or target in search:
            return True

        # Handle common variations
        # e.g., "scatter" matches "scatter1", "scatter_1", etc.
        if search.rstrip("0123456789_") == target.rstrip("0123456789_"):
            return True

        return False

    def record_conflict(self, conflict: ConflictRecord) -> None:
        """Record a detected conflict."""
        self.conflicts.append(conflict)

    def get_nodes(self, min_confidence: float = 0.0) -> Iterator[NodeState]:
        """Get all nodes above minimum confidence threshold."""
        for node in self.nodes.values():
            if node.confidence >= min_confidence:
                yield node

    def get_connections(self, min_confidence: float = 0.0) -> Iterator[ConnectionState]:
        """Get all connections above minimum confidence threshold."""
        for conn in self.connections.values():
            if conn.confidence >= min_confidence:
                yield conn

    def get_node(self, name: str) -> NodeState | None:
        """Get a specific node by name."""
        return self.nodes.get(name)

    def get_display_node(self) -> str | None:
        """Get the node with display flag set (most recent if multiple)."""
        display_nodes = [
            (n.name, n.last_seen)
            for n in self.nodes.values()
            if n.flags.get("display", False)
        ]
        if display_nodes:
            # Return most recently seen display node
            return max(display_nodes, key=lambda x: x[1])[0]
        return None

    def get_render_node(self) -> str | None:
        """Get the node with render flag set (most recent if multiple)."""
        render_nodes = [
            (n.name, n.last_seen)
            for n in self.nodes.values()
            if n.flags.get("render", False)
        ]
        if render_nodes:
            return max(render_nodes, key=lambda x: x[1])[0]
        return None

    def get_average_confidence(self) -> float:
        """Get average confidence across all nodes."""
        if not self.nodes:
            return 0.0
        return sum(n.confidence for n in self.nodes.values()) / len(self.nodes)

    def to_hou_data(self, include_metadata: bool = True) -> dict:
        """
        Convert state to hou_data format output.

        Args:
            include_metadata: Whether to include confidence and timing metadata

        Returns:
            Dictionary in hou_data format
        """
        # Build nodes list
        nodes_output = []
        for node in sorted(self.nodes.values(), key=lambda n: n.first_seen):
            node_data = {
                "name": node.name,
                "type": node.type,
                "position": self._normalize_position(node.position),
                "params": node.get_params_flat(),
            }
            if include_metadata:
                node_data["_metadata"] = {
                    "confidence": round(node.confidence, 2),
                    "first_seen": round(node.first_seen, 2),
                    "last_seen": round(node.last_seen, 2),
                    "observation_count": node.observation_count,
                    "transcript_corroborated": node.transcript_corroborated,
                }
            nodes_output.append(node_data)

        # Build connections list
        connections_output = []
        for conn in sorted(self.connections.values(), key=lambda c: c.first_seen):
            conn_data = [conn.from_node, conn.from_output, conn.to_node, conn.to_input]
            connections_output.append(conn_data)

        # Build network structure
        network = {
            "path": self.parent_path or f"/obj/geo1",
            "type": self._context_to_network_type(self.network_context),
            "nodes": nodes_output,
            "connections": connections_output,
            "display_node": self.get_display_node(),
            "render_node": self.get_render_node(),
        }

        result = {
            "networks": [network],
        }

        if include_metadata:
            # Add confidence metadata per connection
            result["_connection_metadata"] = [
                {
                    "connection": conn.key,
                    "confidence": round(conn.confidence, 2),
                    "observation_count": conn.observation_count,
                }
                for conn in self.connections.values()
            ]

            # Add conflict summary
            if self.conflicts:
                result["_conflicts"] = [c.to_dict() for c in self.conflicts]

        return result

    def _normalize_position(self, pos: list[float]) -> list[float]:
        """
        Normalize position from 0-100 space to Houdini coordinates.

        Houdini uses a coordinate system where:
        - X increases to the right
        - Y increases downward in network view
        """
        if not pos or len(pos) < 2:
            return [0.0, 0.0]

        # Convert 0-100 normalized to roughly -10 to 10 range
        x = (pos[0] / 100.0) * 20 - 10
        y = -((pos[1] / 100.0) * 20 - 10)  # Flip Y axis

        return [round(x, 1), round(y, 1)]

    def _context_to_network_type(self, context: str) -> str:
        """Convert network context to network type."""
        context_map = {
            "SOP": "geo",
            "DOP": "dopnet",
            "VOP": "vopnet",
            "OBJ": "obj",
            "SHOP": "shop",
            "COP": "cop",
            "TOP": "topnet",
            "LOP": "lopnet",
        }
        return context_map.get(context, "geo")

    def get_stats(self) -> dict:
        """Get statistics about the current state."""
        # Count total extracted parameters
        total_params = sum(len(n.params) for n in self.nodes.values())
        nodes_with_params = sum(1 for n in self.nodes.values() if n.params)

        return {
            "total_nodes": len(self.nodes),
            "total_connections": len(self.connections),
            "frames_processed": len(self.processed_timestamps),
            "transcript_events": len(self.transcript_events),
            "conflicts_detected": len(self.conflicts),
            "average_confidence": round(self.get_average_confidence(), 2),
            "corroborated_nodes": sum(
                1 for n in self.nodes.values() if n.transcript_corroborated
            ),
            "total_params_extracted": total_params,
            "nodes_with_params": nodes_with_params,
        }
