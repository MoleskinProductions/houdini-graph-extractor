"""State merger for incremental merge strategies with conflict handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .models import ConflictRecord, EnhancedActionEvent
from .graph_state import GraphStateManager

if TYPE_CHECKING:
    from ..analysis.validator import StructuralValidator
    from ..analysis.visual_extractor import GraphExtraction


@dataclass
class MergeConfig:
    """Configuration for merge operations."""

    # Confidence thresholds
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.3

    # Position change detection
    position_change_threshold: float = 5.0  # In 0-100 coordinate space

    # Corroboration boost
    transcript_boost: float = 0.15

    # Confidence decay
    decay_rate: float = 0.05
    decay_after_frames: int = 3

    # Time windows
    transcript_window_seconds: float = 3.0


@dataclass
class MergeResult:
    """Result of a merge operation."""

    new_nodes: int = 0
    updated_nodes: int = 0
    new_connections: int = 0
    updated_connections: int = 0
    corroborations: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    params_applied: int = 0  # Number of parameters extracted from transcript


class StateMerger:
    """
    Incremental state merger with conflict handling.

    Handles merging visual extractions and transcript hints into
    the canonical graph state with confidence tracking and
    conflict resolution.
    """

    def __init__(
        self,
        state: GraphStateManager,
        config: MergeConfig | None = None,
        validator: StructuralValidator | None = None,
    ):
        """
        Initialize the merger.

        Args:
            state: The GraphStateManager to merge into
            config: Optional merge configuration
            validator: Optional structural validator for schema-aware alias matching
        """
        self.state = state
        self.config = config or MergeConfig()
        self.validator = validator

    def merge_visual_extraction(
        self,
        extraction: GraphExtraction,
        transcript_events: list[EnhancedActionEvent] | None = None,
    ) -> MergeResult:
        """
        Merge a visual extraction into the graph state.

        Args:
            extraction: GraphExtraction from visual extractor
            transcript_events: Optional relevant transcript events for corroboration

        Returns:
            MergeResult with merge statistics
        """
        result = MergeResult()

        if not extraction.graph_visible or not extraction.nodes:
            return result

        # Convert extraction to dict format for state manager
        nodes_data = [
            {
                "name": n.name,
                "type": n.type,
                "position": n.position,
                "flags": n.flags,
            }
            for n in extraction.nodes
        ]

        connections_data = [
            {
                "from_node": c.from_node,
                "from_output": c.from_output,
                "to_node": c.to_node,
                "to_input": c.to_input,
            }
            for c in extraction.connections
        ]

        # Merge into state
        stats = self.state.add_visual_extraction(
            nodes=nodes_data,
            connections=connections_data,
            timestamp=extraction.timestamp,
            extraction_confidence=extraction.extraction_confidence,
            network_context=extraction.network_context,
            parent_path=extraction.parent_path,
        )

        result.new_nodes = stats["new_nodes"]
        result.updated_nodes = stats["updated_nodes"]
        result.new_connections = stats["new_connections"]
        result.updated_connections = stats["updated_connections"]

        # Apply transcript corroboration if events provided
        if transcript_events:
            for event in transcript_events:
                if self._is_within_window(event.timestamp, extraction.timestamp):
                    if self.state.add_transcript_hint(event, self.config.transcript_window_seconds):
                        result.corroborations += 1

            # Apply parameter changes from transcript events
            result.params_applied = self.state.apply_transcript_params(
                transcript_events,
                time_window=self.config.transcript_window_seconds,
            )

        return result

    def merge_transcript_events(
        self,
        events: list[EnhancedActionEvent],
    ) -> MergeResult:
        """
        Merge transcript events and check for corroboration.

        Args:
            events: List of enhanced action events

        Returns:
            MergeResult with corroboration statistics
        """
        result = MergeResult()

        for event in events:
            if self.state.add_transcript_hint(event, self.config.transcript_window_seconds):
                result.corroborations += 1

        return result

    def check_position_change(
        self,
        node_name: str,
        new_position: list[float],
    ) -> bool:
        """
        Check if a node's position has changed significantly.

        Args:
            node_name: Name of the node
            new_position: New position [x, y] in 0-100 space

        Returns:
            True if position changed beyond threshold
        """
        node = self.state.get_node(node_name)
        if not node or not node.position:
            return False

        dx = abs(new_position[0] - node.position[0])
        dy = abs(new_position[1] - node.position[1])

        return dx > self.config.position_change_threshold or dy > self.config.position_change_threshold

    def detect_conflicts(
        self,
        extraction: GraphExtraction,
        transcript_events: list[EnhancedActionEvent],
    ) -> list[ConflictRecord]:
        """
        Detect conflicts between visual extraction and transcript.

        Args:
            extraction: Visual extraction
            transcript_events: Relevant transcript events

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Get nodes mentioned in transcript within time window
        transcript_nodes = {}
        for event in transcript_events:
            if not self._is_within_window(event.timestamp, extraction.timestamp):
                continue
            if event.node_type:
                transcript_nodes[event.node_type.lower()] = event

        # Get nodes from visual extraction
        visual_nodes = {n.name.lower(): n for n in extraction.nodes}
        visual_types = {n.type.lower(): n for n in extraction.nodes}

        # Check for NODE_TYPE_MISMATCH
        for transcript_type, event in transcript_nodes.items():
            # Check if transcript type matches any visual node type
            if transcript_type not in visual_types:
                # Check for aliases
                if not self._is_type_alias(transcript_type, set(visual_types.keys())):
                    # Node mentioned in transcript but not seen visually
                    conflict = ConflictRecord(
                        conflict_type="NODE_MISSING_VISUAL",
                        timestamp=extraction.timestamp,
                        visual_data={"nodes": list(visual_types.keys())},
                        transcript_data={"expected_type": transcript_type, "raw_text": event.raw_text},
                    )
                    conflicts.append(conflict)

        # Check for EXTRA_VISUAL_NODE (nodes in visual not mentioned in transcript)
        # This is a softer conflict - just log it
        if transcript_nodes:
            for visual_type in visual_types.keys():
                found_match = False
                for transcript_type in transcript_nodes.keys():
                    if self._is_type_alias(visual_type, {transcript_type}):
                        found_match = True
                        break
                if not found_match and visual_type not in ["null", "output", "unknown"]:
                    conflict = ConflictRecord(
                        conflict_type="EXTRA_VISUAL_NODE",
                        timestamp=extraction.timestamp,
                        visual_data={"node_type": visual_type},
                        transcript_data={"mentioned_types": list(transcript_nodes.keys())},
                    )
                    conflicts.append(conflict)

        # Record conflicts in state
        for conflict in conflicts:
            self.state.record_conflict(conflict)

        return conflicts

    def resolve_conflicts(self, conflicts: list[ConflictRecord]) -> int:
        """
        Attempt to resolve detected conflicts.

        Args:
            conflicts: List of conflicts to resolve

        Returns:
            Number of conflicts resolved
        """
        resolved = 0

        for conflict in conflicts:
            if conflict.conflict_type == "NODE_MISSING_VISUAL":
                # Trust high-confidence visual - the node might not be visible yet
                conflict.resolution = "DEFERRED"
                conflict.resolution_confidence = 0.5
                resolved += 1

            elif conflict.conflict_type == "EXTRA_VISUAL_NODE":
                # Accept extra visual nodes with confidence penalty
                conflict.resolution = "ACCEPTED_WITH_PENALTY"
                conflict.resolution_confidence = 0.7
                resolved += 1

            elif conflict.conflict_type == "NODE_TYPE_MISMATCH":
                # Try alias resolution
                visual_type = conflict.visual_data.get("node_type") if conflict.visual_data else None
                transcript_type = conflict.transcript_data.get("expected_type") if conflict.transcript_data else None

                if visual_type and transcript_type:
                    if self._is_type_alias(visual_type, {transcript_type}):
                        conflict.resolution = "ALIAS_MATCH"
                        conflict.resolution_confidence = 0.9
                        resolved += 1
                    else:
                        # Trust visual with high confidence
                        conflict.resolution = "TRUST_VISUAL"
                        conflict.resolution_confidence = 0.6

        return resolved

    def _is_within_window(self, event_time: float, extraction_time: float) -> bool:
        """Check if event is within time window of extraction."""
        return abs(event_time - extraction_time) <= self.config.transcript_window_seconds

    def _is_type_alias(self, type1: str, type_set: set[str]) -> bool:
        """Check if type1 is an alias for any type in type_set."""
        type1_lower = type1.lower()

        # Direct match
        if type1_lower in type_set:
            return True

        # Delegate to structural validator when available
        if self.validator:
            for t in type_set:
                if self.validator.types_match(type1, t, self.state.network_context):
                    return True
            return False

        # Common Houdini node type aliases (fallback)
        aliases = {
            "scatter": {"scatter", "scatter sop"},
            "copytopoints": {"copytopoints", "copy to points", "copy"},
            "attribwrangle": {"attribwrangle", "wrangle", "attrib wrangle", "attribute wrangle"},
            "xform": {"xform", "transform"},
            "object_merge": {"object_merge", "objectmerge", "objmerge"},
            "rbdbulletsolver": {"rbdbulletsolver", "rbd", "bullet", "rigid body"},
            "rbdmaterialfracture": {"rbdmaterialfracture", "fracture", "material fracture"},
        }

        # Check alias groups
        for canonical, alias_group in aliases.items():
            if type1_lower in alias_group or type1_lower == canonical:
                # Check if any type in type_set is in this alias group
                for t in type_set:
                    if t in alias_group or t == canonical:
                        return True

        return False

    def apply_confidence_decay(self) -> int:
        """
        Apply confidence decay to nodes not seen recently.

        Returns:
            Number of nodes that had confidence decayed
        """
        decayed = 0
        for node in self.state.nodes.values():
            if node.frames_since_seen >= self.config.decay_after_frames:
                old_confidence = node.confidence
                node.decay_confidence(self.config.decay_rate)
                if node.confidence < old_confidence:
                    decayed += 1
        return decayed

    def get_high_confidence_nodes(self) -> list[str]:
        """Get names of nodes with high confidence."""
        return [
            name for name, node in self.state.nodes.items()
            if node.confidence >= self.config.high_confidence_threshold
        ]

    def get_low_confidence_nodes(self) -> list[str]:
        """Get names of nodes with low confidence."""
        return [
            name for name, node in self.state.nodes.items()
            if node.confidence <= self.config.low_confidence_threshold
        ]

    def get_corroborated_nodes(self) -> list[str]:
        """Get names of nodes corroborated by transcript."""
        return [
            name for name, node in self.state.nodes.items()
            if node.transcript_corroborated
        ]
