"""Harmonizer for cross-validating transcript and visual extractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..state.models import EnhancedActionEvent, ConflictRecord
from ..state.graph_state import GraphStateManager
from .visual_extractor import GraphExtraction
from .transcript_parser import HOUDINI_NODE_ALIASES

if TYPE_CHECKING:
    pass


class ConflictType(str, Enum):
    """Types of conflicts between transcript and visual."""

    NODE_TYPE_MISMATCH = "NODE_TYPE_MISMATCH"
    NODE_MISSING_VISUAL = "NODE_MISSING_VISUAL"
    EXTRA_VISUAL_NODE = "EXTRA_VISUAL_NODE"
    CONNECTION_MISMATCH = "CONNECTION_MISMATCH"
    POSITION_DRIFT = "POSITION_DRIFT"


class ResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts."""

    TRUST_VISUAL = "TRUST_VISUAL"
    TRUST_TRANSCRIPT = "TRUST_TRANSCRIPT"
    ALIAS_MATCH = "ALIAS_MATCH"
    DEFER = "DEFER"
    ACCEPT_WITH_PENALTY = "ACCEPT_WITH_PENALTY"
    REJECT = "REJECT"


@dataclass
class HarmonizationResult:
    """Result of harmonizing visual and transcript data."""

    timestamp: float
    visual_nodes: int = 0
    visual_connections: int = 0
    transcript_events: int = 0
    corroborations: int = 0
    conflicts: list[ConflictRecord] = field(default_factory=list)
    resolutions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "visual_nodes": self.visual_nodes,
            "visual_connections": self.visual_connections,
            "transcript_events": self.transcript_events,
            "corroborations": self.corroborations,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "resolutions": self.resolutions,
        }


@dataclass
class HarmonizerConfig:
    """Configuration for harmonization."""

    # Time windows
    transcript_window_before: float = 3.0
    transcript_window_after: float = 1.0

    # Confidence thresholds
    high_visual_confidence: float = 0.8
    low_visual_confidence: float = 0.4

    # Matching thresholds
    fuzzy_match_threshold: float = 0.7

    # Resolution preferences
    prefer_visual_on_mismatch: bool = True
    accept_extra_visual_nodes: bool = True
    extra_visual_penalty: float = 0.15


class Harmonizer:
    """
    Cross-validate transcript and visual extractions.

    Detects conflicts between what the transcript says and what
    the visual extraction found, then resolves them using configurable
    strategies.
    """

    def __init__(
        self,
        state: GraphStateManager,
        config: HarmonizerConfig | None = None,
    ):
        """
        Initialize the harmonizer.

        Args:
            state: Graph state manager to update
            config: Optional harmonization configuration
        """
        # Import here to avoid circular import
        from ..state.merger import StateMerger

        self.state = state
        self.config = config or HarmonizerConfig()
        self.merger = StateMerger(state)

        # Build reverse alias map for matching
        self._build_alias_map()

    def _build_alias_map(self):
        """Build maps for node type alias resolution."""
        # Canonical type -> all aliases
        self.canonical_to_aliases: dict[str, set[str]] = {}
        # Alias -> canonical type
        self.alias_to_canonical: dict[str, str] = {}

        for alias, canonical in HOUDINI_NODE_ALIASES.items():
            self.alias_to_canonical[alias.lower()] = canonical.lower()
            if canonical.lower() not in self.canonical_to_aliases:
                self.canonical_to_aliases[canonical.lower()] = set()
            self.canonical_to_aliases[canonical.lower()].add(alias.lower())

    def harmonize(
        self,
        visual_extraction: GraphExtraction,
        transcript_events: list[EnhancedActionEvent],
    ) -> HarmonizationResult:
        """
        Harmonize a visual extraction with relevant transcript events.

        Args:
            visual_extraction: Visual extraction from a frame
            transcript_events: All transcript events (will be filtered by time)

        Returns:
            HarmonizationResult with statistics and conflicts
        """
        timestamp = visual_extraction.timestamp

        # Filter transcript events to relevant time window
        relevant_events = self._get_relevant_events(transcript_events, timestamp)

        result = HarmonizationResult(
            timestamp=timestamp,
            visual_nodes=len(visual_extraction.nodes),
            visual_connections=len(visual_extraction.connections),
            transcript_events=len(relevant_events),
        )

        if not visual_extraction.graph_visible:
            return result

        # Detect conflicts
        conflicts = self._detect_conflicts(visual_extraction, relevant_events)
        result.conflicts = conflicts

        # Resolve conflicts
        for conflict in conflicts:
            resolution = self._resolve_conflict(conflict, visual_extraction)
            result.resolutions.append(resolution)

        # Merge visual extraction (applies corroboration)
        merge_result = self.merger.merge_visual_extraction(
            visual_extraction,
            relevant_events,
        )

        result.corroborations = merge_result.corroborations

        return result

    def _get_relevant_events(
        self,
        events: list[EnhancedActionEvent],
        timestamp: float,
    ) -> list[EnhancedActionEvent]:
        """Get transcript events relevant to a timestamp."""
        start = timestamp - self.config.transcript_window_before
        end = timestamp + self.config.transcript_window_after

        return [e for e in events if start <= e.timestamp <= end]

    def _detect_conflicts(
        self,
        visual: GraphExtraction,
        transcript_events: list[EnhancedActionEvent],
    ) -> list[ConflictRecord]:
        """
        Detect conflicts between visual and transcript.

        Conflict types:
        - NODE_TYPE_MISMATCH: transcript says "scatter", visual shows "copytopoints"
        - NODE_MISSING_VISUAL: transcript mentions node not in frame
        - EXTRA_VISUAL_NODE: visual shows undocumented node
        """
        conflicts = []

        # Get visual node info
        visual_nodes = {n.name.lower(): n for n in visual.nodes}
        visual_types = {n.type.lower() for n in visual.nodes}

        # Get transcript node expectations
        expected_types = set()
        expected_names = set()

        for event in transcript_events:
            if event.node_type:
                canonical = self._get_canonical_type(event.node_type)
                expected_types.add(canonical)
            if event.node_name:
                expected_names.add(event.node_name.lower())

        # Check for NODE_MISSING_VISUAL
        for event in transcript_events:
            if not event.node_type:
                continue

            canonical = self._get_canonical_type(event.node_type)
            found = False

            # Check if we have this type or any alias
            for visual_type in visual_types:
                if self._types_match(canonical, visual_type):
                    found = True
                    break

            if not found:
                conflict = ConflictRecord(
                    conflict_type=ConflictType.NODE_MISSING_VISUAL.value,
                    timestamp=visual.timestamp,
                    visual_data={"visible_types": list(visual_types)},
                    transcript_data={
                        "expected_type": event.node_type,
                        "canonical_type": canonical,
                        "raw_text": event.raw_text,
                        "event_timestamp": event.timestamp,
                    },
                )
                conflicts.append(conflict)
                self.state.record_conflict(conflict)

        # Check for EXTRA_VISUAL_NODE (nodes not mentioned in transcript)
        if expected_types:
            for node in visual.nodes:
                node_type = node.type.lower()

                # Skip common utility nodes
                if node_type in ["null", "output", "unknown", "merge", "switch"]:
                    continue

                found_mention = False
                for expected in expected_types:
                    if self._types_match(node_type, expected):
                        found_mention = True
                        break

                if not found_mention:
                    conflict = ConflictRecord(
                        conflict_type=ConflictType.EXTRA_VISUAL_NODE.value,
                        timestamp=visual.timestamp,
                        visual_data={
                            "node_name": node.name,
                            "node_type": node.type,
                        },
                        transcript_data={
                            "mentioned_types": list(expected_types),
                        },
                    )
                    conflicts.append(conflict)
                    self.state.record_conflict(conflict)

        return conflicts

    def _resolve_conflict(
        self,
        conflict: ConflictRecord,
        visual: GraphExtraction,
    ) -> dict:
        """
        Resolve a conflict using configured strategies.

        Returns resolution details.
        """
        resolution = {
            "conflict_type": conflict.conflict_type,
            "timestamp": conflict.timestamp,
            "strategy": None,
            "confidence": 0.0,
            "action": None,
        }

        if conflict.conflict_type == ConflictType.NODE_MISSING_VISUAL.value:
            # Node mentioned in transcript but not visible
            # This could be because:
            # 1. Node hasn't been created yet (spoken before action)
            # 2. Node is out of frame
            # 3. Transcript refers to a different node

            resolution["strategy"] = ResolutionStrategy.DEFER.value
            resolution["confidence"] = 0.5
            resolution["action"] = "Wait for future frames to confirm"
            conflict.resolution = resolution["strategy"]
            conflict.resolution_confidence = resolution["confidence"]

        elif conflict.conflict_type == ConflictType.EXTRA_VISUAL_NODE.value:
            if self.config.accept_extra_visual_nodes:
                # Accept the visual node but with reduced confidence
                resolution["strategy"] = ResolutionStrategy.ACCEPT_WITH_PENALTY.value
                resolution["confidence"] = 0.7
                resolution["action"] = f"Accept visual node with {self.config.extra_visual_penalty*100:.0f}% confidence penalty"

                # Apply confidence penalty to the node
                node_name = conflict.visual_data.get("node_name") if conflict.visual_data else None
                if node_name:
                    node = self.state.get_node(node_name)
                    if node:
                        node.confidence = max(0.1, node.confidence - self.config.extra_visual_penalty)

                conflict.resolution = resolution["strategy"]
                conflict.resolution_confidence = resolution["confidence"]
            else:
                resolution["strategy"] = ResolutionStrategy.TRUST_VISUAL.value
                resolution["confidence"] = visual.extraction_confidence
                resolution["action"] = "Trust visual extraction"
                conflict.resolution = resolution["strategy"]
                conflict.resolution_confidence = resolution["confidence"]

        elif conflict.conflict_type == ConflictType.NODE_TYPE_MISMATCH.value:
            # Check if this is an alias mismatch
            visual_type = conflict.visual_data.get("node_type", "") if conflict.visual_data else ""
            transcript_type = conflict.transcript_data.get("expected_type", "") if conflict.transcript_data else ""

            if self._types_match(visual_type, transcript_type):
                resolution["strategy"] = ResolutionStrategy.ALIAS_MATCH.value
                resolution["confidence"] = 0.9
                resolution["action"] = f"Matched via alias: {transcript_type} = {visual_type}"
                conflict.resolution = resolution["strategy"]
                conflict.resolution_confidence = resolution["confidence"]
            elif self.config.prefer_visual_on_mismatch:
                resolution["strategy"] = ResolutionStrategy.TRUST_VISUAL.value
                resolution["confidence"] = visual.extraction_confidence
                resolution["action"] = "Trust high-confidence visual over transcript"
                conflict.resolution = resolution["strategy"]
                conflict.resolution_confidence = resolution["confidence"]
            else:
                resolution["strategy"] = ResolutionStrategy.DEFER.value
                resolution["confidence"] = 0.4
                resolution["action"] = "Ambiguous - flagged for review"
                conflict.resolution = resolution["strategy"]
                conflict.resolution_confidence = resolution["confidence"]

        return resolution

    def _get_canonical_type(self, node_type: str) -> str:
        """Get canonical node type from an alias."""
        if not node_type:
            return ""

        lower = node_type.lower().strip()

        if lower in self.alias_to_canonical:
            return self.alias_to_canonical[lower]

        return lower

    def _types_match(self, type1: str, type2: str) -> bool:
        """Check if two node types match (including aliases)."""
        if not type1 or not type2:
            return False

        t1 = type1.lower().strip()
        t2 = type2.lower().strip()

        # Direct match
        if t1 == t2:
            return True

        # Get canonical forms
        c1 = self._get_canonical_type(t1)
        c2 = self._get_canonical_type(t2)

        if c1 == c2:
            return True

        # Check if one is contained in the other
        if t1 in t2 or t2 in t1:
            return True

        # Check if they're in the same alias group
        if c1 in self.canonical_to_aliases:
            if t2 in self.canonical_to_aliases[c1] or c2 in self.canonical_to_aliases[c1]:
                return True

        if c2 in self.canonical_to_aliases:
            if t1 in self.canonical_to_aliases[c2] or c1 in self.canonical_to_aliases[c2]:
                return True

        return False

    def get_harmony_score(self) -> float:
        """
        Calculate overall harmony score between transcript and visual.

        Higher score = better agreement.
        """
        stats = self.state.get_stats()

        if stats["total_nodes"] == 0:
            return 0.0

        # Factor in corroboration rate
        corroboration_rate = stats["corroborated_nodes"] / stats["total_nodes"]

        # Factor in conflict rate (inverse)
        if stats["frames_processed"] > 0:
            conflict_rate = stats["conflicts_detected"] / stats["frames_processed"]
        else:
            conflict_rate = 0

        # Calculate harmony score
        # High corroboration + low conflicts = high harmony
        score = (corroboration_rate * 0.6) + ((1 - min(conflict_rate, 1.0)) * 0.4)

        return round(score, 2)

    def get_summary(self) -> dict:
        """Get summary of harmonization state."""
        stats = self.state.get_stats()

        return {
            "total_nodes": stats["total_nodes"],
            "total_connections": stats["total_connections"],
            "corroborated_nodes": stats["corroborated_nodes"],
            "conflicts_detected": stats["conflicts_detected"],
            "harmony_score": self.get_harmony_score(),
            "average_confidence": stats["average_confidence"],
        }
