"""State models for graph elements with temporal metadata."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SourceType(str, Enum):
    """Source of an extraction."""

    VISUAL = "visual"
    TRANSCRIPT = "transcript"
    HARMONIZED = "harmonized"


@dataclass
class SourceRecord:
    """Record of where an element was extracted from."""

    source_type: SourceType
    timestamp: float
    confidence: float
    raw_data: dict | None = None


@dataclass
class NodeState:
    """
    State of a single node with temporal tracking.

    Tracks when a node was first seen, last seen, and from which
    sources (visual/transcript) it was extracted.
    """

    name: str
    type: str
    position: list[float]
    flags: dict = field(default_factory=lambda: {"display": False, "render": False, "bypass": False})
    params: dict = field(default_factory=dict)  # Extracted parameter values
    confidence: float = 0.5
    first_seen: float = 0.0
    last_seen: float = 0.0
    sources: list[SourceRecord] = field(default_factory=list)

    # Observation tracking for confidence decay
    observation_count: int = 1
    frames_since_seen: int = 0

    # Harmonization metadata
    transcript_corroborated: bool = False
    position_stable: bool = True

    def update_from_visual(self, timestamp: float, confidence: float, position: list[float],
                           flags: dict | None = None) -> None:
        """Update node state from a visual extraction."""
        self.last_seen = timestamp
        self.observation_count += 1
        self.frames_since_seen = 0

        # Update confidence (weighted average)
        self.confidence = (self.confidence * 0.6) + (confidence * 0.4)

        # Check position stability
        if self.position:
            dx = abs(position[0] - self.position[0])
            dy = abs(position[1] - self.position[1])
            if dx > 5 or dy > 5:  # More than 5% change
                self.position_stable = False

        self.position = position

        if flags:
            # Only update flags if set to True (don't clear them)
            for key, value in flags.items():
                if value:
                    self.flags[key] = value

        self.sources.append(SourceRecord(
            source_type=SourceType.VISUAL,
            timestamp=timestamp,
            confidence=confidence,
        ))

    def corroborate_with_transcript(self, timestamp: float, confidence: float = 0.2) -> None:
        """Boost confidence when transcript mentions this node."""
        self.transcript_corroborated = True
        # Boost confidence by up to 20%
        self.confidence = min(1.0, self.confidence + confidence)
        self.sources.append(SourceRecord(
            source_type=SourceType.TRANSCRIPT,
            timestamp=timestamp,
            confidence=confidence,
        ))

    def apply_param(self, param_name: str, param_value: Any, timestamp: float) -> None:
        """
        Apply a parameter value extracted from transcript.

        Args:
            param_name: Name of the parameter
            param_value: Value to set
            timestamp: When this parameter was mentioned
        """
        if not param_name:
            return

        # Normalize parameter name (lowercase, strip whitespace)
        param_name = param_name.lower().strip().replace(" ", "_")

        # Store the parameter with its timestamp for tracking
        self.params[param_name] = {
            "value": param_value,
            "timestamp": timestamp,
        }

        # Log source
        self.sources.append(SourceRecord(
            source_type=SourceType.TRANSCRIPT,
            timestamp=timestamp,
            confidence=0.7,
            raw_data={"param_name": param_name, "param_value": param_value},
        ))

    def decay_confidence(self, decay_rate: float = 0.05) -> None:
        """Decay confidence for nodes not seen in recent frames."""
        self.frames_since_seen += 1
        if self.frames_since_seen > 3:  # After 3 frames without observation
            self.confidence = max(0.1, self.confidence - decay_rate)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.type,
            "position": self.position,
            "flags": self.flags,
            "params": self.get_params_flat(),
            "confidence": self.confidence,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "observation_count": self.observation_count,
            "transcript_corroborated": self.transcript_corroborated,
            "position_stable": self.position_stable,
        }

    def get_params_flat(self) -> dict:
        """Get parameters as simple key-value pairs (without metadata)."""
        return {k: v["value"] if isinstance(v, dict) and "value" in v else v
                for k, v in self.params.items()}


@dataclass
class ConnectionState:
    """
    State of a connection between two nodes.

    Tracks connection endpoints and observation metadata.
    """

    from_node: str
    from_output: int
    to_node: str
    to_input: int
    confidence: float = 0.5
    first_seen: float = 0.0
    last_seen: float = 0.0
    observation_count: int = 1
    sources: list[SourceRecord] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        """Unique key for this connection."""
        return (self.from_node, self.from_output, self.to_node, self.to_input)

    def update_observation(self, timestamp: float, confidence: float) -> None:
        """Update connection state from an observation."""
        self.last_seen = timestamp
        self.observation_count += 1
        # Weighted average confidence
        self.confidence = (self.confidence * 0.6) + (confidence * 0.4)
        self.sources.append(SourceRecord(
            source_type=SourceType.VISUAL,
            timestamp=timestamp,
            confidence=confidence,
        ))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "from_node": self.from_node,
            "from_output": self.from_output,
            "to_node": self.to_node,
            "to_input": self.to_input,
            "confidence": self.confidence,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "observation_count": self.observation_count,
        }


@dataclass
class ConflictRecord:
    """Record of a detected conflict between sources."""

    conflict_type: str
    timestamp: float
    visual_data: dict | None = None
    transcript_data: dict | None = None
    resolution: str | None = None
    resolution_confidence: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "conflict_type": self.conflict_type,
            "timestamp": self.timestamp,
            "visual_data": self.visual_data,
            "transcript_data": self.transcript_data,
            "resolution": self.resolution,
            "resolution_confidence": self.resolution_confidence,
        }


@dataclass
class EnhancedActionEvent:
    """
    Enhanced action event with LLM-extracted entities.

    Extends the basic ActionEvent with more structured entity extraction.
    """

    timestamp: float
    event_type: str
    confidence: float
    raw_text: str

    # Enhanced entity extraction from LLM
    node_type: str | None = None
    node_name: str | None = None
    source_node: str | None = None
    target_node: str | None = None
    param_name: str | None = None
    param_value: Any = None

    # Linking to visual extractions
    linked_visual_timestamp: float | None = None
    visual_match_confidence: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
            "node_type": self.node_type,
            "node_name": self.node_name,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "param_name": self.param_name,
            "param_value": self.param_value,
            "linked_visual_timestamp": self.linked_visual_timestamp,
            "visual_match_confidence": self.visual_match_confidence,
        }
