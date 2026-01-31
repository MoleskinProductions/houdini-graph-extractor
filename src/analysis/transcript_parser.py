"""Transcript parsing and action event detection."""

import re
from dataclasses import dataclass, field
from enum import Enum

from ..ingestion.youtube import TranscriptSegment


class EventType(str, Enum):
    """Types of actions detected in transcript."""

    NODE_CREATE = "node_create"
    NODE_CONNECT = "node_connect"
    PARAM_CHANGE = "param_change"
    NODE_DELETE = "node_delete"
    NODE_RENAME = "node_rename"
    CONTEXT_SWITCH = "context_switch"
    SELECTION = "selection"


@dataclass
class ActionEvent:
    """A detected action event from the transcript."""

    timestamp: float
    event_type: EventType
    confidence: float
    raw_text: str
    entities: dict = field(default_factory=dict)


# Houdini node type aliases - common spoken names to actual node types
HOUDINI_NODE_ALIASES = {
    # SOPs
    "sphere": "sphere",
    "box": "box",
    "grid": "grid",
    "tube": "tube",
    "torus": "torus",
    "circle": "circle",
    "line": "line",
    "curve": "curve",
    "null": "null",
    "merge": "merge",
    "switch": "switch",
    "blast": "blast",
    "delete": "delete",
    "transform": "xform",
    "copy to points": "copytopoints",
    "copy stamp": "copy",
    "scatter": "scatter",
    "attrib wrangle": "attribwrangle",
    "attribute wrangle": "attribwrangle",
    "wrangle": "attribwrangle",
    "vex wrangle": "attribwrangle",
    "point wrangle": "attribwrangle",
    "attrib vop": "attribvop",
    "attribute vop": "attribvop",
    "vop": "attribvop",
    "mountain": "mountain",
    "noise": "mountain",
    "subdivide": "subdivide",
    "subdivision": "subdivide",
    "remesh": "remesh",
    "poly reduce": "polyreduce",
    "polybevel": "polybevel",
    "bevel": "polybevel",
    "extrude": "polyextrude",
    "poly extrude": "polyextrude",
    "boolean": "boolean",
    "vdb": "vdb",
    "vdb from polygons": "vdbfrompolygons",
    "convert vdb": "convertvdb",
    "smooth": "smooth",
    "relax": "relax",
    "fuse": "fuse",
    "facet": "facet",
    "normal": "normal",
    "reverse": "reverse",
    "group": "group",
    "group create": "groupcreate",
    "group expression": "groupexpression",
    "for each": "foreach",
    "for loop": "foreach",
    "solver": "solver",
    "file": "file",
    "file cache": "filecache",
    "cache": "filecache",
    "output": "output",
    "object merge": "object_merge",
    "object_merge": "object_merge",
    "attribute create": "attribcreate",
    "attrib create": "attribcreate",
    "attribute promote": "attribpromote",
    "attribute transfer": "attribtransfer",
    "ray": "ray",
    "rest position": "rest",
    "rest": "rest",
    "uv unwrap": "uvunwrap",
    "uv project": "uvproject",
    "uv flatten": "uvflatten",
    "material": "material",
    # DOPs
    "dop network": "dopnet",
    "dopnet": "dopnet",
    "rbd": "rbdpackedobject",
    "rigid body": "rbdpackedobject",
    "pyro": "pyrosolver",
    "smoke": "smokesolver",
    "flip": "flipsolver",
    "vellum": "vellumsolver",
    "cloth": "clothsolver",
    "wire": "wiresolver",
    "pop": "popnet",
    "particles": "popnet",
    "gravity": "gravity",
    "ground plane": "groundplane",
    # Others
    "geometry": "geo",
    "geo": "geo",
    "subnet": "subnet",
    "digital asset": "subnet",
    "hda": "subnet",
}


# Regex patterns for action detection
ACTION_PATTERNS = {
    EventType.NODE_CREATE: [
        r"\b(?:add|drop(?:ping)?(?:\s+down)?|create|put(?:ting)?(?:\s+in)?|place|insert|bring(?:ing)?(?:\s+in)?|throw(?:ing)?(?:\s+in)?|make)\b.*?(?:a|an|the)?\s*(\w+(?:\s+\w+)?)",
        r"\b(?:i(?:'ll| will)?|let(?:'s|me)?|we(?:'ll)?)\s+(?:add|drop|create|put|place)\b.*?(?:a|an|the)?\s*(\w+(?:\s+\w+)?)",
        r"\b(\w+(?:\s+\w+)?)\s+(?:node|sop|dop|vop)\b",
    ],
    EventType.NODE_CONNECT: [
        r"\b(?:connect|wire|plug|feed|pipe|link|hook)\b.*?(?:to|into|in)",
        r"\b(?:connect|wire|plug|feed)(?:ing)?\s+(?:the|this|that)?\s*(?:output|input)?\b",
        r"\b(?:from|take|grab)\s+(?:the|this)?\s*(?:output|input)?\s*(?:and|to)\s*(?:connect|wire|plug|feed)\b",
    ],
    EventType.PARAM_CHANGE: [
        r"\b(?:set|change|adjust|modify|tweak|turn|type|enter|put)\b.*?(?:to|at|as|value)",
        r"\b(?:parameter|param|value|setting)\b.*?(?:to|at|as)\b",
        r"\b(?:in the|into the)\s+(?:parameter|field|box|input)\b",
    ],
    EventType.NODE_DELETE: [
        r"\b(?:delete|remove|get rid of|take out|erase)\b",
    ],
    EventType.NODE_RENAME: [
        r"\b(?:rename|call(?:ing)?\s+(?:this|it)|name(?:d)?)\b",
    ],
    EventType.CONTEXT_SWITCH: [
        r"\b(?:go(?:ing)?|dive|jump|hop|click|double[- ]?click)\s+(?:into|in(?:to)?|inside|to|back)\b",
        r"\b(?:enter|open|inside)\s+(?:the|this|that)?\s*(?:node|network|subnet)\b",
        r"\b(?:back|up|out)\s+(?:to|from)\b",
    ],
    EventType.SELECTION: [
        r"\b(?:select|click(?:ing)?(?:\s+on)?|grab|pick|choose|highlight)\b.*?(?:the|this|that)?\s*(\w+)",
    ],
}


class TranscriptParser:
    """Parse transcripts to detect Houdini action events."""

    def __init__(self):
        # Compile patterns for efficiency
        self.compiled_patterns = {
            event_type: [re.compile(p, re.IGNORECASE) for p in patterns]
            for event_type, patterns in ACTION_PATTERNS.items()
        }

    def normalize_node_type(self, text: str) -> str | None:
        """Convert spoken node name to Houdini node type."""
        text_lower = text.lower().strip()

        # Direct match
        if text_lower in HOUDINI_NODE_ALIASES:
            return HOUDINI_NODE_ALIASES[text_lower]

        # Partial match (e.g., "scatter sop" -> "scatter")
        for alias, node_type in HOUDINI_NODE_ALIASES.items():
            if alias in text_lower or text_lower in alias:
                return node_type

        return None

    def detect_event_type(self, text: str) -> tuple[EventType | None, float, dict]:
        """
        Detect what type of action event is described in text.

        Returns:
            Tuple of (event_type, confidence, entities)
        """
        text_lower = text.lower()
        entities = {}

        for event_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(text_lower)
                if match:
                    confidence = 0.7  # Base confidence for regex match

                    # Try to extract entities
                    if event_type == EventType.NODE_CREATE:
                        # Look for node type in the matched text
                        groups = match.groups()
                        if groups:
                            potential_node = groups[0]
                            node_type = self.normalize_node_type(potential_node)
                            if node_type:
                                entities["node_type"] = node_type
                                confidence = 0.85  # Higher confidence when we identify the node

                    return event_type, confidence, entities

        return None, 0.0, {}

    def parse(self, transcript: list[TranscriptSegment]) -> list[ActionEvent]:
        """
        Parse transcript segments to extract action events.

        Args:
            transcript: List of transcript segments with timing

        Returns:
            List of detected action events
        """
        events = []

        for segment in transcript:
            event_type, confidence, entities = self.detect_event_type(segment.text)

            if event_type is not None:
                event = ActionEvent(
                    timestamp=segment.start,
                    event_type=event_type,
                    confidence=confidence,
                    raw_text=segment.text,
                    entities=entities,
                )
                events.append(event)

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        return events

    def get_sampling_timestamps(
        self,
        events: list[ActionEvent],
        buffer_before: float = 0.5,
        buffer_after: float = 2.0,
        interval: float = 0.5,
    ) -> list[dict]:
        """
        Generate frame sampling timestamps from action events.

        The graph state is typically visible AFTER the narrator describes the action,
        so we sample more heavily after the spoken timestamp.

        Args:
            events: List of action events
            buffer_before: Seconds to sample before event timestamp
            buffer_after: Seconds to sample after event timestamp
            interval: Interval between samples

        Returns:
            List of sampling timestamp records
        """
        timestamps = []
        seen = set()

        for event in events:
            start = event.timestamp - buffer_before
            end = event.timestamp + buffer_after
            t = max(0, start)

            while t <= end:
                # Round to avoid floating point issues
                t_rounded = round(t, 2)

                if t_rounded not in seen:
                    seen.add(t_rounded)
                    timestamps.append(
                        {
                            "time": t_rounded,
                            "source_event": event,
                            "priority": "high" if t > event.timestamp else "low",
                        }
                    )

                t += interval

        # Sort by time
        timestamps.sort(key=lambda x: x["time"])

        return timestamps
