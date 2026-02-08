"""LLM-based transcript analyzer for enhanced entity extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from ..config import get_config
from ..ingestion.youtube import TranscriptSegment
from ..state.models import EnhancedActionEvent
from .transcript_parser import TranscriptParser, ActionEvent, EventType, HOUDINI_NODE_ALIASES

if TYPE_CHECKING:
    from .validator import StructuralValidator


# LLM prompt for entity extraction from transcript segments
ENTITY_EXTRACTION_PROMPT = """You are analyzing transcript segments from a Houdini 3D software tutorial video.

For each segment, extract structured information about what actions the tutorial creator is describing.

Focus on extracting:
1. **event_type**: What kind of action is being described?
   - "node_create": Creating/adding/dropping a new node
   - "node_connect": Wiring/connecting nodes together
   - "param_change": Changing a parameter value
   - "node_delete": Removing/deleting a node
   - "node_rename": Renaming a node
   - "context_switch": Diving into or out of a network/node
   - "selection": Selecting/clicking on something
   - "other": General explanation or non-action content

2. **node_type**: The type of Houdini node mentioned (e.g., "sphere", "box", "scatter", "attribwrangle", "merge", "null", "copytopoints", "mountain", "transform")

3. **node_name**: A specific node instance name if mentioned (e.g., "scatter1", "myGeo", "OUT")

4. **source_node**: For connections, the node being connected FROM

5. **target_node**: For connections, the node being connected TO

6. **param_name**: For parameter changes, the parameter being modified

7. **param_value**: For parameter changes, the new value being set

Analyze these transcript segments and return a JSON array with one object per segment:

SEGMENTS:
{segments}

Return ONLY valid JSON array (no markdown, no explanation):
[
  {{
    "segment_index": 0,
    "event_type": "node_create",
    "confidence": 0.9,
    "node_type": "scatter",
    "node_name": null,
    "source_node": null,
    "target_node": null,
    "param_name": null,
    "param_value": null
  }},
  ...
]

For segments that don't describe Houdini actions, use event_type "other" with low confidence.
Use null for fields that aren't applicable or can't be determined."""


@dataclass
class LLMTranscriptAnalyzer:
    """
    LLM-based transcript analyzer for enhanced entity extraction.

    Uses the same Gemini API as the visual extractor for consistency.
    Falls back to regex-based parsing on failure.
    """

    api_base: str | None = None
    model: str | None = None
    api_key: str | None = None
    batch_size: int = 10
    fallback_parser: TranscriptParser = field(default_factory=TranscriptParser)
    validator: StructuralValidator | None = None

    def __post_init__(self):
        import os
        config = get_config()
        self.api_base = self.api_base or config.vlm.api_base
        self.model = self.model or config.vlm.model_name
        self.max_tokens = config.vlm.max_tokens
        self.temperature = 0.1  # Low temperature for consistent extraction

        self.api_key = (
            self.api_key
            or config.vlm.api_key
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
            or "not-needed"
        )

        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )

    def analyze(
        self,
        transcript: list[TranscriptSegment],
        use_fallback_on_error: bool = True,
    ) -> list[EnhancedActionEvent]:
        """
        Analyze transcript segments to extract enhanced action events.

        Args:
            transcript: List of transcript segments
            use_fallback_on_error: Whether to fall back to regex on LLM failure

        Returns:
            List of EnhancedActionEvent objects
        """
        events = []

        # Process in batches
        for i in range(0, len(transcript), self.batch_size):
            batch = transcript[i:i + self.batch_size]

            try:
                batch_events = self._analyze_batch(batch, start_index=i)
                events.extend(batch_events)
            except Exception as e:
                print(f"LLM analysis failed for batch {i//self.batch_size}: {e}")
                if use_fallback_on_error:
                    # Fall back to regex-based parsing
                    fallback_events = self._fallback_analyze(batch)
                    events.extend(fallback_events)

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        return events

    def _analyze_batch(
        self,
        segments: list[TranscriptSegment],
        start_index: int = 0,
    ) -> list[EnhancedActionEvent]:
        """
        Analyze a batch of transcript segments using LLM.

        Args:
            segments: Batch of transcript segments
            start_index: Starting index for segment numbering

        Returns:
            List of EnhancedActionEvent objects
        """
        # Format segments for the prompt
        segments_text = "\n".join([
            f"[{i}] ({seg.start:.1f}s - {seg.end:.1f}s): {seg.text}"
            for i, seg in enumerate(segments)
        ])

        prompt = ENTITY_EXTRACTION_PROMPT.format(segments=segments_text)

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        results = self._parse_response(response)

        # Convert to EnhancedActionEvent objects
        events = []
        for result in results:
            segment_idx = result.get("segment_index", 0)
            if segment_idx >= len(segments):
                continue

            segment = segments[segment_idx]
            event_type = result.get("event_type", "other")

            # Skip non-action segments
            if event_type == "other" and result.get("confidence", 0) < 0.5:
                continue

            # Normalize node type using alias map
            node_type = result.get("node_type")
            if node_type:
                node_type = self._normalize_node_type(node_type)

            event = EnhancedActionEvent(
                timestamp=segment.start,
                event_type=event_type,
                confidence=result.get("confidence", 0.7),
                raw_text=segment.text,
                node_type=node_type,
                node_name=result.get("node_name"),
                source_node=result.get("source_node"),
                target_node=result.get("target_node"),
                param_name=result.get("param_name"),
                param_value=result.get("param_value"),
            )
            events.append(event)

        return events

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Make an LLM API call with retry logic."""
        import time

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content

            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait_time = (attempt + 1) * 15
                    print(f"Rate limited, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(f"Failed after {max_retries} retries")

    def _parse_response(self, response: str) -> list[dict]:
        """Parse JSON response from LLM."""
        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            response = json_match.group(1)

        response = response.strip()

        try:
            result = json.loads(response)
            if isinstance(result, list):
                return result
            return [result]
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            print(f"Response was: {response[:500]}")
            return []

    def _normalize_node_type(self, node_type: str) -> str:
        """Normalize node type using Houdini alias map."""
        if not node_type:
            return node_type

        # Try structural validator first (covers 4,876 node types)
        if self.validator:
            result = self.validator.validate_node_type(node_type)
            if result.status.value != "unknown":
                return result.resolved_type

        node_type_lower = node_type.lower().strip()

        # Direct match in alias map
        if node_type_lower in HOUDINI_NODE_ALIASES:
            return HOUDINI_NODE_ALIASES[node_type_lower]

        # Partial match
        for alias, canonical in HOUDINI_NODE_ALIASES.items():
            if alias in node_type_lower or node_type_lower in alias:
                return canonical

        return node_type

    def _fallback_analyze(
        self,
        segments: list[TranscriptSegment],
    ) -> list[EnhancedActionEvent]:
        """
        Fall back to regex-based parsing.

        Args:
            segments: Transcript segments to analyze

        Returns:
            List of EnhancedActionEvent objects (converted from ActionEvent)
        """
        action_events = self.fallback_parser.parse(segments)

        enhanced_events = []
        for event in action_events:
            enhanced = EnhancedActionEvent(
                timestamp=event.timestamp,
                event_type=event.event_type.value,
                confidence=event.confidence,
                raw_text=event.raw_text,
                node_type=event.entities.get("node_type"),
            )
            enhanced_events.append(enhanced)

        return enhanced_events

    def get_relevant_events(
        self,
        events: list[EnhancedActionEvent],
        timestamp: float,
        window_before: float = 3.0,
        window_after: float = 1.0,
    ) -> list[EnhancedActionEvent]:
        """
        Get events relevant to a specific timestamp.

        Args:
            events: All events
            timestamp: Target timestamp
            window_before: Seconds before timestamp to include
            window_after: Seconds after timestamp to include

        Returns:
            List of relevant events
        """
        start = timestamp - window_before
        end = timestamp + window_after

        return [
            e for e in events
            if start <= e.timestamp <= end
        ]

    def get_action_events(
        self,
        events: list[EnhancedActionEvent],
    ) -> list[EnhancedActionEvent]:
        """
        Filter to only action-relevant events (not 'other').

        Args:
            events: All events

        Returns:
            List of action events
        """
        return [
            e for e in events
            if e.event_type != "other"
        ]

    def to_sampling_timestamps(
        self,
        events: list[EnhancedActionEvent],
        buffer_before: float = 0.5,
        buffer_after: float = 2.0,
        interval: float = 0.5,
    ) -> list[dict]:
        """
        Generate frame sampling timestamps from enhanced events.

        Similar to TranscriptParser.get_sampling_timestamps but works
        with EnhancedActionEvent objects.

        Args:
            events: List of enhanced action events
            buffer_before: Seconds to sample before event
            buffer_after: Seconds to sample after event
            interval: Sampling interval

        Returns:
            List of timestamp records with priority and source event
        """
        timestamps = []
        seen = set()

        action_events = self.get_action_events(events)

        for event in action_events:
            start = event.timestamp - buffer_before
            end = event.timestamp + buffer_after
            t = max(0, start)

            while t <= end:
                t_rounded = round(t, 2)

                if t_rounded not in seen:
                    seen.add(t_rounded)
                    timestamps.append({
                        "time": t_rounded,
                        "source_event": event,
                        "priority": "high" if t > event.timestamp else "low",
                    })

                t += interval

        timestamps.sort(key=lambda x: x["time"])

        return timestamps
