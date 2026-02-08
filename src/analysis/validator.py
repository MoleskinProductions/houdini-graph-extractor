"""Centralized structural validation against Phase 1A/2A corpora.

Loads SchemaCorpus (4,876 node types) and PatternCorpus (4,005 connection
patterns), builds O(1) lookup indexes, and provides type resolution +
connection validation for the video extraction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ingestion.node_schema.models import SchemaCorpus
    from ..analysis.pattern_mining.models import PatternCorpus


class ValidationStatus(str, Enum):
    """Result of validating a node type or connection."""

    VALID = "valid"
    VALID_ALIAS = "valid_alias"
    UNKNOWN = "unknown"


# Confidence adjustments applied during validation
CONF_VALID = 0.10
CONF_VALID_ALIAS = 0.05
CONF_UNKNOWN = -0.15
CONF_KNOWN_PATTERN = 0.10
CONF_UNKNOWN_PATTERN = -0.05


@dataclass(frozen=True)
class TypeValidationResult:
    """Result of validating a raw node type string."""

    status: ValidationStatus
    resolved_type: str          # canonical type_name (e.g. "scatter")
    resolved_key: str | None    # schema key (e.g. "sop/scatter") or None
    confidence_adjustment: float


@dataclass(frozen=True)
class ConnectionValidationResult:
    """Result of validating a connection against pattern corpus."""

    known_pattern: bool
    pattern_count: int
    confidence_adjustment: float


class StructuralValidator:
    """Validate extracted data against Phase 1A schemas and Phase 2A patterns.

    Build once, query many times.  All lookups are O(1) dict access after
    the initial index build.
    """

    def __init__(
        self,
        schema: SchemaCorpus | None = None,
        patterns: PatternCorpus | None = None,
    ) -> None:
        self._schema = schema
        self._patterns = patterns

        # Schema indexes (populated by _build_schema_indexes)
        self._key_set: set[str] = set()                     # "sop/scatter"
        self._type_name_to_keys: dict[str, list[str]] = {}  # "scatter" -> ["sop/scatter"]
        self._label_to_keys: dict[str, list[str]] = {}      # "copy to points" -> ["sop/copytopoints"]
        self._context_types: dict[str, set[str]] = {}       # "sop" -> {"scatter", "box", ...}

        # Pattern index
        self._edge_keys: dict[str, int] = {}  # "scatter:0->copytopoints:0" -> count

        if schema:
            self._build_schema_indexes()
        if patterns:
            self._build_pattern_index()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_schema_indexes(self) -> None:
        """Pre-compute lookup dicts from schema corpus.  O(n) build."""
        for key, node in self._schema.nodes.items():
            key_lower = key.lower()
            self._key_set.add(key_lower)

            type_name = node.type_name.lower()
            self._type_name_to_keys.setdefault(type_name, []).append(key_lower)

            label = node.label.lower()
            if label:
                self._label_to_keys.setdefault(label, []).append(key_lower)

            ctx = node.context.lower() if hasattr(node, "context") else key_lower.split("/")[0]
            self._context_types.setdefault(ctx, set()).add(type_name)

    def _build_pattern_index(self) -> None:
        """Pre-compute edge_key -> count from pattern corpus."""
        for edge_key, pattern in self._patterns.connection_patterns.items():
            self._edge_keys[edge_key.lower()] = pattern.count

    # ------------------------------------------------------------------
    # Public API: node type validation
    # ------------------------------------------------------------------

    def validate_node_type(
        self,
        raw_type: str,
        context_hint: str | None = None,
    ) -> TypeValidationResult:
        """Resolve a raw type string against the schema corpus.

        Resolution chain:
        1. Direct key match  (e.g. "sop/scatter")
        2. type_name index   (e.g. "scatter" -> "sop/scatter")
        3. label index       (e.g. "Copy to Points" -> "sop/copytopoints")
        4. UNKNOWN
        """
        if not self._schema:
            return TypeValidationResult(
                status=ValidationStatus.UNKNOWN,
                resolved_type=raw_type,
                resolved_key=None,
                confidence_adjustment=0.0,
            )

        raw_lower = raw_type.lower().strip()
        ctx = context_hint.lower().strip() if context_hint else None

        # 1. Direct key match
        if raw_lower in self._key_set:
            type_name = raw_lower.split("/", 1)[1] if "/" in raw_lower else raw_lower
            return TypeValidationResult(
                status=ValidationStatus.VALID,
                resolved_type=type_name,
                resolved_key=raw_lower,
                confidence_adjustment=CONF_VALID,
            )

        # If raw_type contains a slash but didn't match, try just the type part
        if "/" in raw_lower:
            raw_lower = raw_lower.split("/", 1)[1]

        # 2. type_name index
        keys = self._type_name_to_keys.get(raw_lower)
        if keys:
            resolved_key = self._pick_key(keys, ctx)
            return TypeValidationResult(
                status=ValidationStatus.VALID,
                resolved_type=raw_lower,
                resolved_key=resolved_key,
                confidence_adjustment=CONF_VALID,
            )

        # 3. label index
        keys = self._label_to_keys.get(raw_lower)
        if keys:
            resolved_key = self._pick_key(keys, ctx)
            type_name = resolved_key.split("/", 1)[1] if "/" in resolved_key else resolved_key
            return TypeValidationResult(
                status=ValidationStatus.VALID_ALIAS,
                resolved_type=type_name,
                resolved_key=resolved_key,
                confidence_adjustment=CONF_VALID_ALIAS,
            )

        # 4. Unknown
        return TypeValidationResult(
            status=ValidationStatus.UNKNOWN,
            resolved_type=raw_type,
            resolved_key=None,
            confidence_adjustment=CONF_UNKNOWN,
        )

    def normalize_type(
        self,
        raw_type: str,
        context_hint: str | None = None,
    ) -> str:
        """Convenience: return the canonical type_name for *raw_type*."""
        return self.validate_node_type(raw_type, context_hint).resolved_type

    def is_valid_type(
        self,
        raw_type: str,
        context_hint: str | None = None,
    ) -> bool:
        """Return True if *raw_type* resolves to a known schema entry."""
        result = self.validate_node_type(raw_type, context_hint)
        return result.status in (ValidationStatus.VALID, ValidationStatus.VALID_ALIAS)

    # ------------------------------------------------------------------
    # Public API: connection validation
    # ------------------------------------------------------------------

    def validate_connection(
        self,
        source_type: str,
        dest_type: str,
        source_output: int = 0,
        dest_input: int = 0,
    ) -> ConnectionValidationResult:
        """Check whether a connection matches a known pattern."""
        if not self._patterns:
            return ConnectionValidationResult(
                known_pattern=False,
                pattern_count=0,
                confidence_adjustment=0.0,
            )

        # Normalise types through schema if available
        src = self.normalize_type(source_type).lower()
        dst = self.normalize_type(dest_type).lower()

        edge_key = f"{src}:{source_output}->{dst}:{dest_input}"
        count = self._edge_keys.get(edge_key, 0)

        if count > 0:
            return ConnectionValidationResult(
                known_pattern=True,
                pattern_count=count,
                confidence_adjustment=CONF_KNOWN_PATTERN,
            )

        return ConnectionValidationResult(
            known_pattern=False,
            pattern_count=0,
            confidence_adjustment=CONF_UNKNOWN_PATTERN,
        )

    # ------------------------------------------------------------------
    # Public API: type comparison
    # ------------------------------------------------------------------

    def types_match(
        self,
        type1: str,
        type2: str,
        context_hint: str | None = None,
    ) -> bool:
        """Return True if *type1* and *type2* resolve to the same canonical key."""
        r1 = self.validate_node_type(type1, context_hint)
        r2 = self.validate_node_type(type2, context_hint)

        # If both resolved to a schema key, compare keys
        if r1.resolved_key and r2.resolved_key:
            return r1.resolved_key == r2.resolved_key

        # Fall back to comparing resolved type names
        return r1.resolved_type.lower() == r2.resolved_type.lower()

    # ------------------------------------------------------------------
    # Stats (for output metadata)
    # ------------------------------------------------------------------

    @property
    def schema_node_count(self) -> int:
        return len(self._key_set)

    @property
    def pattern_count(self) -> int:
        return len(self._edge_keys)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_key(self, keys: list[str], context_hint: str | None) -> str:
        """Pick the best key when multiple contexts match."""
        if len(keys) == 1:
            return keys[0]

        if context_hint:
            for k in keys:
                if k.startswith(context_hint + "/"):
                    return k

        # Default preference order
        preferred = ["sop", "obj", "dop", "vop", "lop", "top", "cop", "chop", "shop"]
        for pref in preferred:
            for k in keys:
                if k.startswith(pref + "/"):
                    return k

        return keys[0]
