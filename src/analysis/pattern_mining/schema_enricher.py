"""Optional schema enrichment for mined connection patterns.

Resolves port indices to human-readable port names using the
SchemaCorpus from Phase 1A node type introspection.
"""

from __future__ import annotations

from src.ingestion.node_schema.models import SchemaCorpus

from .models import PatternCorpus


class SchemaEnricher:
    """Enriches a PatternCorpus with port names from a SchemaCorpus.

    Looks up each node type in the schema corpus and resolves output/input
    port indices to their names.
    """

    def __init__(self, schema_corpus: SchemaCorpus) -> None:
        self.schema_corpus = schema_corpus

    def enrich(self, patterns: PatternCorpus) -> None:
        """Mutate patterns in-place, adding port names where resolvable."""
        self._enrich_connection_patterns(patterns)
        self._enrich_port_usage(patterns)

    def _resolve_output_name(self, node_type: str, port_index: int) -> str:
        """Resolve an output port index to a name via schema lookup."""
        # Try all contexts — node type names in HDA graphs don't include context
        for schema in self.schema_corpus.nodes.values():
            if schema.type_name == node_type:
                for port in schema.outputs:
                    if port.index == port_index:
                        return port.label or port.name
        return ""

    def _resolve_input_name(self, node_type: str, port_index: int) -> str:
        """Resolve an input port index to a name via schema lookup."""
        for schema in self.schema_corpus.nodes.values():
            if schema.type_name == node_type:
                for port in schema.inputs:
                    if port.index == port_index:
                        return port.label or port.name
        return ""

    def _enrich_connection_patterns(self, patterns: PatternCorpus) -> None:
        """Add port names to connection patterns."""
        for pattern in patterns.connection_patterns.values():
            if not pattern.source_output_name:
                name = self._resolve_output_name(pattern.source_type, pattern.source_output)
                if name:
                    pattern.source_output_name = name
            if not pattern.dest_input_name:
                name = self._resolve_input_name(pattern.dest_type, pattern.dest_input)
                if name:
                    pattern.dest_input_name = name

    def _enrich_port_usage(self, patterns: PatternCorpus) -> None:
        """Add port names to port usage statistics."""
        for npu in patterns.port_usage.values():
            for stat in npu.inputs:
                if not stat.port_name:
                    name = self._resolve_input_name(npu.node_type, stat.port_index)
                    if name:
                        stat.port_name = name
            for stat in npu.outputs:
                if not stat.port_name:
                    name = self._resolve_output_name(npu.node_type, stat.port_index)
                    if name:
                        stat.port_name = name
