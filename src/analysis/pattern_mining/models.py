"""Data models for mined connection patterns from Labs HDA graphs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConnectionPattern:
    """One directed edge type observed across HDAs.

    Tracks how often a (source_type, source_output) -> (dest_type, dest_input)
    edge appears, and in which HDAs.
    """

    source_type: str
    dest_type: str
    source_output: int
    dest_input: int
    count: int = 0
    hda_keys: list[str] = field(default_factory=list)
    source_output_name: str = ""
    dest_input_name: str = ""

    @property
    def edge_key(self) -> str:
        """Canonical key: 'source_type:output->dest_type:input'."""
        return f"{self.source_type}:{self.source_output}->{self.dest_type}:{self.dest_input}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source_type": self.source_type,
            "dest_type": self.dest_type,
            "source_output": self.source_output,
            "dest_input": self.dest_input,
            "count": self.count,
            "hda_keys": self.hda_keys,
        }
        if self.source_output_name:
            d["source_output_name"] = self.source_output_name
        if self.dest_input_name:
            d["dest_input_name"] = self.dest_input_name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConnectionPattern:
        return cls(
            source_type=d["source_type"],
            dest_type=d["dest_type"],
            source_output=d["source_output"],
            dest_input=d["dest_input"],
            count=d["count"],
            hda_keys=d.get("hda_keys", []),
            source_output_name=d.get("source_output_name", ""),
            dest_input_name=d.get("dest_input_name", ""),
        )


@dataclass
class NodeCooccurrence:
    """Two node types co-occurring in the same HDA.

    Types are stored in lexicographic order to avoid duplicates.
    """

    type_a: str
    type_b: str
    count: int = 0
    type_a_total: int = 0
    type_b_total: int = 0
    jaccard: float = 0.0

    @property
    def pair_key(self) -> str:
        """Canonical key: 'type_a+type_b'."""
        return f"{self.type_a}+{self.type_b}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_a": self.type_a,
            "type_b": self.type_b,
            "count": self.count,
            "type_a_total": self.type_a_total,
            "type_b_total": self.type_b_total,
            "jaccard": round(self.jaccard, 4),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeCooccurrence:
        return cls(
            type_a=d["type_a"],
            type_b=d["type_b"],
            count=d["count"],
            type_a_total=d.get("type_a_total", 0),
            type_b_total=d.get("type_b_total", 0),
            jaccard=d.get("jaccard", 0.0),
        )


@dataclass
class PortUsageStat:
    """Usage statistics for one port of a node type."""

    port_index: int
    port_name: str = ""
    usage_count: int = 0
    total_appearances: int = 0
    usage_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "port_index": self.port_index,
            "usage_count": self.usage_count,
            "total_appearances": self.total_appearances,
            "usage_ratio": round(self.usage_ratio, 4),
        }
        if self.port_name:
            d["port_name"] = self.port_name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PortUsageStat:
        return cls(
            port_index=d["port_index"],
            port_name=d.get("port_name", ""),
            usage_count=d["usage_count"],
            total_appearances=d["total_appearances"],
            usage_ratio=d.get("usage_ratio", 0.0),
        )


@dataclass
class NodePortUsage:
    """All port usage stats for one node type."""

    node_type: str
    context: str = ""
    total_appearances: int = 0
    inputs: list[PortUsageStat] = field(default_factory=list)
    outputs: list[PortUsageStat] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "node_type": self.node_type,
            "total_appearances": self.total_appearances,
        }
        if self.context:
            d["context"] = self.context
        if self.inputs:
            d["inputs"] = [p.to_dict() for p in self.inputs]
        if self.outputs:
            d["outputs"] = [p.to_dict() for p in self.outputs]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodePortUsage:
        npu = cls(
            node_type=d["node_type"],
            context=d.get("context", ""),
            total_appearances=d["total_appearances"],
        )
        for p in d.get("inputs", []):
            npu.inputs.append(PortUsageStat.from_dict(p))
        for p in d.get("outputs", []):
            npu.outputs.append(PortUsageStat.from_dict(p))
        return npu


@dataclass
class ChainPattern:
    """Sequential chain of 2-3 node types connected in series."""

    types: list[str] = field(default_factory=list)
    count: int = 0
    hda_keys: list[str] = field(default_factory=list)

    @property
    def chain_key(self) -> str:
        """Canonical key: 'typeA -> typeB -> typeC'."""
        return " -> ".join(self.types)

    @property
    def length(self) -> int:
        return len(self.types)

    def to_dict(self) -> dict[str, Any]:
        return {
            "types": self.types,
            "count": self.count,
            "hda_keys": self.hda_keys,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChainPattern:
        return cls(
            types=d["types"],
            count=d["count"],
            hda_keys=d.get("hda_keys", []),
        )


@dataclass
class DownstreamSuggestion:
    """A ranked downstream neighbor suggestion for a node type."""

    target_type: str
    count: int = 0
    source_output: int = 0
    dest_input: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_type": self.target_type,
            "count": self.count,
            "source_output": self.source_output,
            "dest_input": self.dest_input,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DownstreamSuggestion:
        return cls(
            target_type=d["target_type"],
            count=d["count"],
            source_output=d.get("source_output", 0),
            dest_input=d.get("dest_input", 0),
        )


@dataclass
class UpstreamSuggestion:
    """A ranked upstream neighbor suggestion for a node type."""

    source_type: str
    count: int = 0
    source_output: int = 0
    dest_input: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "count": self.count,
            "source_output": self.source_output,
            "dest_input": self.dest_input,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> UpstreamSuggestion:
        return cls(
            source_type=d["source_type"],
            count=d["count"],
            source_output=d.get("source_output", 0),
            dest_input=d.get("dest_input", 0),
        )


@dataclass
class NodeSuggestions:
    """Per-node-type suggestion lists for downstream and upstream neighbors."""

    node_type: str
    context: str = ""
    downstream: list[DownstreamSuggestion] = field(default_factory=list)
    upstream: list[UpstreamSuggestion] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"node_type": self.node_type}
        if self.context:
            d["context"] = self.context
        if self.downstream:
            d["downstream"] = [s.to_dict() for s in self.downstream]
        if self.upstream:
            d["upstream"] = [s.to_dict() for s in self.upstream]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeSuggestions:
        ns = cls(
            node_type=d["node_type"],
            context=d.get("context", ""),
        )
        for s in d.get("downstream", []):
            ns.downstream.append(DownstreamSuggestion.from_dict(s))
        for s in d.get("upstream", []):
            ns.upstream.append(UpstreamSuggestion.from_dict(s))
        return ns


class PatternCorpus:
    """Container for all mined connection patterns.

    Same API pattern as HDAGraphCorpus and SchemaCorpus: dict-based storage
    with save_json/load_json serialization.
    """

    def __init__(self) -> None:
        self.connection_patterns: dict[str, ConnectionPattern] = {}
        self.node_suggestions: dict[str, NodeSuggestions] = {}
        self.cooccurrences: dict[str, NodeCooccurrence] = {}
        self.port_usage: dict[str, NodePortUsage] = {}
        self.chain_patterns_2: dict[str, ChainPattern] = {}
        self.chain_patterns_3: dict[str, ChainPattern] = {}

    @property
    def pattern_count(self) -> int:
        return len(self.connection_patterns)

    @property
    def suggestion_count(self) -> int:
        return len(self.node_suggestions)

    def get_downstream(self, node_type: str, limit: int = 10) -> list[DownstreamSuggestion]:
        """Get ranked downstream suggestions for a node type."""
        ns = self.node_suggestions.get(node_type)
        if ns is None:
            return []
        return ns.downstream[:limit]

    def get_upstream(self, node_type: str, limit: int = 10) -> list[UpstreamSuggestion]:
        """Get ranked upstream suggestions for a node type."""
        ns = self.node_suggestions.get(node_type)
        if ns is None:
            return []
        return ns.upstream[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "pattern_count": self.pattern_count,
            "suggestion_count": self.suggestion_count,
            "connection_patterns": {
                k: v.to_dict() for k, v in sorted(self.connection_patterns.items())
            },
            "node_suggestions": {
                k: v.to_dict() for k, v in sorted(self.node_suggestions.items())
            },
            "cooccurrences": {
                k: v.to_dict() for k, v in sorted(self.cooccurrences.items())
            },
            "port_usage": {
                k: v.to_dict() for k, v in sorted(self.port_usage.items())
            },
            "chain_patterns_2": {
                k: v.to_dict() for k, v in sorted(self.chain_patterns_2.items())
            },
            "chain_patterns_3": {
                k: v.to_dict() for k, v in sorted(self.chain_patterns_3.items())
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PatternCorpus:
        corpus = cls()
        for k, v in d.get("connection_patterns", {}).items():
            corpus.connection_patterns[k] = ConnectionPattern.from_dict(v)
        for k, v in d.get("node_suggestions", {}).items():
            corpus.node_suggestions[k] = NodeSuggestions.from_dict(v)
        for k, v in d.get("cooccurrences", {}).items():
            corpus.cooccurrences[k] = NodeCooccurrence.from_dict(v)
        for k, v in d.get("port_usage", {}).items():
            corpus.port_usage[k] = NodePortUsage.from_dict(v)
        for k, v in d.get("chain_patterns_2", {}).items():
            corpus.chain_patterns_2[k] = ChainPattern.from_dict(v)
        for k, v in d.get("chain_patterns_3", {}).items():
            corpus.chain_patterns_3[k] = ChainPattern.from_dict(v)
        return corpus

    def save_json(self, path: Path | str) -> None:
        """Write corpus to JSON file with deterministic output."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def load_json(cls, path: Path | str) -> PatternCorpus:
        """Load corpus from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
