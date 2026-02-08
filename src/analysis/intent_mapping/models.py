"""Data models for Phase 2B: Intent-to-Subgraph Mapping."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SubgraphTemplate:
    """One HDA as a reusable subgraph template."""

    hda_key: str
    label: str
    category: str
    context: str
    node_types: list[str] = field(default_factory=list)
    node_count: int = 0
    connection_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hda_key": self.hda_key,
            "label": self.label,
            "category": self.category,
            "context": self.context,
            "node_types": self.node_types,
            "node_count": self.node_count,
            "connection_count": self.connection_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SubgraphTemplate:
        return cls(
            hda_key=d["hda_key"],
            label=d["label"],
            category=d["category"],
            context=d["context"],
            node_types=d.get("node_types", []),
            node_count=d.get("node_count", 0),
            connection_count=d.get("connection_count", 0),
        )


@dataclass
class IntentCluster:
    """All templates for one high-level intent."""

    intent_id: str
    keywords: list[str] = field(default_factory=list)
    description: str = ""
    category: str = ""
    templates: list[SubgraphTemplate] = field(default_factory=list)

    @property
    def template_count(self) -> int:
        return len(self.templates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "keywords": self.keywords,
            "description": self.description,
            "category": self.category,
            "templates": [t.to_dict() for t in self.templates],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IntentCluster:
        cluster = cls(
            intent_id=d["intent_id"],
            keywords=d.get("keywords", []),
            description=d.get("description", ""),
            category=d.get("category", ""),
        )
        for t in d.get("templates", []):
            cluster.templates.append(SubgraphTemplate.from_dict(t))
        return cluster


class IntentLibrary:
    """Intent-indexed template library queryable by keyword.

    The output corpus for Phase 2B. Maps high-level user intents to
    subgraph templates extracted from Labs HDAs.
    """

    def __init__(self) -> None:
        self.intents: dict[str, IntentCluster] = {}

    @property
    def intent_count(self) -> int:
        return len(self.intents)

    @property
    def template_count(self) -> int:
        return sum(c.template_count for c in self.intents.values())

    def search(self, query: str, limit: int = 10) -> list[IntentCluster]:
        """Keyword search across intent descriptions and keywords."""
        query_tokens = query.lower().split()
        scored: list[tuple[int, str, IntentCluster]] = []
        for cluster in self.intents.values():
            score = 0
            searchable = cluster.keywords + cluster.description.lower().split()
            for qt in query_tokens:
                for token in searchable:
                    if qt in token:
                        score += 1
            if score > 0:
                scored.append((score, cluster.intent_id, cluster))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [c for _, _, c in scored[:limit]]

    def get_by_category(self, context: str) -> list[IntentCluster]:
        """Get all intent clusters for a given context (e.g. 'sop')."""
        return sorted(
            [c for c in self.intents.values() if c.category == context],
            key=lambda c: c.intent_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "intent_count": self.intent_count,
            "template_count": self.template_count,
            "intents": {
                k: v.to_dict() for k, v in sorted(self.intents.items())
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> IntentLibrary:
        lib = cls()
        for k, v in d.get("intents", {}).items():
            lib.intents[k] = IntentCluster.from_dict(v)
        return lib

    def save_json(self, path: Path | str) -> None:
        """Write library to JSON file with deterministic output."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def load_json(cls, path: Path | str) -> IntentLibrary:
        """Load library from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
