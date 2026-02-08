"""Data models for extracted Houdini node type schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CATEGORY_TO_CONTEXT = {
    "Sop": "sop",
    "Dop": "dop",
    "Vop": "vop",
    "Object": "obj",
    "Driver": "out",
    "Cop2": "cop2",
    "Chop": "chop",
    "Shop": "shop",
    "Top": "top",
    "Lop": "lop",
}


@dataclass
class ParmSchema:
    """Schema for a single parameter template."""

    name: str
    label: str = ""
    type: str = ""
    size: int = 1
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    min_is_strict: bool = False
    max_is_strict: bool = False
    menu_items: list[str] = field(default_factory=list)
    menu_labels: list[str] = field(default_factory=list)
    is_hidden: bool = False
    help: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    disable_when: str = ""
    hide_when: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.label:
            d["label"] = self.label
        if self.type:
            d["type"] = self.type
        if self.size != 1:
            d["size"] = self.size
        if self.default is not None:
            d["default"] = self.default
        if self.min_value is not None:
            d["min_value"] = self.min_value
        if self.max_value is not None:
            d["max_value"] = self.max_value
        if self.min_is_strict:
            d["min_is_strict"] = True
        if self.max_is_strict:
            d["max_is_strict"] = True
        if self.menu_items:
            d["menu_items"] = self.menu_items
        if self.menu_labels:
            d["menu_labels"] = self.menu_labels
        if self.is_hidden:
            d["is_hidden"] = True
        if self.help:
            d["help"] = self.help
        if self.tags:
            d["tags"] = self.tags
        if self.disable_when:
            d["disable_when"] = self.disable_when
        if self.hide_when:
            d["hide_when"] = self.hide_when
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParmSchema:
        return cls(
            name=d["name"],
            label=d.get("label", ""),
            type=d.get("type", ""),
            size=d.get("size", 1),
            default=d.get("default"),
            min_value=d.get("min_value"),
            max_value=d.get("max_value"),
            min_is_strict=d.get("min_is_strict", False),
            max_is_strict=d.get("max_is_strict", False),
            menu_items=d.get("menu_items", []),
            menu_labels=d.get("menu_labels", []),
            is_hidden=d.get("is_hidden", False),
            help=d.get("help", ""),
            tags=d.get("tags", {}),
            disable_when=d.get("disable_when", ""),
            hide_when=d.get("hide_when", ""),
        )


@dataclass
class PortSchema:
    """Schema for a single input/output connector."""

    index: int
    name: str
    label: str = ""
    type: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"index": self.index, "name": self.name}
        if self.label:
            d["label"] = self.label
        if self.type:
            d["type"] = self.type
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PortSchema:
        return cls(
            index=d["index"],
            name=d["name"],
            label=d.get("label", ""),
            type=d.get("type", ""),
        )


@dataclass
class NodeTypeSchema:
    """Schema for a single Houdini node type."""

    category: str
    type_name: str
    label: str = ""

    # Namespace decomposition from nameComponents()
    scope_namespace: str = ""
    namespace: str = ""
    base_type: str = ""
    version: str = ""

    # Ports
    inputs: list[PortSchema] = field(default_factory=list)
    outputs: list[PortSchema] = field(default_factory=list)
    min_inputs: int = 0
    max_inputs: int = 0
    max_outputs: int = 0

    # Parameters (flattened from folder hierarchy)
    parameters: list[ParmSchema] = field(default_factory=list)

    # Metadata
    icon: str = ""
    is_generator: bool = False
    unordered_inputs: bool = False
    deprecated: bool = False
    is_hda: bool = False

    @property
    def key(self) -> str:
        """Canonical key: context/type_name (e.g. 'sop/scatter')."""
        return f"{self.context}/{self.type_name}"

    @property
    def context(self) -> str:
        """Map category to help docs context string."""
        return CATEGORY_TO_CONTEXT.get(self.category, self.category.lower())

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "category": self.category,
            "type_name": self.type_name,
        }
        if self.label:
            d["label"] = self.label
        if self.scope_namespace:
            d["scope_namespace"] = self.scope_namespace
        if self.namespace:
            d["namespace"] = self.namespace
        if self.base_type:
            d["base_type"] = self.base_type
        if self.version:
            d["version"] = self.version
        if self.inputs:
            d["inputs"] = [p.to_dict() for p in self.inputs]
        if self.outputs:
            d["outputs"] = [p.to_dict() for p in self.outputs]
        if self.min_inputs:
            d["min_inputs"] = self.min_inputs
        if self.max_inputs:
            d["max_inputs"] = self.max_inputs
        if self.max_outputs:
            d["max_outputs"] = self.max_outputs
        if self.parameters:
            d["parameters"] = [p.to_dict() for p in self.parameters]
        if self.icon:
            d["icon"] = self.icon
        if self.is_generator:
            d["is_generator"] = True
        if self.unordered_inputs:
            d["unordered_inputs"] = True
        if self.deprecated:
            d["deprecated"] = True
        if self.is_hda:
            d["is_hda"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeTypeSchema:
        schema = cls(
            category=d["category"],
            type_name=d["type_name"],
            label=d.get("label", ""),
            scope_namespace=d.get("scope_namespace", ""),
            namespace=d.get("namespace", ""),
            base_type=d.get("base_type", ""),
            version=d.get("version", ""),
            min_inputs=d.get("min_inputs", 0),
            max_inputs=d.get("max_inputs", 0),
            max_outputs=d.get("max_outputs", 0),
            icon=d.get("icon", ""),
            is_generator=d.get("is_generator", False),
            unordered_inputs=d.get("unordered_inputs", False),
            deprecated=d.get("deprecated", False),
            is_hda=d.get("is_hda", False),
        )
        for p in d.get("inputs", []):
            schema.inputs.append(PortSchema.from_dict(p))
        for p in d.get("outputs", []):
            schema.outputs.append(PortSchema.from_dict(p))
        for p in d.get("parameters", []):
            schema.parameters.append(ParmSchema.from_dict(p))
        return schema


class SchemaCorpus:
    """Container for all extracted node type schemas."""

    def __init__(
        self,
        houdini_version: str = "",
        extraction_timestamp: str = "",
    ) -> None:
        self.nodes: dict[str, NodeTypeSchema] = {}
        self.houdini_version = houdini_version
        self.extraction_timestamp = extraction_timestamp

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def add(self, schema: NodeTypeSchema) -> None:
        """Add a node type schema, keyed by context/type_name."""
        self.nodes[schema.key] = schema

    def get_node(self, key: str) -> NodeTypeSchema | None:
        """Lookup by 'context/type_name' key."""
        return self.nodes.get(key)

    def get_by_context(self, context: str) -> list[NodeTypeSchema]:
        """Get all schemas for a given context (e.g. 'sop')."""
        return [s for s in self.nodes.values() if s.context == context]

    def contexts(self) -> list[str]:
        """Return sorted list of unique contexts."""
        return sorted({s.context for s in self.nodes.values()})

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire corpus to a JSON-compatible dict."""
        return {
            "version": "1.0",
            "houdini_version": self.houdini_version,
            "extraction_timestamp": self.extraction_timestamp,
            "node_count": self.node_count,
            "contexts": self.contexts(),
            "nodes": {
                key: schema.to_dict()
                for key, schema in sorted(self.nodes.items())
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SchemaCorpus:
        """Deserialize from a dict."""
        corpus = cls(
            houdini_version=d.get("houdini_version", ""),
            extraction_timestamp=d.get("extraction_timestamp", ""),
        )
        for key, node_d in d.get("nodes", {}).items():
            corpus.nodes[key] = NodeTypeSchema.from_dict(node_d)
        return corpus

    def save_json(self, path: Path | str) -> None:
        """Write corpus to JSON file with deterministic output."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def load_json(cls, path: Path | str) -> SchemaCorpus:
        """Load corpus from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
