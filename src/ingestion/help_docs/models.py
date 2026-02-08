"""Data models for parsed Houdini help documentation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class HelpMenuOption:
    """A menu option within a parameter."""

    label: str
    description: str = ""


@dataclass
class HelpParameter:
    """A single parameter from a node's help documentation."""

    label: str
    id: str = ""
    description: str = ""
    channels: str = ""
    menu_options: list[HelpMenuOption] = field(default_factory=list)


@dataclass
class HelpParameterGroup:
    """A group of parameters under a ~~~ subsection ~~~."""

    label: str
    parameters: list[HelpParameter] = field(default_factory=list)


@dataclass
class HelpPort:
    """An input or output port description."""

    name: str
    type: str = ""
    description: str = ""


@dataclass
class HelpSection:
    """A content section (== Section ==) from the help body."""

    title: str
    content: str = ""


@dataclass
class HelpLocalVar:
    """A local variable from @locals."""

    name: str
    description: str = ""


@dataclass
class HelpTopAttribute:
    """A TOP attribute from @top_attributes."""

    name: str
    type: str = ""
    description: str = ""


@dataclass
class NodeHelpDoc:
    """Parsed help documentation for a single Houdini node."""

    # Identity
    context: str = ""
    internal_name: str = ""
    namespace: str = ""

    # Header metadata
    title: str = ""
    icon: str = ""
    tags: list[str] = field(default_factory=list)
    since_version: str = ""
    version: str = ""

    # Brief description (the triple-quoted line)
    brief: str = ""

    # Body content sections (== Section ==)
    sections: list[HelpSection] = field(default_factory=list)

    # Parameters (flat list and grouped)
    parameters: list[HelpParameter] = field(default_factory=list)
    parameter_groups: list[HelpParameterGroup] = field(default_factory=list)

    # Inputs / Outputs
    inputs: list[HelpPort] = field(default_factory=list)
    outputs: list[HelpPort] = field(default_factory=list)

    # Related nodes
    related: list[str] = field(default_factory=list)

    # Local variables (@locals)
    locals: list[HelpLocalVar] = field(default_factory=list)

    # TOP attributes (@top_attributes)
    top_attributes: list[HelpTopAttribute] = field(default_factory=list)

    # Include directives (recorded, not resolved)
    includes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Canonical key: context/internal_name."""
        return f"{self.context}/{self.internal_name}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d: dict[str, Any] = {
            "context": self.context,
            "internal_name": self.internal_name,
            "title": self.title,
            "brief": self.brief,
        }
        if self.namespace:
            d["namespace"] = self.namespace
        if self.icon:
            d["icon"] = self.icon
        if self.tags:
            d["tags"] = self.tags
        if self.since_version:
            d["since_version"] = self.since_version
        if self.version:
            d["version"] = self.version
        if self.sections:
            d["sections"] = [
                {"title": s.title, "content": s.content} for s in self.sections
            ]
        if self.parameters:
            d["parameters"] = [_param_to_dict(p) for p in self.parameters]
        if self.parameter_groups:
            d["parameter_groups"] = [
                {
                    "label": g.label,
                    "parameters": [_param_to_dict(p) for p in g.parameters],
                }
                for g in self.parameter_groups
            ]
        if self.inputs:
            d["inputs"] = [_port_to_dict(p) for p in self.inputs]
        if self.outputs:
            d["outputs"] = [_port_to_dict(p) for p in self.outputs]
        if self.related:
            d["related"] = self.related
        if self.locals:
            d["locals"] = [
                {"name": v.name, "description": v.description} for v in self.locals
            ]
        if self.top_attributes:
            d["top_attributes"] = [
                {"name": a.name, "type": a.type, "description": a.description}
                for a in self.top_attributes
            ]
        if self.includes:
            d["includes"] = self.includes
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NodeHelpDoc:
        """Deserialize from a dict."""
        doc = cls(
            context=d.get("context", ""),
            internal_name=d.get("internal_name", ""),
            namespace=d.get("namespace", ""),
            title=d.get("title", ""),
            icon=d.get("icon", ""),
            tags=d.get("tags", []),
            since_version=d.get("since_version", ""),
            version=d.get("version", ""),
            brief=d.get("brief", ""),
            related=d.get("related", []),
            includes=d.get("includes", []),
        )
        for s in d.get("sections", []):
            doc.sections.append(HelpSection(title=s["title"], content=s["content"]))
        for p in d.get("parameters", []):
            doc.parameters.append(_param_from_dict(p))
        for g in d.get("parameter_groups", []):
            doc.parameter_groups.append(
                HelpParameterGroup(
                    label=g["label"],
                    parameters=[_param_from_dict(p) for p in g["parameters"]],
                )
            )
        for p in d.get("inputs", []):
            doc.inputs.append(HelpPort(name=p["name"], type=p.get("type", ""), description=p.get("description", "")))
        for p in d.get("outputs", []):
            doc.outputs.append(HelpPort(name=p["name"], type=p.get("type", ""), description=p.get("description", "")))
        for v in d.get("locals", []):
            doc.locals.append(HelpLocalVar(name=v["name"], description=v.get("description", "")))
        for a in d.get("top_attributes", []):
            doc.top_attributes.append(
                HelpTopAttribute(name=a["name"], type=a.get("type", ""), description=a.get("description", ""))
            )
        return doc


class HelpCorpus:
    """Container for all parsed node help documents."""

    def __init__(self) -> None:
        self.nodes: dict[str, NodeHelpDoc] = {}

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def add(self, doc: NodeHelpDoc) -> None:
        """Add a parsed doc, keyed by context/internal_name."""
        self.nodes[doc.key] = doc

    def get_node(self, key: str) -> NodeHelpDoc | None:
        """Lookup by 'context/internal_name' key."""
        return self.nodes.get(key)

    def get_by_context(self, context: str) -> list[NodeHelpDoc]:
        """Get all docs for a given context (e.g. 'sop')."""
        return [doc for doc in self.nodes.values() if doc.context == context]

    def contexts(self) -> list[str]:
        """Return sorted list of unique contexts."""
        return sorted({doc.context for doc in self.nodes.values()})

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire corpus to a JSON-compatible dict."""
        return {
            "version": "1.0",
            "node_count": self.node_count,
            "contexts": self.contexts(),
            "nodes": {
                key: doc.to_dict()
                for key, doc in sorted(self.nodes.items())
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HelpCorpus:
        """Deserialize from a dict."""
        corpus = cls()
        for key, node_d in d.get("nodes", {}).items():
            corpus.nodes[key] = NodeHelpDoc.from_dict(node_d)
        return corpus

    def save_json(self, path: Path | str) -> None:
        """Write corpus to JSON file with deterministic output."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def load_json(cls, path: Path | str) -> HelpCorpus:
        """Load corpus from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


def _param_to_dict(p: HelpParameter) -> dict[str, Any]:
    d: dict[str, Any] = {"label": p.label}
    if p.id:
        d["id"] = p.id
    if p.description:
        d["description"] = p.description
    if p.channels:
        d["channels"] = p.channels
    if p.menu_options:
        d["menu_options"] = [
            {"label": m.label, "description": m.description} for m in p.menu_options
        ]
    return d


def _param_from_dict(d: dict[str, Any]) -> HelpParameter:
    p = HelpParameter(
        label=d["label"],
        id=d.get("id", ""),
        description=d.get("description", ""),
        channels=d.get("channels", ""),
    )
    for m in d.get("menu_options", []):
        p.menu_options.append(HelpMenuOption(label=m["label"], description=m.get("description", "")))
    return p


def _port_to_dict(p: HelpPort) -> dict[str, Any]:
    d: dict[str, Any] = {"name": p.name}
    if p.type:
        d["type"] = p.type
    if p.description:
        d["description"] = p.description
    return d
