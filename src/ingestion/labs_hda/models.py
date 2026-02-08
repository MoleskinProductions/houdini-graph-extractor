"""Data models for extracted Labs HDA internal graphs."""

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
class HDANodeParameter:
    """A single non-default parameter value on an internal node."""

    name: str
    value: Any = None
    expression: str | None = None
    expression_language: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.value is not None:
            d["value"] = self.value
        if self.expression is not None:
            d["expression"] = self.expression
        if self.expression_language:
            d["expression_language"] = self.expression_language
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HDANodeParameter:
        return cls(
            name=d["name"],
            value=d.get("value"),
            expression=d.get("expression"),
            expression_language=d.get("expression_language", ""),
        )


@dataclass
class HDAInternalNode:
    """One child node inside an HDA subnet."""

    name: str
    type: str
    position: tuple[float, float] = (0.0, 0.0)
    display_flag: bool = False
    render_flag: bool = False
    bypass_flag: bool = False
    parameters: list[HDANodeParameter] = field(default_factory=list)
    is_output: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
        }
        if self.position != (0.0, 0.0):
            d["position"] = list(self.position)
        if self.display_flag:
            d["display_flag"] = True
        if self.render_flag:
            d["render_flag"] = True
        if self.bypass_flag:
            d["bypass_flag"] = True
        if self.parameters:
            d["parameters"] = [p.to_dict() for p in self.parameters]
        if self.is_output:
            d["is_output"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HDAInternalNode:
        pos = d.get("position", [0.0, 0.0])
        node = cls(
            name=d["name"],
            type=d["type"],
            position=tuple(pos),
            display_flag=d.get("display_flag", False),
            render_flag=d.get("render_flag", False),
            bypass_flag=d.get("bypass_flag", False),
            is_output=d.get("is_output", False),
        )
        for p in d.get("parameters", []):
            node.parameters.append(HDANodeParameter.from_dict(p))
        return node


@dataclass
class HDAConnection:
    """One internal wire between child nodes."""

    source_node: str
    source_output: int
    dest_node: str
    dest_input: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node": self.source_node,
            "source_output": self.source_output,
            "dest_node": self.dest_node,
            "dest_input": self.dest_input,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HDAConnection:
        return cls(
            source_node=d["source_node"],
            source_output=d["source_output"],
            dest_node=d["dest_node"],
            dest_input=d["dest_input"],
        )


@dataclass
class HDASubnetInput:
    """One HDA interface input with its internal connections."""

    index: int
    name: str
    connections: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "index": self.index,
            "name": self.name,
        }
        if self.connections:
            d["connections"] = [
                {"dest_node": dest_node, "dest_input": dest_input}
                for dest_node, dest_input in self.connections
            ]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HDASubnetInput:
        conns = [
            (c["dest_node"], c["dest_input"])
            for c in d.get("connections", [])
        ]
        return cls(
            index=d["index"],
            name=d["name"],
            connections=conns,
        )


@dataclass
class HDAGraph:
    """Complete internal graph of one Labs HDA."""

    # Identity
    type_name: str
    category: str
    label: str = ""
    library_path: str = ""

    # Interface
    min_inputs: int = 0
    max_inputs: int = 0
    max_outputs: int = 0
    subnet_inputs: list[HDASubnetInput] = field(default_factory=list)

    # Graph
    nodes: list[HDAInternalNode] = field(default_factory=list)
    connections: list[HDAConnection] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Canonical key: context/type_name (e.g. 'sop/labs::quickmaterial::2.0')."""
        return f"{self.context}/{self.type_name}"

    @property
    def context(self) -> str:
        """Map category to context string."""
        return CATEGORY_TO_CONTEXT.get(self.category, self.category.lower())

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def connection_count(self) -> int:
        return len(self.connections)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type_name": self.type_name,
            "category": self.category,
        }
        if self.label:
            d["label"] = self.label
        if self.library_path:
            d["library_path"] = self.library_path
        if self.min_inputs:
            d["min_inputs"] = self.min_inputs
        if self.max_inputs:
            d["max_inputs"] = self.max_inputs
        if self.max_outputs:
            d["max_outputs"] = self.max_outputs
        if self.subnet_inputs:
            d["subnet_inputs"] = [si.to_dict() for si in self.subnet_inputs]
        if self.nodes:
            d["nodes"] = [n.to_dict() for n in self.nodes]
        if self.connections:
            d["connections"] = [c.to_dict() for c in self.connections]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HDAGraph:
        graph = cls(
            type_name=d["type_name"],
            category=d["category"],
            label=d.get("label", ""),
            library_path=d.get("library_path", ""),
            min_inputs=d.get("min_inputs", 0),
            max_inputs=d.get("max_inputs", 0),
            max_outputs=d.get("max_outputs", 0),
        )
        for si in d.get("subnet_inputs", []):
            graph.subnet_inputs.append(HDASubnetInput.from_dict(si))
        for n in d.get("nodes", []):
            graph.nodes.append(HDAInternalNode.from_dict(n))
        for c in d.get("connections", []):
            graph.connections.append(HDAConnection.from_dict(c))
        return graph

    def to_hougraph_ir(self):
        """Convert to a HouGraphIR instance.

        Requires the hougraph-ir package. Raises ImportError if not installed.
        """
        try:
            from hougraph_ir import (
                Connection,
                HouGraphIR,
                NetworkContext,
                NodeDefinition,
                ParameterValue,
                Position,
            )
        except ImportError:
            raise ImportError(
                "hougraph-ir package is required for IR conversion. "
                "Install with: pip install hougraph-ir"
            )

        # Map category to NetworkContext
        context_map = {
            "sop": NetworkContext.SOP,
            "dop": NetworkContext.DOP,
            "vop": NetworkContext.VOP,
            "obj": NetworkContext.OBJ,
            "out": NetworkContext.ROP,
            "cop2": NetworkContext.COP,
            "chop": NetworkContext.CHOP,
            "shop": NetworkContext.SHOP,
            "top": NetworkContext.TOP,
            "lop": NetworkContext.LOP,
        }
        network_context = context_map.get(self.context, NetworkContext.SOP)

        ir = HouGraphIR(
            context=network_context,
            parent_path=f"/obj/{self.type_name}",
            source={
                "type": "hda_introspection",
                "hda_type": self.type_name,
                "library": self.library_path,
            },
        )

        # Add subnet input pseudo-nodes
        for si in self.subnet_inputs:
            pseudo_name = f"__subnet_input_{si.index}"
            ir.add_node(NodeDefinition(
                name=pseudo_name,
                type="subnetinput",
                _extraction_confidence=1.0,
            ))
            for dest_node, dest_input in si.connections:
                ir.add_connection(Connection(
                    source_node=pseudo_name,
                    source_output=0,
                    dest_node=dest_node,
                    dest_input=dest_input,
                ))

        # Add internal nodes
        for node in self.nodes:
            params = []
            for p in node.parameters:
                params.append(ParameterValue(
                    name=p.name,
                    value=p.value,
                    expression=p.expression,
                    expression_language=p.expression_language or "hscript",
                ))
            ir.add_node(NodeDefinition(
                name=node.name,
                type=node.type,
                position=Position(x=node.position[0], y=node.position[1]),
                parameters=params,
                display_flag=node.display_flag,
                render_flag=node.render_flag,
                bypass_flag=node.bypass_flag,
                _extraction_confidence=1.0,
            ))

        # Add connections
        for conn in self.connections:
            ir.add_connection(Connection(
                source_node=conn.source_node,
                source_output=conn.source_output,
                dest_node=conn.dest_node,
                dest_input=conn.dest_input,
            ))

        return ir


class HDAGraphCorpus:
    """Container for all extracted Labs HDA internal graphs."""

    def __init__(
        self,
        houdini_version: str = "",
        extraction_timestamp: str = "",
    ) -> None:
        self.graphs: dict[str, HDAGraph] = {}
        self.houdini_version = houdini_version
        self.extraction_timestamp = extraction_timestamp

    @property
    def hda_count(self) -> int:
        return len(self.graphs)

    def add(self, graph: HDAGraph) -> None:
        """Add an HDA graph, keyed by context/type_name."""
        self.graphs[graph.key] = graph

    def get_graph(self, key: str) -> HDAGraph | None:
        """Lookup by 'context/type_name' key."""
        return self.graphs.get(key)

    def get_by_context(self, context: str) -> list[HDAGraph]:
        """Get all graphs for a given context (e.g. 'sop')."""
        return [g for g in self.graphs.values() if g.context == context]

    def contexts(self) -> list[str]:
        """Return sorted list of unique contexts."""
        return sorted({g.context for g in self.graphs.values()})

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire corpus to a JSON-compatible dict."""
        return {
            "version": "1.0",
            "houdini_version": self.houdini_version,
            "extraction_timestamp": self.extraction_timestamp,
            "hda_count": self.hda_count,
            "contexts": self.contexts(),
            "graphs": {
                key: graph.to_dict()
                for key, graph in sorted(self.graphs.items())
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HDAGraphCorpus:
        """Deserialize from a dict."""
        corpus = cls(
            houdini_version=d.get("houdini_version", ""),
            extraction_timestamp=d.get("extraction_timestamp", ""),
        )
        for key, graph_d in d.get("graphs", {}).items():
            corpus.graphs[key] = HDAGraph.from_dict(graph_d)
        return corpus

    def save_json(self, path: Path | str) -> None:
        """Write corpus to JSON file with deterministic output."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=False, ensure_ascii=False)
            f.write("\n")

    @classmethod
    def load_json(cls, path: Path | str) -> HDAGraphCorpus:
        """Load corpus from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
