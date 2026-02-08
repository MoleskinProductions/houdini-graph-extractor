"""Wrapper that launches hython to extract Labs HDA internal graphs."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .models import (
    HDAConnection,
    HDAGraph,
    HDAGraphCorpus,
    HDAInternalNode,
    HDANodeParameter,
    HDASubnetInput,
)

# Location of the hython extraction script, relative to this file.
_HYTHON_SCRIPT = Path(__file__).parent / "_hython_extract.py"

DEFAULT_HYTHON_PATH = Path("/opt/hfs21.0/bin/hython")


class LabsHDAExtractor:
    """Extracts Labs HDA internal graphs by running a hython subprocess."""

    def __init__(
        self,
        hython_path: str | Path = DEFAULT_HYTHON_PATH,
        categories: list[str] | None = None,
        library_filter: str = "SideFXLabs",
        timeout: int = 300,
    ) -> None:
        self.hython_path = Path(hython_path)
        self.categories = categories
        self.library_filter = library_filter
        self.timeout = timeout

    def extract(self) -> HDAGraphCorpus:
        """Run hython subprocess and return an HDAGraphCorpus."""
        if not self.hython_path.exists():
            raise FileNotFoundError(
                f"hython not found at {self.hython_path}. "
                f"Install Houdini or pass --hython-path."
            )

        if not _HYTHON_SCRIPT.exists():
            raise FileNotFoundError(f"Extraction script not found: {_HYTHON_SCRIPT}")

        # Create temp output file
        tmp_dir = Path("/tmp/pixel_vision/extract")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tempfile.NamedTemporaryFile(
            dir=tmp_dir, suffix=".json", delete=False, prefix="labs_hda_"
        )
        tmp_path = Path(tmp_file.name)
        tmp_file.close()

        try:
            # Build command
            cmd = [str(self.hython_path), str(_HYTHON_SCRIPT), str(tmp_path)]
            if self.categories:
                cmd.extend(["--categories", ",".join(self.categories)])
            cmd.extend(["--library-filter", self.library_filter])

            # Run hython
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"hython extraction failed (exit {result.returncode}):\n"
                    f"{result.stderr}"
                )

            # Parse output
            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                raise RuntimeError(
                    f"hython produced no output. stderr:\n{result.stderr}"
                )

            with open(tmp_path) as f:
                raw = json.load(f)

            return _build_corpus(raw)

        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"hython extraction timed out after {self.timeout}s"
            )
        finally:
            # Clean up temp file
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def _build_corpus(raw: dict) -> HDAGraphCorpus:
    """Convert raw hython JSON output into an HDAGraphCorpus."""
    corpus = HDAGraphCorpus(
        houdini_version=raw.get("houdini_version", ""),
        extraction_timestamp=raw.get("extraction_timestamp", ""),
    )

    for graph_data in raw.get("graphs", []):
        graph = HDAGraph(
            type_name=graph_data.get("type_name", ""),
            category=graph_data.get("category", ""),
            label=graph_data.get("label", ""),
            library_path=graph_data.get("library_path", ""),
            min_inputs=graph_data.get("min_inputs", 0),
            max_inputs=graph_data.get("max_inputs", 0),
            max_outputs=graph_data.get("max_outputs", 0),
        )

        # Subnet inputs
        for si_data in graph_data.get("subnet_inputs", []):
            conns = [
                (c["dest_node"], c["dest_input"])
                for c in si_data.get("connections", [])
            ]
            graph.subnet_inputs.append(HDASubnetInput(
                index=si_data.get("index", 0),
                name=si_data.get("name", ""),
                connections=conns,
            ))

        # Internal nodes
        for node_data in graph_data.get("nodes", []):
            pos = node_data.get("position", [0.0, 0.0])
            node = HDAInternalNode(
                name=node_data.get("name", ""),
                type=node_data.get("type", ""),
                position=tuple(pos),
                display_flag=node_data.get("display_flag", False),
                render_flag=node_data.get("render_flag", False),
                bypass_flag=node_data.get("bypass_flag", False),
                is_output=node_data.get("is_output", False),
            )
            for p in node_data.get("parameters", []):
                node.parameters.append(HDANodeParameter(
                    name=p.get("name", ""),
                    value=p.get("value"),
                    expression=p.get("expression"),
                    expression_language=p.get("expression_language", ""),
                ))
            graph.nodes.append(node)

        # Connections
        for conn_data in graph_data.get("connections", []):
            graph.connections.append(HDAConnection(
                source_node=conn_data.get("source_node", ""),
                source_output=conn_data.get("source_output", 0),
                dest_node=conn_data.get("dest_node", ""),
                dest_input=conn_data.get("dest_input", 0),
            ))

        corpus.add(graph)

    return corpus
