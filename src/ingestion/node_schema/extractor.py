"""Wrapper that launches hython to extract node type schemas."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .models import CATEGORY_TO_CONTEXT, NodeTypeSchema, ParmSchema, PortSchema, SchemaCorpus

# Location of the hython extraction script, relative to this file.
_HYTHON_SCRIPT = Path(__file__).parent / "_hython_extract.py"

DEFAULT_HYTHON_PATH = Path("/opt/hfs21.0/bin/hython")


class NodeSchemaExtractor:
    """Extracts node type schemas by running a hython subprocess."""

    def __init__(
        self,
        hython_path: str | Path = DEFAULT_HYTHON_PATH,
        categories: list[str] | None = None,
        timeout: int = 120,
        extract_ports: bool = True,
    ) -> None:
        self.hython_path = Path(hython_path)
        self.categories = categories
        self.timeout = timeout
        self.extract_ports = extract_ports

    def extract(self) -> SchemaCorpus:
        """Run hython subprocess and return a SchemaCorpus."""
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
            dir=tmp_dir, suffix=".json", delete=False, prefix="schema_"
        )
        tmp_path = Path(tmp_file.name)
        tmp_file.close()

        try:
            # Build command
            cmd = [str(self.hython_path), str(_HYTHON_SCRIPT), str(tmp_path)]
            if self.categories:
                cmd.extend(["--categories", ",".join(self.categories)])
            if not self.extract_ports:
                cmd.append("--no-ports")

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


def _build_corpus(raw: dict) -> SchemaCorpus:
    """Convert raw hython JSON output into a SchemaCorpus."""
    corpus = SchemaCorpus(
        houdini_version=raw.get("houdini_version", ""),
        extraction_timestamp=raw.get("extraction_timestamp", ""),
    )

    for node_data in raw.get("nodes", []):
        category = node_data.get("category", "")
        type_name = node_data.get("type_name", "")

        schema = NodeTypeSchema(
            category=category,
            type_name=type_name,
            label=node_data.get("label", ""),
            scope_namespace=node_data.get("scope_namespace", ""),
            namespace=node_data.get("namespace", ""),
            base_type=node_data.get("base_type", ""),
            version=node_data.get("version", ""),
            min_inputs=node_data.get("min_inputs", 0),
            max_inputs=node_data.get("max_inputs", 0),
            max_outputs=node_data.get("max_outputs", 0),
            icon=node_data.get("icon", ""),
            is_generator=node_data.get("is_generator", False),
            unordered_inputs=node_data.get("unordered_inputs", False),
            deprecated=node_data.get("deprecated", False),
            is_hda=node_data.get("is_hda", False),
        )

        # Parameters
        for p in node_data.get("parameters", []):
            schema.parameters.append(ParmSchema.from_dict(p))

        # Ports
        for p in node_data.get("inputs", []):
            schema.inputs.append(PortSchema.from_dict(p))
        for p in node_data.get("outputs", []):
            schema.outputs.append(PortSchema.from_dict(p))

        corpus.add(schema)

    return corpus
