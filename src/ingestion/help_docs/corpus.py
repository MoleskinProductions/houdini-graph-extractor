"""Batch parsing of Houdini help docs from the nodes.zip archive."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from .models import HelpCorpus, NodeHelpDoc
from .parser import HelpFileParser

# Default location of Houdini's help archive
DEFAULT_ZIP_PATH = Path("/opt/hfs21.0/houdini/help/nodes.zip")

# Contexts that contain node docs (skip index files, state contexts, etc.)
NODE_CONTEXTS = frozenset({
    "apex", "chop", "cop", "cop2", "dop", "lop", "obj", "out",
    "shop", "sop", "top", "vop",
})

# Match versioned filenames: name-2.0.txt → ("name", "2.0")
_RE_VERSION_SUFFIX = re.compile(r"^(.+?)-(\d+(?:\.\d+)?)\.txt$")


class HelpCorpusParser:
    """Parses all node help docs from a Houdini nodes.zip archive."""

    def __init__(
        self,
        zip_path: Path | str = DEFAULT_ZIP_PATH,
        contexts: frozenset[str] | set[str] | None = None,
    ) -> None:
        self.zip_path = Path(zip_path)
        self.contexts = frozenset(contexts) if contexts else NODE_CONTEXTS
        self._parser = HelpFileParser()

    def parse(self) -> HelpCorpus:
        """Parse the zip archive and return a HelpCorpus.

        Handles:
        - Filtering by context
        - Skipping include-type files (parser returns None)
        - Versioned duplicates: keeps highest version per node
        """
        corpus = HelpCorpus()
        # Track versions: key → (version_tuple, doc)
        versions: dict[str, tuple[tuple[int, ...], NodeHelpDoc]] = {}

        with zipfile.ZipFile(self.zip_path, "r") as zf:
            for entry in sorted(zf.namelist()):
                if not entry.endswith(".txt"):
                    continue

                # Skip deprecated files (bare dash suffix like scatter-.txt)
                if entry.endswith("-.txt"):
                    continue

                # Determine context from path
                parts = entry.split("/")
                if len(parts) < 2:
                    continue  # Skip root-level files like index.txt
                context = parts[0]
                if context not in self.contexts:
                    continue

                try:
                    raw = zf.read(entry)
                    text = raw.decode("utf-8", errors="replace")
                except Exception:
                    continue

                doc = self._parser.parse(text, filename=entry)
                if doc is None:
                    continue

                # Handle versioned duplicates
                base_name = parts[-1].removesuffix(".txt")
                vm = _RE_VERSION_SUFFIX.match(parts[-1])
                if vm:
                    version_tuple = tuple(int(x) for x in vm.group(2).split("."))
                else:
                    version_tuple = (0,)

                key = doc.key
                if key in versions:
                    existing_version = versions[key][0]
                    if version_tuple > existing_version:
                        versions[key] = (version_tuple, doc)
                else:
                    versions[key] = (version_tuple, doc)

        # Build final corpus from de-duped docs, filtering by declared context
        for key, (_, doc) in versions.items():
            if doc.context in self.contexts:
                corpus.add(doc)

        return corpus
