"""Phase 2B: Intent mapper — builds IntentLibrary from HDAGraphCorpus."""

from __future__ import annotations

from collections import defaultdict

from ...ingestion.labs_hda.models import HDAGraph, HDAGraphCorpus
from .models import IntentCluster, IntentLibrary, SubgraphTemplate

# Known vendor prefixes stripped from HDA labels before tokenizing.
VENDOR_PREFIXES = ("Gaea ", "Kelvin ", "GameDev ")

# Stopwords removed from keyword tokens.
STOPWORDS = frozenset({"the", "a", "an", "from", "to", "and", "for", "with", "of", "in", "on"})


class IntentMapper:
    """Map Labs HDAs to high-level user intents.

    Three-step pipeline:
      1. Label normalization — strip prefixes, tokenize, remove stopwords
      2. Prefix-based clustering — group HDAs by shared keyword prefix
      3. Template extraction — build SubgraphTemplate for each HDA
    """

    def __init__(
        self,
        exclude_types: set[str] | None = None,
        min_node_count: int = 1,
    ) -> None:
        self.exclude_types = exclude_types if exclude_types is not None else {"output"}
        self.min_node_count = min_node_count

    def map(self, corpus: HDAGraphCorpus) -> IntentLibrary:
        """Build an IntentLibrary from an HDAGraphCorpus."""
        # Step 1+2: normalize labels and cluster by prefix
        clusters: dict[str, list[tuple[list[str], HDAGraph]]] = defaultdict(list)

        for graph in corpus.graphs.values():
            tokens = self._normalize_label(graph.label)
            if not tokens:
                continue

            # Filter by min_node_count (counting non-excluded nodes)
            effective_count = sum(
                1 for n in graph.nodes if n.type not in self.exclude_types
            )
            if effective_count < self.min_node_count:
                continue

            # Use full token list as the cluster key
            intent_id = "_".join(tokens)
            clusters[intent_id].append((tokens, graph))

        # Step 3: build IntentLibrary
        library = IntentLibrary()
        for intent_id, members in sorted(clusters.items()):
            tokens = members[0][0]
            description = " ".join(w.capitalize() for w in tokens)
            templates = []
            for _, graph in members:
                templates.append(self._build_template(graph))
            # Sort templates by node_count descending (richest first)
            templates.sort(key=lambda t: t.node_count, reverse=True)

            # Dominant category = most common context among templates
            context_counts: dict[str, int] = defaultdict(int)
            for t in templates:
                context_counts[t.context] += 1
            dominant_context = max(context_counts, key=context_counts.get)  # type: ignore[arg-type]

            cluster = IntentCluster(
                intent_id=intent_id,
                keywords=list(tokens),
                description=description,
                category=dominant_context,
                templates=templates,
            )
            library.intents[intent_id] = cluster

        return library

    def _normalize_label(self, label: str) -> list[str]:
        """Strip prefixes, tokenize, and remove stopwords from an HDA label."""
        text = label

        # Strip "Labs" prefix (with or without trailing space)
        if text == "Labs":
            return []
        if text.startswith("Labs "):
            text = text[5:]

        # Strip vendor prefixes
        for prefix in VENDOR_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):]

        # Strip version suffixes like " 2.0", " 1.5" at end
        parts = text.rsplit(" ", 1)
        if len(parts) == 2:
            try:
                float(parts[1])
                text = parts[0]
            except ValueError:
                pass

        # Tokenize: lowercase, split on spaces
        tokens = text.lower().split()

        # Remove stopwords
        tokens = [t for t in tokens if t not in STOPWORDS]

        return tokens

    def _build_template(self, graph: HDAGraph) -> SubgraphTemplate:
        """Build a SubgraphTemplate from an HDAGraph."""
        node_types = sorted({
            n.type for n in graph.nodes if n.type not in self.exclude_types
        })
        node_count = sum(
            1 for n in graph.nodes if n.type not in self.exclude_types
        )
        return SubgraphTemplate(
            hda_key=graph.key,
            label=graph.label,
            category=graph.category,
            context=graph.context,
            node_types=node_types,
            node_count=node_count,
            connection_count=graph.connection_count,
        )
