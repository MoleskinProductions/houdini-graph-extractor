"""Pattern mining analyzer for Labs HDA graph corpus.

Mines connection patterns, node co-occurrences, port usage statistics,
and chain patterns from the structural data extracted in Phase 1C.
"""

from __future__ import annotations

from collections import defaultdict

from src.ingestion.labs_hda.models import HDAGraphCorpus, HDAGraph

from .models import (
    ChainPattern,
    ConnectionPattern,
    DownstreamSuggestion,
    NodeCooccurrence,
    NodePortUsage,
    NodeSuggestions,
    PatternCorpus,
    PortUsageStat,
    UpstreamSuggestion,
)

# Sentinel type for subnet input pseudo-connections
SUBNET_INPUT_TYPE = "__subnet_input__"


class PatternMiner:
    """Mines connection patterns from an HDAGraphCorpus.

    Five mining passes:
    1. Connection frequency — directed edge types across HDAs
    2. Downstream/upstream suggestions — ranked neighbor lists
    3. Node co-occurrence — Jaccard similarity between type pairs
    4. Port usage statistics — which ports are actually connected
    5. Chain patterns — sequential 2-chains and 3-chains
    """

    def __init__(
        self,
        min_pattern_count: int = 1,
        max_chain_length: int = 3,
        exclude_types: set[str] | None = None,
    ) -> None:
        self.min_pattern_count = min_pattern_count
        self.max_chain_length = max_chain_length
        self.exclude_types = exclude_types if exclude_types is not None else {"output"}

    def mine(self, corpus: HDAGraphCorpus) -> PatternCorpus:
        """Run all mining passes on the corpus and return a PatternCorpus."""
        result = PatternCorpus()

        # Build a name->type mapping per graph and collect edges
        edges_by_graph = self._collect_edges(corpus)

        # Pass 1: Connection frequency
        self._mine_connection_frequency(edges_by_graph, result)

        # Pass 2: Downstream/upstream suggestions (derived from pass 1)
        self._mine_suggestions(result)

        # Pass 3: Node co-occurrence
        self._mine_cooccurrence(corpus, result)

        # Pass 4: Port usage statistics
        self._mine_port_usage(edges_by_graph, corpus, result)

        # Pass 5: Chain patterns
        self._mine_chains(edges_by_graph, result)

        return result

    def _should_exclude(self, node_type: str) -> bool:
        """Check if a node type should be excluded from mining."""
        return node_type in self.exclude_types

    def _collect_edges(
        self, corpus: HDAGraphCorpus
    ) -> dict[str, list[tuple[str, int, str, int]]]:
        """Collect typed edges per graph.

        Returns: {hda_key: [(source_type, source_output, dest_type, dest_input), ...]}
        """
        edges_by_graph: dict[str, list[tuple[str, int, str, int]]] = {}

        for hda_key, graph in corpus.graphs.items():
            # Build name->type mapping for this graph
            name_to_type: dict[str, str] = {}
            for node in graph.nodes:
                name_to_type[node.name] = node.type

            edges: list[tuple[str, int, str, int]] = []

            # Regular connections
            for conn in graph.connections:
                src_type = name_to_type.get(conn.source_node)
                dst_type = name_to_type.get(conn.dest_node)
                if src_type is None or dst_type is None:
                    continue
                if self._should_exclude(src_type) or self._should_exclude(dst_type):
                    continue
                edges.append((src_type, conn.source_output, dst_type, conn.dest_input))

            # Subnet input pseudo-connections
            for si in graph.subnet_inputs:
                for dest_node, dest_input in si.connections:
                    dst_type = name_to_type.get(dest_node)
                    if dst_type is None:
                        continue
                    if self._should_exclude(SUBNET_INPUT_TYPE) or self._should_exclude(dst_type):
                        continue
                    edges.append((SUBNET_INPUT_TYPE, si.index, dst_type, dest_input))

            edges_by_graph[hda_key] = edges

        return edges_by_graph

    def _mine_connection_frequency(
        self,
        edges_by_graph: dict[str, list[tuple[str, int, str, int]]],
        result: PatternCorpus,
    ) -> None:
        """Pass 1: Count unique edge types across HDAs, deduplicated per graph."""
        # edge_key -> {hda_keys}
        pattern_hdas: dict[tuple[str, int, str, int], set[str]] = defaultdict(set)

        for hda_key, edges in edges_by_graph.items():
            # Deduplicate edges within this graph
            seen: set[tuple[str, int, str, int]] = set()
            for edge in edges:
                if edge not in seen:
                    seen.add(edge)
                    pattern_hdas[edge].add(hda_key)

        for edge, hda_keys in pattern_hdas.items():
            count = len(hda_keys)
            if count < self.min_pattern_count:
                continue
            src_type, src_out, dst_type, dst_in = edge
            pattern = ConnectionPattern(
                source_type=src_type,
                dest_type=dst_type,
                source_output=src_out,
                dest_input=dst_in,
                count=count,
                hda_keys=sorted(hda_keys),
            )
            result.connection_patterns[pattern.edge_key] = pattern

    def _mine_suggestions(self, result: PatternCorpus) -> None:
        """Pass 2: Build downstream/upstream suggestion lists from connection patterns."""
        # Group by source_type for downstream
        downstream_map: dict[str, list[ConnectionPattern]] = defaultdict(list)
        # Group by dest_type for upstream
        upstream_map: dict[str, list[ConnectionPattern]] = defaultdict(list)

        for pattern in result.connection_patterns.values():
            downstream_map[pattern.source_type].append(pattern)
            upstream_map[pattern.dest_type].append(pattern)

        # All node types that appear in any pattern
        all_types = set(downstream_map.keys()) | set(upstream_map.keys())

        for node_type in all_types:
            ns = NodeSuggestions(node_type=node_type)

            # Downstream: sorted by count descending
            for p in sorted(downstream_map.get(node_type, []),
                            key=lambda p: p.count, reverse=True):
                ns.downstream.append(DownstreamSuggestion(
                    target_type=p.dest_type,
                    count=p.count,
                    source_output=p.source_output,
                    dest_input=p.dest_input,
                ))

            # Upstream: sorted by count descending
            for p in sorted(upstream_map.get(node_type, []),
                            key=lambda p: p.count, reverse=True):
                ns.upstream.append(UpstreamSuggestion(
                    source_type=p.source_type,
                    count=p.count,
                    source_output=p.source_output,
                    dest_input=p.dest_input,
                ))

            result.node_suggestions[node_type] = ns

    def _mine_cooccurrence(
        self, corpus: HDAGraphCorpus, result: PatternCorpus
    ) -> None:
        """Pass 3: Compute co-occurrence counts and Jaccard similarity."""
        # Track which HDAs contain each node type
        type_to_hdas: dict[str, set[str]] = defaultdict(set)
        # Track node type sets per HDA
        hda_type_sets: dict[str, set[str]] = {}

        for hda_key, graph in corpus.graphs.items():
            types_in_hda: set[str] = set()
            for node in graph.nodes:
                if not self._should_exclude(node.type):
                    types_in_hda.add(node.type)
            hda_type_sets[hda_key] = types_in_hda
            for t in types_in_hda:
                type_to_hdas[t].add(hda_key)

        # Count co-occurrences for each pair
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for types_in_hda in hda_type_sets.values():
            type_list = sorted(types_in_hda)
            for i in range(len(type_list)):
                for j in range(i + 1, len(type_list)):
                    pair_counts[(type_list[i], type_list[j])] += 1

        for (type_a, type_b), count in pair_counts.items():
            if count < self.min_pattern_count:
                continue
            a_total = len(type_to_hdas[type_a])
            b_total = len(type_to_hdas[type_b])
            union = a_total + b_total - count
            jaccard = count / union if union > 0 else 0.0

            cooc = NodeCooccurrence(
                type_a=type_a,
                type_b=type_b,
                count=count,
                type_a_total=a_total,
                type_b_total=b_total,
                jaccard=jaccard,
            )
            result.cooccurrences[cooc.pair_key] = cooc

    def _mine_port_usage(
        self,
        edges_by_graph: dict[str, list[tuple[str, int, str, int]]],
        corpus: HDAGraphCorpus,
        result: PatternCorpus,
    ) -> None:
        """Pass 4: Track which ports are actually connected per node type."""
        # Count appearances of each node type across all HDAs
        type_appearances: dict[str, set[str]] = defaultdict(set)
        for hda_key, graph in corpus.graphs.items():
            for node in graph.nodes:
                if not self._should_exclude(node.type):
                    type_appearances[node.type].add(hda_key)

        # Count port usage from edges
        # output_usage[type][(port_index)] = set of hda_keys
        output_usage: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
        # input_usage[type][(port_index)] = set of hda_keys
        input_usage: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))

        for hda_key, edges in edges_by_graph.items():
            for src_type, src_out, dst_type, dst_in in edges:
                if src_type != SUBNET_INPUT_TYPE:
                    output_usage[src_type][src_out].add(hda_key)
                input_usage[dst_type][dst_in].add(hda_key)

        # Build NodePortUsage for each type
        for node_type, hda_keys in type_appearances.items():
            total = len(hda_keys)
            npu = NodePortUsage(node_type=node_type, total_appearances=total)

            for port_idx in sorted(input_usage.get(node_type, {}).keys()):
                usage = len(input_usage[node_type][port_idx])
                npu.inputs.append(PortUsageStat(
                    port_index=port_idx,
                    usage_count=usage,
                    total_appearances=total,
                    usage_ratio=usage / total if total > 0 else 0.0,
                ))

            for port_idx in sorted(output_usage.get(node_type, {}).keys()):
                usage = len(output_usage[node_type][port_idx])
                npu.outputs.append(PortUsageStat(
                    port_index=port_idx,
                    usage_count=usage,
                    total_appearances=total,
                    usage_ratio=usage / total if total > 0 else 0.0,
                ))

            result.port_usage[node_type] = npu

    def _mine_chains(
        self,
        edges_by_graph: dict[str, list[tuple[str, int, str, int]]],
        result: PatternCorpus,
    ) -> None:
        """Pass 5: Enumerate 2-chains and 3-chains by walking downstream neighbors."""
        chain2_hdas: dict[tuple[str, ...], set[str]] = defaultdict(set)
        chain3_hdas: dict[tuple[str, ...], set[str]] = defaultdict(set)

        for hda_key, edges in edges_by_graph.items():
            # Build adjacency list: src_type -> set of dest_types
            adjacency: dict[str, set[str]] = defaultdict(set)
            for src_type, _, dst_type, _ in edges:
                adjacency[src_type].add(dst_type)

            # Deduplicate chains within this graph
            seen_2: set[tuple[str, ...]] = set()
            seen_3: set[tuple[str, ...]] = set()

            for src_type in adjacency:
                for mid_type in adjacency[src_type]:
                    chain2 = (src_type, mid_type)
                    if chain2 not in seen_2:
                        seen_2.add(chain2)
                        chain2_hdas[chain2].add(hda_key)

                    if self.max_chain_length >= 3 and mid_type in adjacency:
                        for end_type in adjacency[mid_type]:
                            chain3 = (src_type, mid_type, end_type)
                            if chain3 not in seen_3:
                                seen_3.add(chain3)
                                chain3_hdas[chain3].add(hda_key)

        for chain_types, hda_keys in chain2_hdas.items():
            count = len(hda_keys)
            if count < self.min_pattern_count:
                continue
            cp = ChainPattern(
                types=list(chain_types),
                count=count,
                hda_keys=sorted(hda_keys),
            )
            result.chain_patterns_2[cp.chain_key] = cp

        for chain_types, hda_keys in chain3_hdas.items():
            count = len(hda_keys)
            if count < self.min_pattern_count:
                continue
            cp = ChainPattern(
                types=list(chain_types),
                count=count,
                hda_keys=sorted(hda_keys),
            )
            result.chain_patterns_3[cp.chain_key] = cp
