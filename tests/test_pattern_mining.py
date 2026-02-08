"""Tests for Phase 2A: Connection Pattern Mining."""

import json
from pathlib import Path

import pytest

from src.analysis.pattern_mining.models import (
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
from src.analysis.pattern_mining.analyzer import PatternMiner, SUBNET_INPUT_TYPE
from src.analysis.pattern_mining.schema_enricher import SchemaEnricher
from src.ingestion.labs_hda.models import HDAGraphCorpus
from src.ingestion.node_schema.models import SchemaCorpus

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "labs_hda" / "sample_graphs.json"
SCHEMA_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "node_schema" / "sample_schema.json"


# ---------------------------------------------------------------------------
# Unit tests: ConnectionPattern
# ---------------------------------------------------------------------------

class TestConnectionPattern:
    def test_edge_key(self):
        p = ConnectionPattern(
            source_type="scatter", dest_type="mountain::2.0",
            source_output=0, dest_input=0,
        )
        assert p.edge_key == "scatter:0->mountain::2.0:0"

    def test_roundtrip(self):
        p = ConnectionPattern(
            source_type="groupcreate", dest_type="mountain::2.0",
            source_output=0, dest_input=0,
            count=5, hda_keys=["sop/labs::a", "sop/labs::b"],
            source_output_name="Output", dest_input_name="Geometry",
        )
        d = p.to_dict()
        p2 = ConnectionPattern.from_dict(d)
        assert p2.edge_key == p.edge_key
        assert p2.count == 5
        assert p2.source_output_name == "Output"
        assert p2.dest_input_name == "Geometry"
        assert len(p2.hda_keys) == 2

    def test_optional_names_omitted(self):
        p = ConnectionPattern(
            source_type="a", dest_type="b",
            source_output=0, dest_input=0, count=1,
        )
        d = p.to_dict()
        assert "source_output_name" not in d
        assert "dest_input_name" not in d


# ---------------------------------------------------------------------------
# Unit tests: NodeCooccurrence
# ---------------------------------------------------------------------------

class TestNodeCooccurrence:
    def test_pair_key(self):
        c = NodeCooccurrence(type_a="box", type_b="scatter")
        assert c.pair_key == "box+scatter"

    def test_roundtrip(self):
        c = NodeCooccurrence(
            type_a="box", type_b="scatter",
            count=3, type_a_total=5, type_b_total=4,
            jaccard=0.5,
        )
        d = c.to_dict()
        c2 = NodeCooccurrence.from_dict(d)
        assert c2.pair_key == "box+scatter"
        assert c2.count == 3
        assert c2.jaccard == 0.5


# ---------------------------------------------------------------------------
# Unit tests: PortUsageStat / NodePortUsage
# ---------------------------------------------------------------------------

class TestPortUsage:
    def test_port_usage_stat_roundtrip(self):
        s = PortUsageStat(port_index=0, port_name="Geometry", usage_count=10,
                          total_appearances=20, usage_ratio=0.5)
        d = s.to_dict()
        s2 = PortUsageStat.from_dict(d)
        assert s2.port_index == 0
        assert s2.port_name == "Geometry"
        assert s2.usage_ratio == 0.5

    def test_port_name_omitted_when_empty(self):
        s = PortUsageStat(port_index=1, usage_count=1, total_appearances=1, usage_ratio=1.0)
        d = s.to_dict()
        assert "port_name" not in d

    def test_node_port_usage_roundtrip(self):
        npu = NodePortUsage(
            node_type="scatter", context="sop", total_appearances=5,
            inputs=[PortUsageStat(port_index=0, usage_count=5, total_appearances=5, usage_ratio=1.0)],
            outputs=[PortUsageStat(port_index=0, usage_count=3, total_appearances=5, usage_ratio=0.6)],
        )
        d = npu.to_dict()
        npu2 = NodePortUsage.from_dict(d)
        assert npu2.node_type == "scatter"
        assert npu2.total_appearances == 5
        assert len(npu2.inputs) == 1
        assert len(npu2.outputs) == 1


# ---------------------------------------------------------------------------
# Unit tests: ChainPattern
# ---------------------------------------------------------------------------

class TestChainPattern:
    def test_chain_key_2(self):
        cp = ChainPattern(types=["scatter", "mountain::2.0"], count=1)
        assert cp.chain_key == "scatter -> mountain::2.0"
        assert cp.length == 2

    def test_chain_key_3(self):
        cp = ChainPattern(types=["box", "scatter", "mountain::2.0"], count=1)
        assert cp.chain_key == "box -> scatter -> mountain::2.0"
        assert cp.length == 3

    def test_roundtrip(self):
        cp = ChainPattern(
            types=["a", "b", "c"], count=3,
            hda_keys=["sop/labs::x"],
        )
        d = cp.to_dict()
        cp2 = ChainPattern.from_dict(d)
        assert cp2.types == ["a", "b", "c"]
        assert cp2.count == 3
        assert cp2.hda_keys == ["sop/labs::x"]


# ---------------------------------------------------------------------------
# Unit tests: Suggestions
# ---------------------------------------------------------------------------

class TestSuggestions:
    def test_downstream_roundtrip(self):
        s = DownstreamSuggestion(target_type="mountain::2.0", count=5,
                                  source_output=0, dest_input=0)
        d = s.to_dict()
        s2 = DownstreamSuggestion.from_dict(d)
        assert s2.target_type == "mountain::2.0"
        assert s2.count == 5

    def test_upstream_roundtrip(self):
        s = UpstreamSuggestion(source_type="groupcreate", count=3,
                                source_output=0, dest_input=0)
        d = s.to_dict()
        s2 = UpstreamSuggestion.from_dict(d)
        assert s2.source_type == "groupcreate"
        assert s2.count == 3

    def test_node_suggestions_roundtrip(self):
        ns = NodeSuggestions(
            node_type="scatter", context="sop",
            downstream=[DownstreamSuggestion(target_type="mountain::2.0", count=5)],
            upstream=[UpstreamSuggestion(source_type="box", count=3)],
        )
        d = ns.to_dict()
        ns2 = NodeSuggestions.from_dict(d)
        assert ns2.node_type == "scatter"
        assert len(ns2.downstream) == 1
        assert len(ns2.upstream) == 1


# ---------------------------------------------------------------------------
# Unit tests: PatternCorpus
# ---------------------------------------------------------------------------

class TestPatternCorpus:
    def test_empty(self):
        pc = PatternCorpus()
        assert pc.pattern_count == 0
        assert pc.suggestion_count == 0
        assert pc.get_downstream("nonexistent") == []
        assert pc.get_upstream("nonexistent") == []

    def test_get_downstream_limit(self):
        pc = PatternCorpus()
        ns = NodeSuggestions(node_type="test")
        for i in range(15):
            ns.downstream.append(DownstreamSuggestion(target_type=f"t{i}", count=15 - i))
        pc.node_suggestions["test"] = ns
        assert len(pc.get_downstream("test")) == 10
        assert len(pc.get_downstream("test", limit=5)) == 5

    def test_save_load_roundtrip(self, tmp_path):
        pc = PatternCorpus()
        pc.connection_patterns["a:0->b:0"] = ConnectionPattern(
            source_type="a", dest_type="b",
            source_output=0, dest_input=0, count=1,
        )
        pc.node_suggestions["a"] = NodeSuggestions(node_type="a")
        pc.cooccurrences["a+b"] = NodeCooccurrence(type_a="a", type_b="b", count=1)
        pc.chain_patterns_2["a -> b"] = ChainPattern(types=["a", "b"], count=1)

        path = tmp_path / "patterns.json"
        pc.save_json(path)
        loaded = PatternCorpus.load_json(path)
        assert loaded.pattern_count == 1
        assert loaded.suggestion_count == 1
        assert "a+b" in loaded.cooccurrences
        assert "a -> b" in loaded.chain_patterns_2

    def test_to_dict_structure(self):
        pc = PatternCorpus()
        d = pc.to_dict()
        assert d["version"] == "1.0"
        assert d["pattern_count"] == 0
        assert d["suggestion_count"] == 0
        assert "connection_patterns" in d
        assert "node_suggestions" in d
        assert "cooccurrences" in d
        assert "port_usage" in d
        assert "chain_patterns_2" in d
        assert "chain_patterns_3" in d


# ---------------------------------------------------------------------------
# Fixture-based tests: PatternMiner with default exclude_types={"output"}
# ---------------------------------------------------------------------------

class TestPatternMinerFixture:
    """Tests mining against the sample_graphs.json fixture.

    With exclude_types={"output"} (default):
    - quickmaterial: nodes = material, principledshader::2.0 | connections: none (material->output excluded)
    - edge_damage: nodes = groupcreate, measure, mountain::2.0 | connections: groupcreate->mountain::2.0
    - wedge_postprocess: nodes = pythonscript | connections: none

    Expected: 1 connection pattern, 1 two-chain, 0 three-chains.
    """

    @pytest.fixture(scope="class")
    def corpus(self):
        return HDAGraphCorpus.load_json(FIXTURE_PATH)

    @pytest.fixture(scope="class")
    def result(self, corpus):
        miner = PatternMiner()  # default: exclude_types={"output"}
        return miner.mine(corpus)

    def test_connection_patterns(self, result):
        # Only groupcreate:0->mountain::2.0:0 survives (material->output excluded)
        assert result.pattern_count >= 1
        assert "groupcreate:0->mountain::2.0:0" in result.connection_patterns
        p = result.connection_patterns["groupcreate:0->mountain::2.0:0"]
        assert p.count == 1
        assert "sop/labs::edge_damage::2.0" in p.hda_keys

    def test_suggestions(self, result):
        # groupcreate should have mountain::2.0 as downstream
        ds = result.get_downstream("groupcreate")
        assert len(ds) >= 1
        assert ds[0].target_type == "mountain::2.0"

        # mountain::2.0 should have groupcreate as upstream
        us = result.get_upstream("mountain::2.0")
        assert len(us) >= 1
        assert us[0].source_type == "groupcreate"

    def test_cooccurrences(self, result):
        # edge_damage has groupcreate, measure, mountain::2.0
        assert "groupcreate+measure" in result.cooccurrences
        assert "groupcreate+mountain::2.0" in result.cooccurrences
        assert "measure+mountain::2.0" in result.cooccurrences

        # quickmaterial has material, principledshader::2.0
        assert "material+principledshader::2.0" in result.cooccurrences

    def test_jaccard(self, result):
        # groupcreate and mountain::2.0 each appear in 1 HDA, co-occur in 1
        # jaccard = 1 / (1 + 1 - 1) = 1.0
        cooc = result.cooccurrences["groupcreate+mountain::2.0"]
        assert cooc.jaccard == 1.0

    def test_port_usage(self, result):
        # groupcreate should have output 0 used
        assert "groupcreate" in result.port_usage
        npu = result.port_usage["groupcreate"]
        assert npu.total_appearances == 1
        assert len(npu.outputs) >= 1
        assert npu.outputs[0].port_index == 0
        assert npu.outputs[0].usage_ratio == 1.0

    def test_chain_patterns_2(self, result):
        # groupcreate -> mountain::2.0
        assert "groupcreate -> mountain::2.0" in result.chain_patterns_2

    def test_chain_patterns_3(self, result):
        # With output excluded, only subnet_input->groupcreate->mountain::2.0 exists
        assert "__subnet_input__ -> groupcreate -> mountain::2.0" in result.chain_patterns_3

    def test_output_type_excluded(self, result):
        # No patterns should reference the "output" type
        for p in result.connection_patterns.values():
            assert p.source_type != "output"
            assert p.dest_type != "output"

    def test_save_load_roundtrip(self, result, tmp_path):
        path = tmp_path / "test_patterns.json"
        result.save_json(path)
        loaded = PatternCorpus.load_json(path)
        assert loaded.pattern_count == result.pattern_count
        assert loaded.suggestion_count == result.suggestion_count


# ---------------------------------------------------------------------------
# Fixture-based tests: PatternMiner with exclude_types=set() (nothing excluded)
# ---------------------------------------------------------------------------

class TestPatternMinerNoExclude:
    """Tests mining with no type exclusions.

    With exclude_types=set():
    - quickmaterial: __subnet_input__:0->material:0, material:0->output:0
    - edge_damage: __subnet_input__:0->groupcreate:0, __subnet_input__:0->measure:0,
                   groupcreate:0->mountain::2.0:0, mountain::2.0:0->output:0
    - wedge_postprocess: no connections

    Expected: 6 connection patterns (3 regular + 3 subnet), 6 two-chains, 3 three-chains.
    """

    @pytest.fixture(scope="class")
    def corpus(self):
        return HDAGraphCorpus.load_json(FIXTURE_PATH)

    @pytest.fixture(scope="class")
    def result(self, corpus):
        miner = PatternMiner(exclude_types=set())
        return miner.mine(corpus)

    def test_connection_patterns(self, result):
        assert result.pattern_count == 6
        assert "material:0->output:0" in result.connection_patterns
        assert "groupcreate:0->mountain::2.0:0" in result.connection_patterns
        assert "mountain::2.0:0->output:0" in result.connection_patterns
        assert "__subnet_input__:0->material:0" in result.connection_patterns
        assert "__subnet_input__:0->groupcreate:0" in result.connection_patterns
        assert "__subnet_input__:0->measure:0" in result.connection_patterns

    def test_chain_patterns_2(self, result):
        assert len(result.chain_patterns_2) == 6
        assert "material -> output" in result.chain_patterns_2
        assert "groupcreate -> mountain::2.0" in result.chain_patterns_2
        assert "mountain::2.0 -> output" in result.chain_patterns_2
        assert "__subnet_input__ -> material" in result.chain_patterns_2
        assert "__subnet_input__ -> groupcreate" in result.chain_patterns_2

    def test_chain_patterns_3(self, result):
        assert len(result.chain_patterns_3) == 3
        assert "groupcreate -> mountain::2.0 -> output" in result.chain_patterns_3
        assert "__subnet_input__ -> groupcreate -> mountain::2.0" in result.chain_patterns_3
        assert "__subnet_input__ -> material -> output" in result.chain_patterns_3
        cp = result.chain_patterns_3["groupcreate -> mountain::2.0 -> output"]
        assert cp.count == 1
        assert cp.length == 3


# ---------------------------------------------------------------------------
# Fixture-based tests: subnet inputs
# ---------------------------------------------------------------------------

class TestSubnetInputMining:
    """Test that subnet inputs are synthesized as __subnet_input__ type."""

    @pytest.fixture(scope="class")
    def corpus(self):
        return HDAGraphCorpus.load_json(FIXTURE_PATH)

    @pytest.fixture(scope="class")
    def result(self, corpus):
        # Include subnet input type but exclude output
        miner = PatternMiner(exclude_types={"output"})
        return miner.mine(corpus)

    def test_subnet_input_connections(self, result):
        # quickmaterial: __subnet_input__:0->material:0
        # edge_damage: __subnet_input__:0->groupcreate:0, __subnet_input__:0->measure:0
        subnet_patterns = [
            p for p in result.connection_patterns.values()
            if p.source_type == SUBNET_INPUT_TYPE
        ]
        assert len(subnet_patterns) >= 2

    def test_subnet_input_suggestions(self, result):
        # material should have __subnet_input__ as upstream
        us = result.get_upstream("material")
        subnet_upstream = [u for u in us if u.source_type == SUBNET_INPUT_TYPE]
        assert len(subnet_upstream) == 1


# ---------------------------------------------------------------------------
# PatternMiner: min_pattern_count filter
# ---------------------------------------------------------------------------

class TestMinPatternCount:
    @pytest.fixture(scope="class")
    def corpus(self):
        return HDAGraphCorpus.load_json(FIXTURE_PATH)

    def test_min_count_filters(self, corpus):
        # With min_count=2, all patterns from fixture should be filtered out
        # (each appears in only 1 HDA)
        miner = PatternMiner(min_pattern_count=2)
        result = miner.mine(corpus)
        assert result.pattern_count == 0
        assert len(result.chain_patterns_2) == 0
        assert len(result.cooccurrences) == 0


# ---------------------------------------------------------------------------
# SchemaEnricher tests
# ---------------------------------------------------------------------------

class TestSchemaEnricher:
    @pytest.fixture(scope="class")
    def schema_corpus(self):
        return SchemaCorpus.load_json(SCHEMA_FIXTURE_PATH)

    def test_enrich_adds_port_names(self, schema_corpus):
        pc = PatternCorpus()
        pc.connection_patterns["scatter:0->scatter:0"] = ConnectionPattern(
            source_type="scatter", dest_type="scatter",
            source_output=0, dest_input=0, count=1,
        )
        pc.port_usage["scatter"] = NodePortUsage(
            node_type="scatter", total_appearances=1,
            inputs=[PortUsageStat(port_index=0, usage_count=1, total_appearances=1, usage_ratio=1.0)],
            outputs=[PortUsageStat(port_index=0, usage_count=1, total_appearances=1, usage_ratio=1.0)],
        )

        enricher = SchemaEnricher(schema_corpus)
        enricher.enrich(pc)

        p = pc.connection_patterns["scatter:0->scatter:0"]
        assert p.source_output_name == "Scattered Points"
        assert p.dest_input_name == "Surface to Scatter On"

        npu = pc.port_usage["scatter"]
        assert npu.inputs[0].port_name == "Surface to Scatter On"
        assert npu.outputs[0].port_name == "Scattered Points"

    def test_enrich_unknown_type(self, schema_corpus):
        pc = PatternCorpus()
        pc.connection_patterns["unknown:0->unknown:0"] = ConnectionPattern(
            source_type="unknown_type", dest_type="unknown_type",
            source_output=0, dest_input=0, count=1,
        )

        enricher = SchemaEnricher(schema_corpus)
        enricher.enrich(pc)

        p = pc.connection_patterns["unknown:0->unknown:0"]
        assert p.source_output_name == ""
        assert p.dest_input_name == ""

    def test_enrich_preserves_existing_names(self, schema_corpus):
        pc = PatternCorpus()
        pc.connection_patterns["scatter:0->scatter:0"] = ConnectionPattern(
            source_type="scatter", dest_type="scatter",
            source_output=0, dest_input=0, count=1,
            source_output_name="Already Set",
        )

        enricher = SchemaEnricher(schema_corpus)
        enricher.enrich(pc)

        p = pc.connection_patterns["scatter:0->scatter:0"]
        assert p.source_output_name == "Already Set"
        assert p.dest_input_name == "Surface to Scatter On"
