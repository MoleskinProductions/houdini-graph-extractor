"""Tests for Labs HDA internal graph extraction (Phase 1C)."""

import json
from pathlib import Path

import pytest

from src.ingestion.labs_hda.models import (
    CATEGORY_TO_CONTEXT,
    HDAConnection,
    HDAGraph,
    HDAGraphCorpus,
    HDAInternalNode,
    HDANodeParameter,
    HDASubnetInput,
)
from src.ingestion.labs_hda.extractor import DEFAULT_HYTHON_PATH, LabsHDAExtractor

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "labs_hda" / "sample_graphs.json"
HYTHON_AVAILABLE = DEFAULT_HYTHON_PATH.exists()


# ---------------------------------------------------------------------------
# Unit tests: HDANodeParameter
# ---------------------------------------------------------------------------

class TestHDANodeParameter:
    def test_minimal(self):
        p = HDANodeParameter(name="foo")
        assert p.name == "foo"
        assert p.value is None
        assert p.expression is None
        d = p.to_dict()
        assert d == {"name": "foo"}

    def test_roundtrip(self):
        p = HDANodeParameter(
            name="height",
            value=0.02,
            expression='ch("../damage_amount")',
            expression_language="hscript",
        )
        d = p.to_dict()
        p2 = HDANodeParameter.from_dict(d)
        assert p2.name == p.name
        assert p2.value == p.value
        assert p2.expression == p.expression
        assert p2.expression_language == "hscript"

    def test_value_only(self):
        p = HDANodeParameter(name="rough", value=0.3)
        d = p.to_dict()
        assert d == {"name": "rough", "value": 0.3}
        assert "expression" not in d

    def test_expression_omitted_when_none(self):
        p = HDANodeParameter(name="foo", value=1)
        d = p.to_dict()
        assert "expression" not in d
        assert "expression_language" not in d


# ---------------------------------------------------------------------------
# Unit tests: HDAInternalNode
# ---------------------------------------------------------------------------

class TestHDAInternalNode:
    def test_minimal(self):
        n = HDAInternalNode(name="box1", type="box")
        d = n.to_dict()
        assert d == {"name": "box1", "type": "box"}
        assert "position" not in d  # (0,0) is default, omitted
        assert "display_flag" not in d

    def test_with_flags_and_position(self):
        n = HDAInternalNode(
            name="out",
            type="output",
            position=(3.0, -2.0),
            display_flag=True,
            is_output=True,
        )
        d = n.to_dict()
        assert d["position"] == [3.0, -2.0]
        assert d["display_flag"] is True
        assert d["is_output"] is True
        assert "render_flag" not in d

    def test_roundtrip(self):
        n = HDAInternalNode(
            name="mtl",
            type="material",
            position=(1.5, -3.0),
            display_flag=True,
            render_flag=True,
            bypass_flag=False,
        )
        n.parameters.append(HDANodeParameter(name="path", value="/mat/test"))
        d = n.to_dict()
        n2 = HDAInternalNode.from_dict(d)
        assert n2.name == "mtl"
        assert n2.type == "material"
        assert n2.position == (1.5, -3.0)
        assert n2.display_flag is True
        assert n2.render_flag is True
        assert n2.bypass_flag is False
        assert len(n2.parameters) == 1
        assert n2.parameters[0].value == "/mat/test"

    def test_flags_omitted_when_false(self):
        n = HDAInternalNode(name="n", type="t", display_flag=False, bypass_flag=False)
        d = n.to_dict()
        assert "display_flag" not in d
        assert "bypass_flag" not in d
        assert "is_output" not in d


# ---------------------------------------------------------------------------
# Unit tests: HDAConnection
# ---------------------------------------------------------------------------

class TestHDAConnection:
    def test_roundtrip(self):
        c = HDAConnection(
            source_node="scatter1",
            source_output=0,
            dest_node="mountain1",
            dest_input=0,
        )
        d = c.to_dict()
        c2 = HDAConnection.from_dict(d)
        assert c2.source_node == "scatter1"
        assert c2.source_output == 0
        assert c2.dest_node == "mountain1"
        assert c2.dest_input == 0

    def test_multi_output(self):
        c = HDAConnection(
            source_node="split1",
            source_output=2,
            dest_node="merge1",
            dest_input=1,
        )
        d = c.to_dict()
        assert d["source_output"] == 2
        assert d["dest_input"] == 1


# ---------------------------------------------------------------------------
# Unit tests: HDASubnetInput
# ---------------------------------------------------------------------------

class TestHDASubnetInput:
    def test_roundtrip(self):
        si = HDASubnetInput(
            index=0,
            name="Geometry",
            connections=[("box1", 0), ("scatter1", 0)],
        )
        d = si.to_dict()
        si2 = HDASubnetInput.from_dict(d)
        assert si2.index == 0
        assert si2.name == "Geometry"
        assert len(si2.connections) == 2
        assert si2.connections[0] == ("box1", 0)
        assert si2.connections[1] == ("scatter1", 0)

    def test_empty_connections(self):
        si = HDASubnetInput(index=1, name="Input 2")
        d = si.to_dict()
        assert "connections" not in d
        si2 = HDASubnetInput.from_dict(d)
        assert si2.connections == []


# ---------------------------------------------------------------------------
# Unit tests: HDAGraph
# ---------------------------------------------------------------------------

class TestHDAGraph:
    def test_key_and_context(self):
        g = HDAGraph(type_name="labs::quickmaterial::2.0", category="Sop")
        assert g.key == "sop/labs::quickmaterial::2.0"
        assert g.context == "sop"

    def test_context_mapping(self):
        for category, expected_ctx in CATEGORY_TO_CONTEXT.items():
            g = HDAGraph(type_name="test", category=category)
            assert g.context == expected_ctx

    def test_counts(self):
        g = HDAGraph(type_name="test", category="Sop")
        assert g.node_count == 0
        assert g.connection_count == 0
        g.nodes.append(HDAInternalNode(name="a", type="box"))
        g.nodes.append(HDAInternalNode(name="b", type="scatter"))
        g.connections.append(HDAConnection("a", 0, "b", 0))
        assert g.node_count == 2
        assert g.connection_count == 1

    def test_roundtrip(self):
        g = HDAGraph(
            type_name="labs::edge_damage::2.0",
            category="Sop",
            label="Edge Damage",
            library_path="/path/to/SideFX_Labs.hda",
            min_inputs=1,
            max_inputs=1,
            max_outputs=1,
        )
        g.subnet_inputs.append(HDASubnetInput(0, "Geometry", [("n1", 0)]))
        g.nodes.append(HDAInternalNode(name="n1", type="groupcreate"))
        g.connections.append(HDAConnection("n1", 0, "n2", 0))

        d = g.to_dict()
        g2 = HDAGraph.from_dict(d)
        assert g2.key == "sop/labs::edge_damage::2.0"
        assert g2.label == "Edge Damage"
        assert g2.min_inputs == 1
        assert len(g2.subnet_inputs) == 1
        assert len(g2.nodes) == 1
        assert len(g2.connections) == 1

    def test_boolean_fields_omitted_when_zero(self):
        g = HDAGraph(type_name="test", category="Sop")
        d = g.to_dict()
        assert "min_inputs" not in d
        assert "max_inputs" not in d
        assert "subnet_inputs" not in d
        assert "nodes" not in d
        assert "connections" not in d


# ---------------------------------------------------------------------------
# Unit tests: HDAGraphCorpus
# ---------------------------------------------------------------------------

class TestHDAGraphCorpus:
    def test_empty(self):
        c = HDAGraphCorpus()
        assert c.hda_count == 0
        assert c.contexts() == []

    def test_add_and_get(self):
        c = HDAGraphCorpus()
        g = HDAGraph(type_name="labs::test", category="Sop")
        c.add(g)
        assert c.hda_count == 1
        assert c.get_graph("sop/labs::test") is g
        assert c.get_graph("sop/missing") is None

    def test_contexts(self):
        c = HDAGraphCorpus()
        c.add(HDAGraph(type_name="labs::a", category="Sop"))
        c.add(HDAGraph(type_name="labs::b", category="Top"))
        c.add(HDAGraph(type_name="labs::c", category="Sop"))
        assert c.contexts() == ["sop", "top"]

    def test_get_by_context(self):
        c = HDAGraphCorpus()
        c.add(HDAGraph(type_name="labs::a", category="Sop"))
        c.add(HDAGraph(type_name="labs::b", category="Top"))
        c.add(HDAGraph(type_name="labs::c", category="Sop"))
        sops = c.get_by_context("sop")
        assert len(sops) == 2
        assert all(g.context == "sop" for g in sops)

    def test_save_load_roundtrip(self, tmp_path):
        c = HDAGraphCorpus(
            houdini_version="21.0.506",
            extraction_timestamp="2025-01-20T12:00:00+00:00",
        )
        g = HDAGraph(
            type_name="labs::quickmaterial::2.0",
            category="Sop",
            label="Quick Material",
            min_inputs=1,
            max_inputs=1,
        )
        g.nodes.append(HDAInternalNode(name="mtl", type="material", display_flag=True))
        c.add(g)

        path = tmp_path / "test.json"
        c.save_json(path)
        loaded = HDAGraphCorpus.load_json(path)
        assert loaded.hda_count == 1
        assert loaded.houdini_version == "21.0.506"
        graph = loaded.get_graph("sop/labs::quickmaterial::2.0")
        assert graph is not None
        assert graph.label == "Quick Material"
        assert graph.node_count == 1
        assert graph.nodes[0].display_flag is True

    def test_idempotent_output(self, tmp_path):
        c = HDAGraphCorpus()
        c.add(HDAGraph(type_name="labs::test", category="Sop"))
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        c.save_json(p1)
        c.save_json(p2)
        assert p1.read_text() == p2.read_text()

    def test_json_structure(self, tmp_path):
        c = HDAGraphCorpus(houdini_version="21.0")
        c.add(HDAGraph(type_name="labs::test", category="Sop"))
        path = tmp_path / "structure.json"
        c.save_json(path)
        data = json.loads(path.read_text())
        assert data["version"] == "1.0"
        assert data["houdini_version"] == "21.0"
        assert data["hda_count"] == 1
        assert "contexts" in data
        assert "graphs" in data


# ---------------------------------------------------------------------------
# Fixture-based tests (sample_graphs.json)
# ---------------------------------------------------------------------------

class TestFixture:
    @pytest.fixture(scope="class")
    def corpus(self):
        return HDAGraphCorpus.load_json(FIXTURE_PATH)

    def test_hda_count(self, corpus):
        assert corpus.hda_count == 3

    def test_contexts(self, corpus):
        assert "sop" in corpus.contexts()
        assert "top" in corpus.contexts()

    def test_quickmaterial(self, corpus):
        g = corpus.get_graph("sop/labs::quickmaterial::2.0")
        assert g is not None
        assert g.label == "Quick Material"
        assert g.category == "Sop"
        assert g.context == "sop"
        assert g.min_inputs == 1
        assert g.max_inputs == 1
        assert g.node_count == 3
        assert g.connection_count == 1

    def test_quickmaterial_subnet_input(self, corpus):
        g = corpus.get_graph("sop/labs::quickmaterial::2.0")
        assert len(g.subnet_inputs) == 1
        si = g.subnet_inputs[0]
        assert si.index == 0
        assert si.name == "Input 1"
        assert len(si.connections) == 1
        assert si.connections[0] == ("material_assign", 0)

    def test_quickmaterial_params(self, corpus):
        g = corpus.get_graph("sop/labs::quickmaterial::2.0")
        mtl = next(n for n in g.nodes if n.name == "material_assign")
        assert len(mtl.parameters) == 1
        p = mtl.parameters[0]
        assert p.name == "shop_materialpath1"
        assert p.expression == 'ch("../material_path")'
        assert p.expression_language == "hscript"

    def test_quickmaterial_output_node(self, corpus):
        g = corpus.get_graph("sop/labs::quickmaterial::2.0")
        output = next(n for n in g.nodes if n.name == "output0")
        assert output.is_output is True
        assert output.type == "output"

    def test_edge_damage(self, corpus):
        g = corpus.get_graph("sop/labs::edge_damage::2.0")
        assert g is not None
        assert g.node_count == 4
        assert g.connection_count == 2

    def test_edge_damage_subnet_input_fan_out(self, corpus):
        g = corpus.get_graph("sop/labs::edge_damage::2.0")
        assert len(g.subnet_inputs) == 1
        si = g.subnet_inputs[0]
        # Subnet input fans out to two nodes
        assert len(si.connections) == 2

    def test_top_node(self, corpus):
        g = corpus.get_graph("top/labs::wedge_postprocess")
        assert g is not None
        assert g.category == "Top"
        assert g.context == "top"
        assert g.node_count == 1
        assert g.connection_count == 0

    def test_library_path(self, corpus):
        g = corpus.get_graph("sop/labs::quickmaterial::2.0")
        assert "SideFX_Labs" in g.library_path

    def test_roundtrip_via_file(self, corpus, tmp_path):
        path = tmp_path / "rt.json"
        corpus.save_json(path)
        loaded = HDAGraphCorpus.load_json(path)
        assert loaded.hda_count == corpus.hda_count
        assert loaded.contexts() == corpus.contexts()
        g = loaded.get_graph("sop/labs::quickmaterial::2.0")
        assert g.node_count == 3
        assert len(g.subnet_inputs) == 1

    def test_keys_lowercase(self, corpus):
        """All keys should be context/type_name."""
        for key in corpus.graphs:
            ctx, name = key.split("/", 1)
            assert ctx == ctx.lower()


# ---------------------------------------------------------------------------
# Integration tests (requires hython + Labs HDAs)
# ---------------------------------------------------------------------------

LABS_HDA_PATH = Path(
    "/opt/sidefx/sidefx_packages/SideFXLabs21.0/otls/SideFX_Labs.hda"
)
LABS_AVAILABLE = HYTHON_AVAILABLE and LABS_HDA_PATH.exists()


@pytest.mark.skipif(not LABS_AVAILABLE, reason="hython or Labs HDAs not available")
class TestHythonIntegration:
    @pytest.fixture(scope="class")
    def corpus(self, tmp_path_factory):
        extractor = LabsHDAExtractor(
            categories=["Sop"],
            timeout=120,
        )
        return extractor.extract()

    def test_sop_count(self, corpus):
        # Labs should have many SOP HDAs
        assert corpus.hda_count > 50

    def test_has_internal_nodes(self, corpus):
        # At least some HDAs should have child nodes
        has_children = [g for g in corpus.graphs.values() if g.node_count > 0]
        assert len(has_children) > 10

    def test_has_connections(self, corpus):
        has_conns = [g for g in corpus.graphs.values() if g.connection_count > 0]
        assert len(has_conns) > 5

    def test_houdini_version(self, corpus):
        assert corpus.houdini_version != ""

    def test_all_have_category(self, corpus):
        for g in corpus.graphs.values():
            assert g.category == "Sop"
            assert g.context == "sop"

    def test_library_paths_contain_labs(self, corpus):
        for g in corpus.graphs.values():
            assert "SideFXLabs" in g.library_path


# ---------------------------------------------------------------------------
# Extractor error handling
# ---------------------------------------------------------------------------

class TestExtractorErrors:
    def test_missing_hython(self):
        extractor = LabsHDAExtractor(hython_path="/nonexistent/hython")
        with pytest.raises(FileNotFoundError, match="hython not found"):
            extractor.extract()
