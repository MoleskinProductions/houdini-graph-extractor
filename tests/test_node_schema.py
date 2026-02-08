"""Tests for node type schema extraction (Phase 1A)."""

import json
import shutil
from pathlib import Path

import pytest

from src.ingestion.node_schema.models import (
    CATEGORY_TO_CONTEXT,
    NodeTypeSchema,
    ParmSchema,
    PortSchema,
    SchemaCorpus,
)
from src.ingestion.node_schema.extractor import DEFAULT_HYTHON_PATH, NodeSchemaExtractor

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "node_schema" / "sample_schema.json"
HYTHON_AVAILABLE = DEFAULT_HYTHON_PATH.exists()


# ---------------------------------------------------------------------------
# Unit tests: ParmSchema
# ---------------------------------------------------------------------------

class TestParmSchema:
    def test_minimal(self):
        p = ParmSchema(name="foo")
        assert p.name == "foo"
        assert p.type == ""
        assert p.size == 1
        assert p.default is None
        d = p.to_dict()
        assert d == {"name": "foo"}

    def test_roundtrip(self):
        p = ParmSchema(
            name="npts",
            label="Force Total Count",
            type="Int",
            default=[1000],
            min_value=0,
            min_is_strict=True,
            help="Number of points",
            tags={"script_callback": "hou.phm().update()"},
        )
        d = p.to_dict()
        p2 = ParmSchema.from_dict(d)
        assert p2.name == p.name
        assert p2.label == p.label
        assert p2.default == p.default
        assert p2.min_value == p.min_value
        assert p2.min_is_strict is True
        assert p2.help == p.help
        assert p2.tags == p.tags

    def test_menu_parm(self):
        p = ParmSchema(
            name="xOrd",
            type="Menu",
            menu_items=["srt", "str"],
            menu_labels=["Scale Rot Trans", "Scale Trans Rot"],
        )
        d = p.to_dict()
        assert d["menu_items"] == ["srt", "str"]
        p2 = ParmSchema.from_dict(d)
        assert p2.menu_labels == ["Scale Rot Trans", "Scale Trans Rot"]

    def test_conditionals(self):
        p = ParmSchema(
            name="densityattrib",
            type="String",
            disable_when="{ usedensityattrib == 0 }",
            hide_when="{ mode == 1 }",
        )
        d = p.to_dict()
        assert d["disable_when"] == "{ usedensityattrib == 0 }"
        p2 = ParmSchema.from_dict(d)
        assert p2.hide_when == "{ mode == 1 }"

    def test_hidden_omitted_when_false(self):
        p = ParmSchema(name="foo", is_hidden=False)
        d = p.to_dict()
        assert "is_hidden" not in d


# ---------------------------------------------------------------------------
# Unit tests: PortSchema
# ---------------------------------------------------------------------------

class TestPortSchema:
    def test_roundtrip(self):
        p = PortSchema(index=0, name="input0", label="Geometry", type="Geo")
        d = p.to_dict()
        p2 = PortSchema.from_dict(d)
        assert p2.index == 0
        assert p2.name == "input0"
        assert p2.label == "Geometry"
        assert p2.type == "Geo"

    def test_minimal(self):
        p = PortSchema(index=1, name="output1")
        d = p.to_dict()
        assert d == {"index": 1, "name": "output1"}


# ---------------------------------------------------------------------------
# Unit tests: NodeTypeSchema
# ---------------------------------------------------------------------------

class TestNodeTypeSchema:
    def test_key_and_context(self):
        s = NodeTypeSchema(category="Sop", type_name="scatter")
        assert s.key == "sop/scatter"
        assert s.context == "sop"

    def test_context_mapping(self):
        for category, expected_ctx in CATEGORY_TO_CONTEXT.items():
            s = NodeTypeSchema(category=category, type_name="test")
            assert s.context == expected_ctx

    def test_roundtrip(self):
        s = NodeTypeSchema(
            category="Sop",
            type_name="scatter",
            label="Scatter",
            base_type="scatter",
            version="2.0",
            min_inputs=1,
            max_inputs=3,
            max_outputs=1,
            icon="SOP_scatter",
            is_generator=False,
            is_hda=False,
        )
        s.parameters.append(ParmSchema(name="npts", type="Int", default=[1000]))
        s.inputs.append(PortSchema(index=0, name="input0", label="Surface"))
        s.outputs.append(PortSchema(index=0, name="output0", label="Points"))

        d = s.to_dict()
        s2 = NodeTypeSchema.from_dict(d)
        assert s2.key == "sop/scatter"
        assert s2.label == "Scatter"
        assert s2.version == "2.0"
        assert len(s2.parameters) == 1
        assert s2.parameters[0].name == "npts"
        assert len(s2.inputs) == 1
        assert len(s2.outputs) == 1

    def test_boolean_flags_omitted_when_false(self):
        s = NodeTypeSchema(category="Sop", type_name="test")
        d = s.to_dict()
        assert "is_generator" not in d
        assert "deprecated" not in d
        assert "is_hda" not in d


# ---------------------------------------------------------------------------
# Unit tests: SchemaCorpus
# ---------------------------------------------------------------------------

class TestSchemaCorpus:
    def test_empty(self):
        c = SchemaCorpus()
        assert c.node_count == 0
        assert c.contexts() == []

    def test_add_and_get(self):
        c = SchemaCorpus()
        s = NodeTypeSchema(category="Sop", type_name="scatter")
        c.add(s)
        assert c.node_count == 1
        assert c.get_node("sop/scatter") is s
        assert c.get_node("sop/missing") is None

    def test_contexts(self):
        c = SchemaCorpus()
        c.add(NodeTypeSchema(category="Sop", type_name="a"))
        c.add(NodeTypeSchema(category="Vop", type_name="b"))
        c.add(NodeTypeSchema(category="Sop", type_name="c"))
        assert c.contexts() == ["sop", "vop"]

    def test_get_by_context(self):
        c = SchemaCorpus()
        c.add(NodeTypeSchema(category="Sop", type_name="a"))
        c.add(NodeTypeSchema(category="Vop", type_name="b"))
        c.add(NodeTypeSchema(category="Sop", type_name="c"))
        sops = c.get_by_context("sop")
        assert len(sops) == 2
        assert all(s.context == "sop" for s in sops)

    def test_save_load_roundtrip(self, tmp_path):
        c = SchemaCorpus(houdini_version="21.0.506", extraction_timestamp="2025-01-15T12:00:00+00:00")
        c.add(NodeTypeSchema(
            category="Sop",
            type_name="scatter",
            label="Scatter",
            min_inputs=1,
            max_inputs=3,
        ))
        path = tmp_path / "test.json"
        c.save_json(path)
        loaded = SchemaCorpus.load_json(path)
        assert loaded.node_count == 1
        assert loaded.houdini_version == "21.0.506"
        s = loaded.get_node("sop/scatter")
        assert s is not None
        assert s.label == "Scatter"
        assert s.min_inputs == 1

    def test_idempotent_output(self, tmp_path):
        c = SchemaCorpus()
        c.add(NodeTypeSchema(category="Sop", type_name="box"))
        p1 = tmp_path / "a.json"
        p2 = tmp_path / "b.json"
        c.save_json(p1)
        c.save_json(p2)
        assert p1.read_text() == p2.read_text()

    def test_json_structure(self, tmp_path):
        c = SchemaCorpus(houdini_version="21.0")
        c.add(NodeTypeSchema(category="Sop", type_name="test"))
        path = tmp_path / "structure.json"
        c.save_json(path)
        data = json.loads(path.read_text())
        assert data["version"] == "1.0"
        assert data["houdini_version"] == "21.0"
        assert data["node_count"] == 1
        assert "contexts" in data
        assert "nodes" in data


# ---------------------------------------------------------------------------
# Fixture-based tests (sample_schema.json)
# ---------------------------------------------------------------------------

class TestFixture:
    @pytest.fixture(scope="class")
    def corpus(self):
        return SchemaCorpus.load_json(FIXTURE_PATH)

    def test_node_count(self, corpus):
        assert corpus.node_count == 4

    def test_contexts(self, corpus):
        assert "sop" in corpus.contexts()
        assert "vop" in corpus.contexts()
        assert "dop" in corpus.contexts()
        assert "obj" in corpus.contexts()

    def test_scatter(self, corpus):
        s = corpus.get_node("sop/scatter")
        assert s is not None
        assert s.label == "Scatter"
        assert s.category == "Sop"
        assert s.context == "sop"
        assert s.version == "2.0"
        assert s.min_inputs == 1
        assert s.max_inputs == 3
        assert len(s.parameters) == 10
        assert len(s.inputs) == 3
        assert len(s.outputs) == 1

    def test_scatter_parm_details(self, corpus):
        s = corpus.get_node("sop/scatter")
        npts = next(p for p in s.parameters if p.name == "npts")
        assert npts.type == "Int"
        assert npts.default == [1000]
        assert npts.min_value == 0
        assert npts.min_is_strict is True

        density = next(p for p in s.parameters if p.name == "densityattrib")
        assert density.disable_when == "{ usedensityattrib == 0 }"

        hidden = next(p for p in s.parameters if p.name == "emergencyLimit")
        assert hidden.is_hidden is True

    def test_geo_object(self, corpus):
        s = corpus.get_node("obj/geo")
        assert s is not None
        assert s.is_generator is True
        assert s.category == "Object"
        assert s.context == "obj"

    def test_vop_aanoise(self, corpus):
        s = corpus.get_node("vop/aanoise")
        assert s is not None
        assert len(s.parameters) == 5
        sig = next(p for p in s.parameters if p.name == "signature")
        assert sig.type == "Menu"
        assert len(sig.menu_items) == 5

    def test_dop_node(self, corpus):
        s = corpus.get_node("dop/rbdpackedobject")
        assert s is not None
        assert s.label == "RBD Packed Object"

    def test_roundtrip_via_file(self, corpus, tmp_path):
        path = tmp_path / "rt.json"
        corpus.save_json(path)
        loaded = SchemaCorpus.load_json(path)
        assert loaded.node_count == corpus.node_count
        assert loaded.contexts() == corpus.contexts()
        s = loaded.get_node("sop/scatter")
        assert len(s.parameters) == 10
        assert len(s.inputs) == 3

    def test_keys_lowercase(self, corpus):
        """All keys should be context/type_name, matching help corpus convention."""
        for key in corpus.nodes:
            ctx, name = key.split("/", 1)
            assert ctx == ctx.lower()


# ---------------------------------------------------------------------------
# Integration tests (requires hython)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HYTHON_AVAILABLE, reason="hython not found at default path")
class TestHythonIntegration:
    @pytest.fixture(scope="class")
    def corpus(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("schema")
        extractor = NodeSchemaExtractor(
            categories=["Sop"],
            timeout=60,
            extract_ports=False,
        )
        return extractor.extract()

    def test_sop_count(self, corpus):
        assert corpus.node_count > 200

    def test_scatter_exists(self, corpus):
        s = corpus.get_node("sop/scatter")
        assert s is not None
        assert s.label != ""
        assert len(s.parameters) > 5

    def test_box_exists(self, corpus):
        s = corpus.get_node("sop/box")
        assert s is not None

    def test_houdini_version(self, corpus):
        assert corpus.houdini_version != ""

    def test_all_have_category(self, corpus):
        for s in corpus.nodes.values():
            assert s.category == "Sop"
            assert s.context == "sop"


@pytest.mark.skipif(not HYTHON_AVAILABLE, reason="hython not found at default path")
class TestHythonWithPorts:
    def test_ports_extracted(self, tmp_path):
        extractor = NodeSchemaExtractor(
            categories=["Sop"],
            timeout=60,
            extract_ports=True,
        )
        corpus = extractor.extract()
        scatter = corpus.get_node("sop/scatter")
        if scatter:
            # scatter has at least 1 input
            assert len(scatter.inputs) >= 1


class TestExtractorErrors:
    def test_missing_hython(self):
        extractor = NodeSchemaExtractor(hython_path="/nonexistent/hython")
        with pytest.raises(FileNotFoundError, match="hython not found"):
            extractor.extract()
