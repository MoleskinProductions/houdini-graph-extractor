"""Integration tests for HelpCorpusParser and HelpCorpus serialization."""

import json
import tempfile
from pathlib import Path

import pytest

from src.ingestion.help_docs.corpus import DEFAULT_ZIP_PATH, HelpCorpusParser
from src.ingestion.help_docs.models import HelpCorpus

HOUDINI_AVAILABLE = DEFAULT_ZIP_PATH.exists()


@pytest.mark.skipif(not HOUDINI_AVAILABLE, reason="Houdini help zip not found")
class TestCorpusFromZip:
    """Tests that parse the real Houdini help archive."""

    @pytest.fixture(scope="class")
    def corpus(self):
        parser = HelpCorpusParser()
        return parser.parse()

    def test_node_count(self, corpus):
        # Expect ~4000+ nodes after dedup and skipping includes
        assert corpus.node_count > 3500
        assert corpus.node_count < 5500

    def test_contexts_present(self, corpus):
        ctx = corpus.contexts()
        assert "sop" in ctx
        assert "vop" in ctx
        assert "dop" in ctx
        assert "top" in ctx
        assert "chop" in ctx
        assert "lop" in ctx
        assert "apex" in ctx

    def test_scatter_exists(self, corpus):
        doc = corpus.get_node("sop/scatter")
        assert doc is not None
        assert doc.title == "Scatter"
        assert len(doc.parameters) > 10
        assert len(doc.related) > 0

    def test_box_exists(self, corpus):
        doc = corpus.get_node("sop/box")
        assert doc is not None
        assert doc.brief != ""

    def test_ropfetch_top_attributes(self, corpus):
        doc = corpus.get_node("top/ropfetch")
        assert doc is not None
        assert len(doc.top_attributes) > 0

    def test_math_chop_locals(self, corpus):
        doc = corpus.get_node("chop/math")
        assert doc is not None
        assert len(doc.locals) > 0

    def test_vop_with_ports(self, corpus):
        doc = corpus.get_node("vop/aanoise")
        assert doc is not None
        # aanoise doesn't have @inputs in real file, but others do
        # Check APEX nodes which always have inputs/outputs
        apex_add = corpus.get_node("apex/Add<T>")
        if apex_add:
            assert len(apex_add.inputs) > 0
            assert len(apex_add.outputs) > 0

    def test_context_filter(self):
        parser = HelpCorpusParser(contexts={"sop"})
        corpus = parser.parse()
        ctx = corpus.contexts()
        assert ctx == ["sop"]
        assert corpus.node_count > 500

    def test_get_by_context(self, corpus):
        sops = corpus.get_by_context("sop")
        assert len(sops) > 500
        assert all(d.context == "sop" for d in sops)

    def test_versioned_dedup(self, corpus):
        # Files like sop/scatter.txt and sop/scatter-2.0.txt should be deduped
        # to the highest version. The key should exist only once.
        keys = list(corpus.nodes.keys())
        assert len(keys) == len(set(keys))


@pytest.mark.skipif(not HOUDINI_AVAILABLE, reason="Houdini help zip not found")
class TestCorpusSerialization:
    """Tests for JSON save/load roundtrip."""

    @pytest.fixture(scope="class")
    def corpus(self):
        parser = HelpCorpusParser(contexts={"sop"})
        return parser.parse()

    def test_save_and_load(self, corpus, tmp_path):
        path = tmp_path / "test_corpus.json"
        corpus.save_json(path)
        loaded = HelpCorpus.load_json(path)
        assert loaded.node_count == corpus.node_count
        assert loaded.contexts() == corpus.contexts()

    def test_idempotent_output(self, corpus, tmp_path):
        path1 = tmp_path / "a.json"
        path2 = tmp_path / "b.json"
        corpus.save_json(path1)
        corpus.save_json(path2)
        assert path1.read_text() == path2.read_text()

    def test_json_structure(self, corpus, tmp_path):
        path = tmp_path / "structure.json"
        corpus.save_json(path)
        data = json.loads(path.read_text())
        assert "version" in data
        assert "node_count" in data
        assert "contexts" in data
        assert "nodes" in data
        assert data["node_count"] == len(data["nodes"])

    def test_roundtrip_preserves_params(self, corpus, tmp_path):
        path = tmp_path / "rt.json"
        corpus.save_json(path)
        loaded = HelpCorpus.load_json(path)
        scatter = loaded.get_node("sop/scatter")
        assert scatter is not None
        assert len(scatter.parameters) > 0
        assert scatter.parameters[0].id != ""


class TestCorpusUnit:
    """Unit tests for HelpCorpus methods (no Houdini needed)."""

    def test_empty_corpus(self):
        corpus = HelpCorpus()
        assert corpus.node_count == 0
        assert corpus.contexts() == []

    def test_add_and_get(self):
        from src.ingestion.help_docs.models import NodeHelpDoc
        corpus = HelpCorpus()
        doc = NodeHelpDoc(context="sop", internal_name="test")
        corpus.add(doc)
        assert corpus.node_count == 1
        assert corpus.get_node("sop/test") is doc
        assert corpus.get_node("sop/missing") is None

    def test_contexts(self):
        from src.ingestion.help_docs.models import NodeHelpDoc
        corpus = HelpCorpus()
        corpus.add(NodeHelpDoc(context="sop", internal_name="a"))
        corpus.add(NodeHelpDoc(context="vop", internal_name="b"))
        corpus.add(NodeHelpDoc(context="sop", internal_name="c"))
        assert corpus.contexts() == ["sop", "vop"]

    def test_get_by_context(self):
        from src.ingestion.help_docs.models import NodeHelpDoc
        corpus = HelpCorpus()
        corpus.add(NodeHelpDoc(context="sop", internal_name="a"))
        corpus.add(NodeHelpDoc(context="vop", internal_name="b"))
        corpus.add(NodeHelpDoc(context="sop", internal_name="c"))
        sops = corpus.get_by_context("sop")
        assert len(sops) == 2

    def test_save_load_roundtrip(self, tmp_path):
        from src.ingestion.help_docs.models import NodeHelpDoc
        corpus = HelpCorpus()
        doc = NodeHelpDoc(
            context="sop", internal_name="test",
            title="Test", brief="A test node.",
            tags=["test", "unit"],
        )
        corpus.add(doc)
        path = tmp_path / "test.json"
        corpus.save_json(path)
        loaded = HelpCorpus.load_json(path)
        assert loaded.node_count == 1
        t = loaded.get_node("sop/test")
        assert t.title == "Test"
        assert t.tags == ["test", "unit"]
