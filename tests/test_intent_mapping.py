"""Tests for Phase 2B: Intent-to-Subgraph Mapping."""

from pathlib import Path

import pytest

from src.analysis.intent_mapping.models import (
    IntentCluster,
    IntentLibrary,
    SubgraphTemplate,
)
from src.analysis.intent_mapping.mapper import IntentMapper, STOPWORDS

from src.ingestion.labs_hda.models import HDAGraphCorpus

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "labs_hda" / "sample_graphs.json"


# ---------------------------------------------------------------------------
# Unit tests: SubgraphTemplate
# ---------------------------------------------------------------------------

class TestSubgraphTemplate:
    def test_roundtrip(self):
        t = SubgraphTemplate(
            hda_key="sop/labs::road_generator::2.0",
            label="Labs Road Generator",
            category="Sop",
            context="sop",
            node_types=["scatter", "mountain::2.0"],
            node_count=5,
            connection_count=4,
        )
        d = t.to_dict()
        t2 = SubgraphTemplate.from_dict(d)
        assert t2.hda_key == t.hda_key
        assert t2.label == t.label
        assert t2.category == "Sop"
        assert t2.context == "sop"
        assert t2.node_types == ["scatter", "mountain::2.0"]
        assert t2.node_count == 5
        assert t2.connection_count == 4

    def test_defaults(self):
        t = SubgraphTemplate(
            hda_key="sop/labs::test",
            label="Test",
            category="Sop",
            context="sop",
        )
        assert t.node_types == []
        assert t.node_count == 0
        assert t.connection_count == 0


# ---------------------------------------------------------------------------
# Unit tests: IntentCluster
# ---------------------------------------------------------------------------

class TestIntentCluster:
    def test_template_count(self):
        c = IntentCluster(intent_id="test", keywords=["test"])
        assert c.template_count == 0
        c.templates.append(SubgraphTemplate(
            hda_key="sop/labs::test",
            label="Test",
            category="Sop",
            context="sop",
        ))
        assert c.template_count == 1

    def test_roundtrip(self):
        c = IntentCluster(
            intent_id="road_generator",
            keywords=["road", "generator"],
            description="Road Generator",
            category="sop",
            templates=[
                SubgraphTemplate(
                    hda_key="sop/labs::road_generator::2.0",
                    label="Labs Road Generator",
                    category="Sop",
                    context="sop",
                    node_types=["scatter"],
                    node_count=5,
                    connection_count=4,
                ),
            ],
        )
        d = c.to_dict()
        c2 = IntentCluster.from_dict(d)
        assert c2.intent_id == "road_generator"
        assert c2.keywords == ["road", "generator"]
        assert c2.description == "Road Generator"
        assert c2.category == "sop"
        assert c2.template_count == 1
        assert c2.templates[0].hda_key == "sop/labs::road_generator::2.0"


# ---------------------------------------------------------------------------
# Unit tests: IntentLibrary
# ---------------------------------------------------------------------------

class TestIntentLibrary:
    def test_empty(self):
        lib = IntentLibrary()
        assert lib.intent_count == 0
        assert lib.template_count == 0
        assert lib.search("anything") == []
        assert lib.get_by_category("sop") == []

    def test_counts(self):
        lib = IntentLibrary()
        lib.intents["a"] = IntentCluster(
            intent_id="a", keywords=["a"], category="sop",
            templates=[
                SubgraphTemplate(hda_key="sop/a1", label="A1", category="Sop", context="sop"),
                SubgraphTemplate(hda_key="sop/a2", label="A2", category="Sop", context="sop"),
            ],
        )
        lib.intents["b"] = IntentCluster(
            intent_id="b", keywords=["b"], category="top",
            templates=[
                SubgraphTemplate(hda_key="top/b1", label="B1", category="Top", context="top"),
            ],
        )
        assert lib.intent_count == 2
        assert lib.template_count == 3

    def test_search(self):
        lib = IntentLibrary()
        lib.intents["road_generator"] = IntentCluster(
            intent_id="road_generator",
            keywords=["road", "generator"],
            description="Road Generator",
            category="sop",
        )
        lib.intents["tree_generator"] = IntentCluster(
            intent_id="tree_generator",
            keywords=["tree", "generator"],
            description="Tree Generator",
            category="sop",
        )
        results = lib.search("road")
        assert len(results) == 1
        assert results[0].intent_id == "road_generator"

        # "generator" matches both
        results = lib.search("generator")
        assert len(results) == 2

    def test_search_limit(self):
        lib = IntentLibrary()
        for i in range(20):
            lib.intents[f"test_{i}"] = IntentCluster(
                intent_id=f"test_{i}",
                keywords=["test"],
                description="Test",
                category="sop",
            )
        results = lib.search("test", limit=5)
        assert len(results) == 5

    def test_get_by_category(self):
        lib = IntentLibrary()
        lib.intents["a"] = IntentCluster(intent_id="a", category="sop")
        lib.intents["b"] = IntentCluster(intent_id="b", category="top")
        lib.intents["c"] = IntentCluster(intent_id="c", category="sop")
        sop = lib.get_by_category("sop")
        assert len(sop) == 2
        assert sop[0].intent_id == "a"
        assert sop[1].intent_id == "c"
        assert lib.get_by_category("dop") == []

    def test_save_load_roundtrip(self, tmp_path):
        lib = IntentLibrary()
        lib.intents["test"] = IntentCluster(
            intent_id="test",
            keywords=["test"],
            description="Test",
            category="sop",
            templates=[
                SubgraphTemplate(
                    hda_key="sop/test",
                    label="Test",
                    category="Sop",
                    context="sop",
                    node_types=["box"],
                    node_count=1,
                    connection_count=0,
                ),
            ],
        )
        path = tmp_path / "intent_library.json"
        lib.save_json(path)
        loaded = IntentLibrary.load_json(path)
        assert loaded.intent_count == 1
        assert loaded.template_count == 1
        assert loaded.intents["test"].keywords == ["test"]

    def test_to_dict_structure(self):
        lib = IntentLibrary()
        d = lib.to_dict()
        assert d["version"] == "1.0"
        assert d["intent_count"] == 0
        assert d["template_count"] == 0
        assert "intents" in d


# ---------------------------------------------------------------------------
# Unit tests: IntentMapper label normalization
# ---------------------------------------------------------------------------

class TestIntentMapperNormalization:
    def setup_method(self):
        self.mapper = IntentMapper()

    def test_labs_prefix_stripped(self):
        assert self.mapper._normalize_label("Labs Road Generator") == ["road", "generator"]

    def test_vendor_prefix_stripped(self):
        assert self.mapper._normalize_label("Labs Gaea Heightfield Import") == [
            "heightfield", "import"
        ]

    def test_version_suffix_stripped(self):
        assert self.mapper._normalize_label("Labs Road Generator 2.0") == ["road", "generator"]

    def test_stopwords_removed(self):
        assert self.mapper._normalize_label("Labs Building from Patterns") == [
            "building", "patterns"
        ]

    def test_av_prefix_preserved(self):
        tokens = self.mapper._normalize_label("Labs AV Structure from Motion")
        assert "av" in tokens
        assert "structure" in tokens
        assert "motion" in tokens
        assert "from" not in tokens

    def test_empty_label(self):
        assert self.mapper._normalize_label("") == []

    def test_labs_only(self):
        assert self.mapper._normalize_label("Labs") == []

    def test_no_labs_prefix(self):
        assert self.mapper._normalize_label("Edge Damage") == ["edge", "damage"]


# ---------------------------------------------------------------------------
# Fixture-based tests: IntentMapper with sample_graphs.json
# ---------------------------------------------------------------------------

class TestIntentMapperFixture:
    """Tests mapping against the sample_graphs.json fixture.

    Fixture has 3 HDAs (labels stripped of "Labs " prefix since fixture labels
    don't have it):
    - "Quick Material" (Sop) → intent_id: "quick_material"
    - "Edge Damage" (Sop) → intent_id: "edge_damage"
    - "Wedge Postprocess" (Top) → intent_id: "wedge_postprocess"
    """

    @pytest.fixture(scope="class")
    def corpus(self):
        return HDAGraphCorpus.load_json(FIXTURE_PATH)

    @pytest.fixture(scope="class")
    def library(self, corpus):
        mapper = IntentMapper()
        return mapper.map(corpus)

    def test_intent_count(self, library):
        assert library.intent_count == 3

    def test_template_count(self, library):
        assert library.template_count == 3

    def test_intent_ids(self, library):
        assert "quick_material" in library.intents
        assert "edge_damage" in library.intents
        assert "wedge_postprocess" in library.intents

    def test_keywords(self, library):
        assert library.intents["quick_material"].keywords == ["quick", "material"]
        assert library.intents["edge_damage"].keywords == ["edge", "damage"]
        assert library.intents["wedge_postprocess"].keywords == ["wedge", "postprocess"]

    def test_descriptions(self, library):
        assert library.intents["quick_material"].description == "Quick Material"
        assert library.intents["edge_damage"].description == "Edge Damage"
        assert library.intents["wedge_postprocess"].description == "Wedge Postprocess"

    def test_categories(self, library):
        assert library.intents["quick_material"].category == "sop"
        assert library.intents["edge_damage"].category == "sop"
        assert library.intents["wedge_postprocess"].category == "top"

    def test_template_node_types(self, library):
        # quick_material: material, principledshader::2.0 (output excluded)
        qm = library.intents["quick_material"].templates[0]
        assert "material" in qm.node_types
        assert "principledshader::2.0" in qm.node_types
        assert "output" not in qm.node_types
        assert qm.node_count == 2

        # edge_damage: groupcreate, measure, mountain::2.0 (output excluded)
        ed = library.intents["edge_damage"].templates[0]
        assert "groupcreate" in ed.node_types
        assert "measure" in ed.node_types
        assert "mountain::2.0" in ed.node_types
        assert ed.node_count == 3

    def test_template_hda_keys(self, library):
        assert library.intents["quick_material"].templates[0].hda_key == \
            "sop/labs::quickmaterial::2.0"
        assert library.intents["edge_damage"].templates[0].hda_key == \
            "sop/labs::edge_damage::2.0"
        assert library.intents["wedge_postprocess"].templates[0].hda_key == \
            "top/labs::wedge_postprocess"

    def test_search_material(self, library):
        results = library.search("material")
        assert len(results) == 1
        assert results[0].intent_id == "quick_material"

    def test_search_edge(self, library):
        # "edge" substring-matches both "edge" (edge_damage) and "wedge" (wedge_postprocess)
        results = library.search("edge")
        assert len(results) == 2
        assert results[0].intent_id == "edge_damage"

    def test_search_damage(self, library):
        # "damage" only matches edge_damage
        results = library.search("damage")
        assert len(results) == 1
        assert results[0].intent_id == "edge_damage"

    def test_search_no_match(self, library):
        results = library.search("nonexistent")
        assert len(results) == 0

    def test_get_by_category_sop(self, library):
        sop = library.get_by_category("sop")
        assert len(sop) == 2
        ids = {c.intent_id for c in sop}
        assert "quick_material" in ids
        assert "edge_damage" in ids

    def test_get_by_category_top(self, library):
        top = library.get_by_category("top")
        assert len(top) == 1
        assert top[0].intent_id == "wedge_postprocess"

    def test_save_load_roundtrip(self, library, tmp_path):
        path = tmp_path / "test_intent_library.json"
        library.save_json(path)
        loaded = IntentLibrary.load_json(path)
        assert loaded.intent_count == library.intent_count
        assert loaded.template_count == library.template_count
        assert loaded.intents["edge_damage"].keywords == ["edge", "damage"]


# ---------------------------------------------------------------------------
# IntentMapper: min_node_count filter
# ---------------------------------------------------------------------------

class TestIntentMapperMinNodes:
    @pytest.fixture(scope="class")
    def corpus(self):
        return HDAGraphCorpus.load_json(FIXTURE_PATH)

    def test_min_nodes_2(self, corpus):
        # wedge_postprocess has only 1 non-output node → excluded
        mapper = IntentMapper(min_node_count=2)
        library = mapper.map(corpus)
        assert library.intent_count == 2
        assert "wedge_postprocess" not in library.intents

    def test_min_nodes_3(self, corpus):
        # quick_material has 2 non-output nodes → excluded
        # only edge_damage (3 non-output nodes) survives
        mapper = IntentMapper(min_node_count=3)
        library = mapper.map(corpus)
        assert library.intent_count == 1
        assert "edge_damage" in library.intents

    def test_min_nodes_10(self, corpus):
        # All filtered out
        mapper = IntentMapper(min_node_count=10)
        library = mapper.map(corpus)
        assert library.intent_count == 0


# ---------------------------------------------------------------------------
# IntentMapper: exclude_types override
# ---------------------------------------------------------------------------

class TestIntentMapperExcludeTypes:
    @pytest.fixture(scope="class")
    def corpus(self):
        return HDAGraphCorpus.load_json(FIXTURE_PATH)

    def test_no_exclusion(self, corpus):
        mapper = IntentMapper(exclude_types=set())
        library = mapper.map(corpus)
        # All 3 HDAs included, output nodes counted
        assert library.intent_count == 3
        qm = library.intents["quick_material"].templates[0]
        assert "output" in qm.node_types
        assert qm.node_count == 3  # material + principledshader + output
