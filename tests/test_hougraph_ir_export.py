"""Tests for HouGraph IR export adapter."""

import pytest
from pathlib import Path

from src.analysis.visual_extractor import (
    GraphExtraction,
    NodeExtraction,
    ConnectionExtraction,
)

# Skip tests if hougraph-ir is not installed
pytest.importorskip("hougraph_ir")

from src.output.hougraph_ir_export import (
    HouGraphIRExporter,
    export_extractions_to_hougraph_ir,
    export_extractions_to_dict,
    _map_network_context,
)
from hougraph_ir import HouGraphIR, NetworkContext


class TestNetworkContextMapping:
    def test_map_sop_context(self):
        assert _map_network_context("SOP") == NetworkContext.SOP
        assert _map_network_context("sop") == NetworkContext.SOP

    def test_map_dop_context(self):
        assert _map_network_context("DOP") == NetworkContext.DOP

    def test_map_vop_context(self):
        assert _map_network_context("VOP") == NetworkContext.VOP

    def test_map_unknown_defaults_to_sop(self):
        assert _map_network_context("unknown") == NetworkContext.SOP
        assert _map_network_context("invalid") == NetworkContext.SOP


class TestHouGraphIRExporter:
    @pytest.fixture
    def exporter(self):
        return HouGraphIRExporter()

    @pytest.fixture
    def sample_extraction(self):
        return GraphExtraction(
            network_context="SOP",
            parent_path="/obj/geo1",
            nodes=[
                NodeExtraction(
                    name="sphere1",
                    type="sphere",
                    position=[25.0, 20.0],
                    flags={"display": False, "render": False, "bypass": False},
                ),
                NodeExtraction(
                    name="mountain1",
                    type="mountain",
                    position=[25.0, 40.0],
                    flags={"display": True, "render": True, "bypass": False},
                ),
            ],
            connections=[
                ConnectionExtraction(
                    from_node="sphere1",
                    from_output=0,
                    to_node="mountain1",
                    to_input=0,
                ),
            ],
            extraction_confidence=0.85,
            graph_visible=True,
            graph_area_percent=65,
            readability="high",
            timestamp=45.5,
        )

    def test_export_basic(self, exporter, sample_extraction):
        result = exporter.export([sample_extraction])

        assert isinstance(result, HouGraphIR)
        assert len(result.nodes) == 2
        assert len(result.connections) == 1
        assert result.context == NetworkContext.SOP

    def test_export_nodes(self, exporter, sample_extraction):
        result = exporter.export([sample_extraction])

        node_names = {n.name for n in result.nodes}
        assert "sphere1" in node_names
        assert "mountain1" in node_names

        mountain = next(n for n in result.nodes if n.name == "mountain1")
        assert mountain.display_flag is True
        assert mountain.render_flag is True

    def test_export_connections(self, exporter, sample_extraction):
        result = exporter.export([sample_extraction])

        conn = result.connections[0]
        assert conn.source_node == "sphere1"
        assert conn.dest_node == "mountain1"
        assert conn.source_output == 0
        assert conn.dest_input == 0

    def test_export_with_metadata(self, exporter, sample_extraction):
        result = exporter.export(
            [sample_extraction],
            video_url="https://youtube.com/watch?v=abc123",
            video_title="Houdini Tutorial",
        )

        assert result.source is not None
        assert result.source["type"] == "youtube"
        assert result.source["url"] == "https://youtube.com/watch?v=abc123"
        assert result.source["title"] == "Houdini Tutorial"

    def test_export_empty_extractions(self, exporter):
        result = exporter.export([])
        assert len(result.nodes) == 0
        assert len(result.connections) == 0

    def test_export_no_valid_extractions(self, exporter):
        empty_extraction = GraphExtraction(
            network_context="SOP",
            parent_path=None,
            nodes=[],
            connections=[],
            extraction_confidence=0.0,
            graph_visible=False,
            graph_area_percent=0,
            readability="none",
        )
        result = exporter.export([empty_extraction])
        assert len(result.nodes) == 0

    def test_position_normalization(self, exporter, sample_extraction):
        result = exporter.export([sample_extraction])

        sphere = next(n for n in result.nodes if n.name == "sphere1")
        # Position [25.0, 20.0] should be normalized
        # x = (25/100)*20 - 10 = -5
        # y = -(20/100)*20 + 10 = 6
        assert sphere.position.x == -5.0
        assert sphere.position.y == 6.0

    def test_export_to_dict(self, exporter, sample_extraction):
        result = exporter.export_to_dict([sample_extraction])

        assert isinstance(result, dict)
        assert "nodes" in result
        assert "connections" in result
        assert len(result["nodes"]) == 2

    def test_export_to_json(self, exporter, sample_extraction):
        result = exporter.export_to_json([sample_extraction])

        assert isinstance(result, str)
        assert "sphere1" in result
        assert "mountain1" in result


class TestDeduplication:
    @pytest.fixture
    def exporter(self):
        return HouGraphIRExporter()

    def test_deduplicate_nodes_keeps_highest_confidence(self, exporter):
        extraction1 = GraphExtraction(
            network_context="SOP",
            parent_path=None,
            nodes=[
                NodeExtraction(name="sphere1", type="sphere", position=[10, 10]),
            ],
            connections=[],
            extraction_confidence=0.7,
            graph_visible=True,
            graph_area_percent=50,
            readability="medium",
            timestamp=10.0,
        )
        extraction2 = GraphExtraction(
            network_context="SOP",
            parent_path=None,
            nodes=[
                NodeExtraction(name="sphere1", type="sphere", position=[12, 12]),
            ],
            connections=[],
            extraction_confidence=0.9,
            graph_visible=True,
            graph_area_percent=60,
            readability="high",
            timestamp=20.0,
        )

        result = exporter.export([extraction1, extraction2])
        assert len(result.nodes) == 1
        # Higher confidence extraction should be used for position
        sphere = result.nodes[0]
        # Position from extraction2 (higher confidence)
        assert sphere.position.x == pytest.approx(-7.6, rel=0.1)

    def test_deduplicate_connections(self, exporter):
        extraction1 = GraphExtraction(
            network_context="SOP",
            parent_path=None,
            nodes=[
                NodeExtraction(name="a", type="null", position=[10, 10]),
                NodeExtraction(name="b", type="null", position=[10, 30]),
            ],
            connections=[
                ConnectionExtraction(from_node="a", from_output=0, to_node="b", to_input=0),
            ],
            extraction_confidence=0.8,
            graph_visible=True,
            graph_area_percent=50,
            readability="high",
            timestamp=10.0,
        )
        extraction2 = GraphExtraction(
            network_context="SOP",
            parent_path=None,
            nodes=[
                NodeExtraction(name="a", type="null", position=[10, 10]),
                NodeExtraction(name="b", type="null", position=[10, 30]),
            ],
            connections=[
                ConnectionExtraction(from_node="a", from_output=0, to_node="b", to_input=0),
            ],
            extraction_confidence=0.85,
            graph_visible=True,
            graph_area_percent=55,
            readability="high",
            timestamp=15.0,
        )

        result = exporter.export([extraction1, extraction2])
        assert len(result.connections) == 1


class TestConvenienceFunctions:
    @pytest.fixture
    def sample_extraction(self):
        return GraphExtraction(
            network_context="SOP",
            parent_path="/obj/geo1",
            nodes=[
                NodeExtraction(name="test1", type="null", position=[50, 50]),
            ],
            connections=[],
            extraction_confidence=0.8,
            graph_visible=True,
            graph_area_percent=50,
            readability="high",
        )

    def test_export_extractions_to_hougraph_ir(self, sample_extraction):
        result = export_extractions_to_hougraph_ir([sample_extraction])
        assert isinstance(result, HouGraphIR)
        assert len(result.nodes) == 1

    def test_export_extractions_to_dict(self, sample_extraction):
        result = export_extractions_to_dict([sample_extraction])
        assert isinstance(result, dict)
        assert len(result["nodes"]) == 1


class TestSerialization:
    @pytest.fixture
    def exporter(self):
        return HouGraphIRExporter()

    @pytest.fixture
    def sample_extraction(self):
        return GraphExtraction(
            network_context="SOP",
            parent_path="/obj/geo1",
            nodes=[
                NodeExtraction(name="sphere1", type="sphere", position=[20, 20]),
            ],
            connections=[],
            extraction_confidence=0.9,
            graph_visible=True,
            graph_area_percent=70,
            readability="high",
        )

    def test_save_and_load(self, exporter, sample_extraction, tmp_path):
        output_file = tmp_path / "test_output.json"

        exporter.save([sample_extraction], output_file)
        assert output_file.exists()

        # Load and verify
        loaded = HouGraphIR.from_json(output_file.read_text())
        assert len(loaded.nodes) == 1
        assert loaded.nodes[0].name == "sphere1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
