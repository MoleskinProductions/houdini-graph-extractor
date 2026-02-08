"""Unit tests for HelpFileParser."""

from pathlib import Path

import pytest

from src.ingestion.help_docs.parser import HelpFileParser

FIXTURES = Path(__file__).parent / "fixtures" / "help_docs"


@pytest.fixture
def parser():
    return HelpFileParser()


def _load(name: str) -> str:
    return (FIXTURES / name).read_text()


# ---- Include-type files should be skipped ----

class TestIncludeSkip:
    def test_include_type_returns_none(self, parser):
        doc = parser.parse(_load("include_type.txt"), filename="apex/_transform_shared.txt")
        assert doc is None


# ---- Header metadata ----

class TestHeaderParsing:
    def test_scatter_header(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        assert doc is not None
        assert doc.context == "sop"
        assert doc.internal_name == "scatter"
        assert doc.title == "Scatter"
        assert doc.icon == "SOP/scatter"
        assert doc.tags == ["copy", "random", "points"]
        assert doc.version == "2.0"
        assert doc.since_version == "14.0"

    def test_box_header(self, parser):
        doc = parser.parse(_load("box_sop.txt"), filename="sop/box.txt")
        assert doc.context == "sop"
        assert doc.internal_name == "box"
        assert doc.tags == ["create", "model"]

    def test_noise_vop_header(self, parser):
        doc = parser.parse(_load("noise_vop.txt"), filename="vop/aanoise.txt")
        assert doc.context == "vop"
        assert doc.internal_name == "aanoise"
        assert doc.title == "Anti-Aliased Noise"

    def test_ropfetch_header_with_whitespace(self, parser):
        """The ropfetch fixture has extra whitespace in directives."""
        doc = parser.parse(_load("ropfetch_top.txt"), filename="top/ropfetch.txt")
        assert doc.context == "top"
        assert doc.internal_name == "ropfetch"
        assert doc.since_version == "17.5"

    def test_key_property(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        assert doc.key == "sop/scatter"


# ---- Brief description ----

class TestBrief:
    def test_scatter_brief(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        assert doc.brief == "Scatters new points randomly across a surface or through a volume."

    def test_noise_brief(self, parser):
        doc = parser.parse(_load("noise_vop.txt"), filename="vop/aanoise.txt")
        assert "anti-aliased noise" in doc.brief.lower()


# ---- Content sections ----

class TestSections:
    def test_scatter_has_sections(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        titles = [s.title for s in doc.sections]
        assert "Overview" in titles
        assert "Making scattered points stick on changing geometry" in titles

    def test_section_content(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        overview = next(s for s in doc.sections if s.title == "Overview")
        assert "distributes new points" in overview.content


# ---- Parameters ----

class TestParameters:
    def test_scatter_param_count(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        # Group, Generate, Density Scale, Force Total Count, Point Attributes
        # (Distribution group params are in parameter_groups)
        assert len(doc.parameters) >= 4

    def test_param_id(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        group_param = next((p for p in doc.parameters if p.label == "Group"), None)
        assert group_param is not None
        assert group_param.id == "group"

    def test_param_channels(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        density = next((p for p in doc.parameters if p.label == "Density Scale"), None)
        assert density is not None
        assert density.channels == "/densityscale"

    def test_param_description(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        group_param = next(p for p in doc.parameters if p.label == "Group")
        assert "primitives" in group_param.description.lower()

    def test_menu_options(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        gen = next((p for p in doc.parameters if p.label == "Generate"), None)
        assert gen is not None
        assert len(gen.menu_options) == 3
        labels = [m.label for m in gen.menu_options]
        assert "By Density" in labels
        assert "Count per Primitive" in labels
        assert "In Texture Space" in labels

    def test_box_params(self, parser):
        doc = parser.parse(_load("box_sop.txt"), filename="sop/box.txt")
        labels = [p.label for p in doc.parameters]
        assert "Size" in labels
        assert "Center" in labels

    def test_box_primitive_type_menu(self, parser):
        doc = parser.parse(_load("box_sop.txt"), filename="sop/box.txt")
        ptype = next((p for p in doc.parameters if p.label == "Primitive Type"), None)
        assert ptype is not None
        assert len(ptype.menu_options) >= 3

    def test_math_chop_params(self, parser):
        doc = parser.parse(_load("math_chop.txt"), filename="chop/math.txt")
        pre_add = next((p for p in doc.parameters if p.label == "Pre-Add"), None)
        assert pre_add is not None
        assert pre_add.channels == "/preoff"


# ---- Parameter groups (~~~ Subsection ~~~) ----

class TestParameterGroups:
    def test_scatter_distribution_group(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        assert len(doc.parameter_groups) >= 1
        dist = next((g for g in doc.parameter_groups if g.label == "Distribution"), None)
        assert dist is not None
        labels = [p.label for p in dist.parameters]
        assert "Relax Iterations" in labels
        assert "Max Relax Radius" in labels


# ---- Inputs / Outputs ----

class TestInputsOutputs:
    def test_noise_inputs(self, parser):
        doc = parser.parse(_load("noise_vop.txt"), filename="vop/aanoise.txt")
        assert len(doc.inputs) == 2
        pos = next(p for p in doc.inputs if p.name == "pos")
        assert pos.type == "Vector3"

    def test_noise_outputs(self, parser):
        doc = parser.parse(_load("noise_vop.txt"), filename="vop/aanoise.txt")
        assert len(doc.outputs) == 2
        noise = next(p for p in doc.outputs if p.name == "noise")
        assert noise.type == "Float"
        dnoise = next(p for p in doc.outputs if p.name == "dNdP")
        assert dnoise.type == "Vector3"


# ---- Related nodes ----

class TestRelated:
    def test_scatter_related(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        assert "sop/copy" in doc.related
        assert "sop/attribinterpolate" in doc.related
        assert len(doc.related) == 4

    def test_noise_related(self, parser):
        doc = parser.parse(_load("noise_vop.txt"), filename="vop/aanoise.txt")
        assert "vop/periodicnoise" in doc.related


# ---- Local variables (@locals) ----

class TestLocals:
    def test_math_locals(self, parser):
        doc = parser.parse(_load("math_chop.txt"), filename="chop/math.txt")
        assert len(doc.locals) == 4
        names = [v.name for v in doc.locals]
        assert "I" in names
        assert "V" in names
        assert "NC" in names

    def test_local_description(self, parser):
        doc = parser.parse(_load("math_chop.txt"), filename="chop/math.txt")
        i_var = next(v for v in doc.locals if v.name == "I")
        assert "current index" in i_var.description.lower()


# ---- TOP attributes ----

class TestTopAttributes:
    def test_ropfetch_top_attributes(self, parser):
        doc = parser.parse(_load("ropfetch_top.txt"), filename="top/ropfetch.txt")
        assert len(doc.top_attributes) == 5
        hip = next(a for a in doc.top_attributes if a.name == "hip")
        assert hip.type == "string"
        assert ".hip" in hip.description

    def test_ropfetch_range_attr(self, parser):
        doc = parser.parse(_load("ropfetch_top.txt"), filename="top/ropfetch.txt")
        rng = next(a for a in doc.top_attributes if a.name == "range")
        assert rng.type == "float3"


# ---- Include directives ----

class TestIncludes:
    def test_scatter_includes(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        assert "/shelf/scatter#includeme" in doc.includes

    def test_math_includes(self, parser):
        doc = parser.parse(_load("math_chop.txt"), filename="chop/math.txt")
        assert "common#common" in doc.includes

    def test_ropfetch_includes(self, parser):
        doc = parser.parse(_load("ropfetch_top.txt"), filename="top/ropfetch.txt")
        assert "processor_common#pdg_workitemgeneration" in doc.includes


# ---- BOM handling ----

class TestBOMHandling:
    def test_bom_stripped(self, parser):
        text = "\ufeff" + _load("box_sop.txt")
        doc = parser.parse(text, filename="sop/box.txt")
        assert doc is not None
        assert doc.context == "sop"


# ---- Serialization roundtrip ----

class TestSerialization:
    def test_to_dict_roundtrip(self, parser):
        doc = parser.parse(_load("scatter_sop.txt"), filename="sop/scatter.txt")
        d = doc.to_dict()
        from src.ingestion.help_docs.models import NodeHelpDoc
        doc2 = NodeHelpDoc.from_dict(d)
        assert doc2.context == doc.context
        assert doc2.internal_name == doc.internal_name
        assert doc2.title == doc.title
        assert doc2.brief == doc.brief
        assert len(doc2.parameters) == len(doc.parameters)
        assert len(doc2.related) == len(doc.related)

    def test_noise_roundtrip(self, parser):
        doc = parser.parse(_load("noise_vop.txt"), filename="vop/aanoise.txt")
        d = doc.to_dict()
        from src.ingestion.help_docs.models import NodeHelpDoc
        doc2 = NodeHelpDoc.from_dict(d)
        assert len(doc2.inputs) == len(doc.inputs)
        assert len(doc2.outputs) == len(doc.outputs)
        assert doc2.inputs[0].type == doc.inputs[0].type
