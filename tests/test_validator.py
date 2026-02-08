"""Tests for the StructuralValidator (Phase 1A/2A validation)."""

from pathlib import Path

import pytest

from src.analysis.validator import (
    StructuralValidator,
    ValidationStatus,
    CONF_VALID,
    CONF_VALID_ALIAS,
    CONF_UNKNOWN,
    CONF_KNOWN_PATTERN,
    CONF_UNKNOWN_PATTERN,
)
from src.ingestion.node_schema.models import SchemaCorpus
from src.analysis.pattern_mining.models import PatternCorpus

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA_PATH = FIXTURES / "node_schema" / "sample_schema.json"
PATTERNS_PATH = FIXTURES / "sample_patterns.json"


@pytest.fixture
def schema():
    return SchemaCorpus.load_json(SCHEMA_PATH)


@pytest.fixture
def patterns():
    return PatternCorpus.load_json(PATTERNS_PATH)


@pytest.fixture
def validator(schema, patterns):
    return StructuralValidator(schema=schema, patterns=patterns)


@pytest.fixture
def schema_only_validator(schema):
    return StructuralValidator(schema=schema)


@pytest.fixture
def patterns_only_validator(patterns):
    return StructuralValidator(patterns=patterns)


# ------------------------------------------------------------------
# Schema index building
# ------------------------------------------------------------------

class TestSchemaIndexBuilding:
    def test_key_set_populated(self, validator):
        assert "sop/scatter" in validator._key_set
        assert "vop/aanoise" in validator._key_set
        assert "obj/geo" in validator._key_set
        assert "dop/rbdpackedobject" in validator._key_set

    def test_type_name_index(self, validator):
        assert "scatter" in validator._type_name_to_keys
        assert "sop/scatter" in validator._type_name_to_keys["scatter"]

    def test_label_index(self, validator):
        assert "scatter" in validator._label_to_keys
        assert "anti-aliased noise" in validator._label_to_keys

    def test_context_types(self, validator):
        assert "scatter" in validator._context_types.get("sop", set())
        assert "aanoise" in validator._context_types.get("vop", set())

    def test_schema_node_count(self, validator):
        assert validator.schema_node_count == 4


# ------------------------------------------------------------------
# Type resolution: direct key match
# ------------------------------------------------------------------

class TestDirectKeyMatch:
    def test_full_key(self, validator):
        result = validator.validate_node_type("sop/scatter")
        assert result.status == ValidationStatus.VALID
        assert result.resolved_type == "scatter"
        assert result.resolved_key == "sop/scatter"
        assert result.confidence_adjustment == CONF_VALID

    def test_full_key_case_insensitive(self, validator):
        result = validator.validate_node_type("SOP/Scatter")
        assert result.status == ValidationStatus.VALID
        assert result.resolved_key == "sop/scatter"


# ------------------------------------------------------------------
# Type resolution: type_name index
# ------------------------------------------------------------------

class TestTypeNameResolution:
    def test_bare_type_name(self, validator):
        result = validator.validate_node_type("scatter")
        assert result.status == ValidationStatus.VALID
        assert result.resolved_type == "scatter"
        assert result.resolved_key == "sop/scatter"

    def test_with_context_hint(self, validator):
        result = validator.validate_node_type("scatter", context_hint="SOP")
        assert result.status == ValidationStatus.VALID
        assert result.resolved_key == "sop/scatter"

    def test_geo_resolves_to_obj(self, validator):
        result = validator.validate_node_type("geo")
        assert result.status == ValidationStatus.VALID
        assert result.resolved_key == "obj/geo"


# ------------------------------------------------------------------
# Type resolution: label index
# ------------------------------------------------------------------

class TestLabelResolution:
    def test_label_scatter(self, validator):
        """'Scatter' is both type_name and label — type_name wins."""
        result = validator.validate_node_type("Scatter")
        assert result.status == ValidationStatus.VALID
        assert result.resolved_type == "scatter"

    def test_label_anti_aliased_noise(self, validator):
        result = validator.validate_node_type("Anti-Aliased Noise")
        assert result.status == ValidationStatus.VALID_ALIAS
        assert result.resolved_type == "aanoise"
        assert result.resolved_key == "vop/aanoise"
        assert result.confidence_adjustment == CONF_VALID_ALIAS

    def test_label_rbd_packed_object(self, validator):
        result = validator.validate_node_type("RBD Packed Object")
        assert result.status == ValidationStatus.VALID_ALIAS
        assert result.resolved_type == "rbdpackedobject"
        assert result.resolved_key == "dop/rbdpackedobject"

    def test_label_geometry(self, validator):
        result = validator.validate_node_type("Geometry")
        assert result.status == ValidationStatus.VALID_ALIAS
        assert result.resolved_type == "geo"


# ------------------------------------------------------------------
# Unknown type
# ------------------------------------------------------------------

class TestUnknownType:
    def test_unknown_returns_negative_confidence(self, validator):
        result = validator.validate_node_type("totallyFakeNode")
        assert result.status == ValidationStatus.UNKNOWN
        assert result.resolved_type == "totallyFakeNode"
        assert result.resolved_key is None
        assert result.confidence_adjustment == CONF_UNKNOWN


# ------------------------------------------------------------------
# Connection validation
# ------------------------------------------------------------------

class TestConnectionValidation:
    def test_known_pattern(self, validator):
        result = validator.validate_connection("scatter", "copytopoints", 0, 0)
        assert result.known_pattern is True
        assert result.pattern_count == 12
        assert result.confidence_adjustment == CONF_KNOWN_PATTERN

    def test_unknown_pattern(self, validator):
        result = validator.validate_connection("scatter", "aanoise", 0, 0)
        assert result.known_pattern is False
        assert result.pattern_count == 0
        assert result.confidence_adjustment == CONF_UNKNOWN_PATTERN

    def test_no_patterns_corpus(self, schema_only_validator):
        """When no patterns corpus is loaded, returns neutral result."""
        result = schema_only_validator.validate_connection("scatter", "box")
        assert result.known_pattern is False
        assert result.confidence_adjustment == 0.0

    def test_pattern_count(self, validator):
        assert validator.pattern_count == 3


# ------------------------------------------------------------------
# types_match
# ------------------------------------------------------------------

class TestTypesMatch:
    def test_same_type(self, validator):
        assert validator.types_match("scatter", "scatter") is True

    def test_same_canonical_via_different_paths(self, validator):
        """'sop/scatter' and 'scatter' should match via same resolved key."""
        assert validator.types_match("sop/scatter", "scatter") is True

    def test_label_matches_type(self, validator):
        """'RBD Packed Object' label should match 'rbdpackedobject'."""
        assert validator.types_match("RBD Packed Object", "rbdpackedobject") is True

    def test_different_types_dont_match(self, validator):
        assert validator.types_match("scatter", "aanoise") is False

    def test_unknown_types_match_by_name(self, validator):
        """Two unknown types with same string should match."""
        assert validator.types_match("fakeNode", "fakeNode") is True

    def test_unknown_types_different_names(self, validator):
        assert validator.types_match("fakeA", "fakeB") is False


# ------------------------------------------------------------------
# Convenience methods
# ------------------------------------------------------------------

class TestConvenience:
    def test_normalize_type(self, validator):
        assert validator.normalize_type("sop/scatter") == "scatter"
        assert validator.normalize_type("Anti-Aliased Noise") == "aanoise"
        assert validator.normalize_type("unknownThing") == "unknownThing"

    def test_is_valid_type(self, validator):
        assert validator.is_valid_type("scatter") is True
        assert validator.is_valid_type("Anti-Aliased Noise") is True
        assert validator.is_valid_type("unknownThing") is False


# ------------------------------------------------------------------
# Backward compatibility: validator=None
# ------------------------------------------------------------------

class TestBackwardCompat:
    def test_none_validator_returns_neutral(self):
        """When no corpora loaded, all results are neutral."""
        v = StructuralValidator()
        result = v.validate_node_type("scatter")
        assert result.status == ValidationStatus.UNKNOWN
        assert result.confidence_adjustment == 0.0
        assert result.resolved_type == "scatter"

    def test_none_connection(self):
        v = StructuralValidator()
        result = v.validate_connection("scatter", "box")
        assert result.known_pattern is False
        assert result.confidence_adjustment == 0.0

    def test_none_types_match(self):
        v = StructuralValidator()
        assert v.types_match("scatter", "scatter") is True
        assert v.types_match("scatter", "box") is False
