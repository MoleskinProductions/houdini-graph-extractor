"""Configuration management for Houdini Graph Extractor."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExtractionConfig:
    """Frame extraction settings."""

    frame_buffer_seconds: float = 2.0
    frame_interval_seconds: float = 0.5
    min_graph_area_percent: int = 30
    min_readability: str = "medium"


@dataclass
class VLMConfig:
    """Vision-language model configuration."""

    api_base: str = "https://openrouter.ai/api/v1"  # OpenRouter
    model_name: str = "qwen/qwen2.5-vl-72b-instruct"  # Vision model
    api_key: str = ""  # Set via OPENROUTER_API_KEY env var or --api-key
    max_tokens: int = 4096
    temperature: float = 0.1


@dataclass
class OutputConfig:
    """Output settings."""

    format: str = "hou_data"  # "hou_data", "script", or "both"
    include_metadata: bool = True
    include_timeline: bool = True


@dataclass
class TranscriptAnalysisConfig:
    """Transcript analysis settings."""

    use_llm: bool = True  # Use LLM-based analysis (vs regex fallback)
    batch_size: int = 10  # Segments per LLM API call
    fallback_on_error: bool = True  # Fall back to regex on LLM failure


@dataclass
class HarmonizationConfig:
    """Harmonization settings."""

    enabled: bool = True  # Enable transcript/visual harmonization
    transcript_window_before: float = 3.0  # Seconds before extraction
    transcript_window_after: float = 1.0  # Seconds after extraction
    prefer_visual_on_mismatch: bool = True  # Trust visual over transcript on conflict
    accept_extra_visual_nodes: bool = True  # Accept nodes not in transcript
    extra_visual_penalty: float = 0.15  # Confidence penalty for extra nodes


@dataclass
class StateConfig:
    """State management settings."""

    confidence_decay_rate: float = 0.05  # Decay per frame not seen
    confidence_decay_after_frames: int = 3  # Frames before decay starts
    high_confidence_threshold: float = 0.8
    low_confidence_threshold: float = 0.3
    transcript_boost: float = 0.15  # Boost when transcript corroborates


@dataclass
class NodeSchemaConfig:
    """Node type schema extraction settings."""

    hython_path: Path = field(default_factory=lambda: Path("/opt/hfs21.0/bin/hython"))
    categories: list[str] = field(default_factory=list)  # Empty = all
    timeout: int = 120
    extract_ports: bool = True


@dataclass
class HelpDocsConfig:
    """Help documentation parsing settings."""

    zip_path: Path = field(default_factory=lambda: Path("/opt/hfs21.0/houdini/help/nodes.zip"))
    contexts: list[str] = field(default_factory=lambda: [
        "apex", "chop", "cop", "cop2", "dop", "lop", "obj", "out",
        "shop", "sop", "top", "vop",
    ])


@dataclass
class LabsHDAConfig:
    """Labs HDA internal graph extraction settings."""

    hython_path: Path = field(default_factory=lambda: Path("/opt/hfs21.0/bin/hython"))
    categories: list[str] = field(default_factory=list)  # Empty = all
    library_filter: str = "SideFXLabs"
    timeout: int = 300


@dataclass
class PatternMiningConfig:
    """Pattern mining settings (Phase 2A)."""

    min_pattern_count: int = 1
    max_chain_length: int = 3
    exclude_types: list[str] = field(default_factory=lambda: ["output"])


@dataclass
class IntentMappingConfig:
    """Intent mapping settings (Phase 2B)."""

    exclude_types: list[str] = field(default_factory=lambda: ["output"])
    min_node_count: int = 1


@dataclass
class Config:
    """Main configuration container."""

    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    transcript_analysis: TranscriptAnalysisConfig = field(default_factory=TranscriptAnalysisConfig)
    harmonization: HarmonizationConfig = field(default_factory=HarmonizationConfig)
    state: StateConfig = field(default_factory=StateConfig)
    help_docs: HelpDocsConfig = field(default_factory=HelpDocsConfig)
    node_schema: NodeSchemaConfig = field(default_factory=NodeSchemaConfig)
    labs_hda: LabsHDAConfig = field(default_factory=LabsHDAConfig)
    pattern_mining: PatternMiningConfig = field(default_factory=PatternMiningConfig)
    intent_mapping: IntentMappingConfig = field(default_factory=IntentMappingConfig)

    # Paths
    temp_dir: Path = field(default_factory=lambda: Path("/tmp/houdini-extractor"))

    def __post_init__(self):
        self.temp_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
