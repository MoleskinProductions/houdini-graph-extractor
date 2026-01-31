"""Output formatters for hou.data, HouGraph IR, and Python scripts."""

from .hou_data import HouDataFormatter

# HouGraph IR exporter - only available if hougraph-ir is installed
try:
    from .hougraph_ir_export import (
        HouGraphIRExporter,
        export_extractions_to_hougraph_ir,
        export_extractions_to_dict,
        HOUGRAPH_IR_AVAILABLE,
    )
except ImportError:
    HouGraphIRExporter = None
    export_extractions_to_hougraph_ir = None
    export_extractions_to_dict = None
    HOUGRAPH_IR_AVAILABLE = False

__all__ = [
    "HouDataFormatter",
    "HouGraphIRExporter",
    "export_extractions_to_hougraph_ir",
    "export_extractions_to_dict",
    "HOUGRAPH_IR_AVAILABLE",
]
