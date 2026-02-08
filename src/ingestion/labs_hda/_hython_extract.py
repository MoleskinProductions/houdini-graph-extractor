#!/usr/bin/env python
"""Standalone Labs HDA internal graph extractor — runs inside hython.

This script is executed by hython (Houdini's Python interpreter) and
cannot import from this repo. It communicates via JSON output file.

Usage:
    hython _hython_extract.py <output_path> [--categories Sop,Dop] [--library-filter SideFXLabs]
"""

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone

# --- Houdini imports (only available inside hython) ---
try:
    import hou
except ImportError:
    print("ERROR: This script must be run inside hython.", file=sys.stderr)
    sys.exit(1)


CATEGORY_TO_CONTEXT = {
    "Sop": "sop",
    "Dop": "dop",
    "Vop": "vop",
    "Object": "obj",
    "Driver": "out",
    "Cop2": "cop2",
    "Chop": "chop",
    "Shop": "shop",
    "Top": "top",
    "Lop": "lop",
}

# Maps category name to (parent_path, container_type) for creating temp parents.
# container_type=None means we use the parent_path directly.
TEMP_PARENT_MAP = {
    "Sop": ("/obj", "geo"),
    "Dop": ("/obj", "dopnet"),
    "Vop": (None, None),  # Special: needs geo -> attribvop
    "Object": ("/obj", None),
    "Driver": ("/out", None),
    "Cop2": ("/img", "img"),
    "Chop": ("/obj", "chopnet"),
    "Shop": ("/shop", None),
    "Top": ("/obj", "topnet"),
    "Lop": ("/stage", None),
}


def create_temp_parent(category_name):
    """Create a temporary parent node suitable for instantiating nodes of the given category.

    Returns (parent_node, cleanup_nodes) where cleanup_nodes is a list of nodes
    to destroy when done.
    """
    if category_name == "Vop":
        geo = hou.node("/obj").createNode("geo", "_labs_hda_vop_tmp")
        avop = geo.createNode("attribvop")
        return avop, [geo]

    mapping = TEMP_PARENT_MAP.get(category_name)
    if mapping is None:
        return None, []

    parent_path, container_type = mapping
    parent = hou.node(parent_path)
    if parent is None:
        return None, []

    if container_type is None:
        return parent, []

    container = parent.createNode(container_type, "_labs_hda_tmp")
    return container, [container]


def find_labs_hdas(category, library_filter):
    """Find all HDA types in a category whose library path matches the filter.

    Returns list of (type_name, node_type) tuples.
    """
    results = []
    for type_name, node_type in sorted(category.nodeTypes().items()):
        try:
            defn = node_type.definition()
            if defn is None:
                continue
            lib_path = defn.libraryFilePath()
            if library_filter in lib_path:
                results.append((type_name, node_type))
        except Exception:
            continue
    return results



# Parameters whose string values should never be captured (environment dumps, secrets).
_SCRUB_PARAM_NAMES = {"customenv"}

# Substrings that indicate a string value contains secrets and should be redacted.
_SECRET_MARKERS = (
    "API_KEY",
    "API_TOKEN",
    "SECRET_KEY",
    "ACCESS_TOKEN",
    "HF_TOKEN",
    "ANTHROPIC_",
    "OPENROUTER_",
    "REPLICATE_",
    "sk-ant-",
    "sk-or-v1",
)


def _should_scrub_value(param_name, value):
    """Return True if a parameter value should be redacted."""
    if param_name in _SCRUB_PARAM_NAMES:
        return True
    if isinstance(value, str) and any(marker in value for marker in _SECRET_MARKERS):
        return True
    return False


def extract_node_params(node):
    """Extract non-default parameters from a node instance."""
    params = []
    for p in node.parms():
        try:
            if p.isAtDefault():
                continue
        except Exception:
            continue

        param_data = {"name": p.name()}

        # Get evaluated value
        try:
            val = p.eval()
            # Only keep JSON-serializable types
            if isinstance(val, (int, float, bool)):
                param_data["value"] = val
            elif isinstance(val, str):
                if _should_scrub_value(p.name(), val):
                    continue
                param_data["value"] = val
            else:
                # hou.Ramp, hou.Geometry, etc. — skip
                continue
        except Exception:
            continue

        # Check for expressions
        try:
            expr = p.expression()
            if expr:
                param_data["expression"] = expr
                try:
                    lang = p.expressionLanguage()
                    param_data["expression_language"] = (
                        "python" if lang == hou.exprLanguage.Python else "hscript"
                    )
                except Exception:
                    param_data["expression_language"] = "hscript"
        except hou.OperationFailed:
            # No expression set
            pass
        except Exception:
            pass

        params.append(param_data)
    return params


def extract_hda_graph(parent_node, type_name, node_type):
    """Extract the internal graph of a single Labs HDA.

    Returns a dict with the graph data, or None on failure.
    """
    inst = None
    try:
        inst = parent_node.createNode(type_name, "_labs_probe")
    except Exception:
        return None

    try:
        # Unlock HDA to expose children
        inst.allowEditingOfContents()

        graph = {
            "type_name": type_name,
            "category": node_type.category().name(),
        }

        # Label
        try:
            graph["label"] = node_type.description()
        except Exception:
            graph["label"] = ""

        # Library path
        try:
            defn = node_type.definition()
            if defn:
                graph["library_path"] = defn.libraryFilePath()
        except Exception:
            pass

        # Input/output counts
        try:
            graph["min_inputs"] = node_type.minNumInputs()
        except Exception:
            graph["min_inputs"] = 0
        try:
            graph["max_inputs"] = node_type.maxNumInputs()
        except Exception:
            graph["max_inputs"] = 0
        try:
            graph["max_outputs"] = node_type.maxNumOutputs()
        except Exception:
            graph["max_outputs"] = 0

        # Subnet inputs (HDA interface inputs)
        subnet_inputs = []
        try:
            indirect_inputs = inst.indirectInputs()
            for i, indirect in enumerate(indirect_inputs):
                si = {
                    "index": i,
                    "name": indirect.name() if hasattr(indirect, "name") else f"input{i}",
                    "connections": [],
                }
                subnet_inputs.append(si)
        except Exception:
            pass

        # Extract children
        nodes = []
        connections = []
        children = inst.children()

        for child in children:
            try:
                node_data = {
                    "name": child.name(),
                    "type": child.type().name(),
                }

                # Position
                try:
                    pos = child.position()
                    node_data["position"] = [pos[0], pos[1]]
                except Exception:
                    node_data["position"] = [0.0, 0.0]

                # Flags
                try:
                    if child.isDisplayFlagSet():
                        node_data["display_flag"] = True
                except Exception:
                    pass
                try:
                    if child.isRenderFlagSet():
                        node_data["render_flag"] = True
                except Exception:
                    pass
                try:
                    if child.isBypassed():
                        node_data["bypass_flag"] = True
                except Exception:
                    pass

                # Check if this is an output node
                try:
                    if child.type().name() == "output":
                        node_data["is_output"] = True
                except Exception:
                    pass

                # Non-default parameters
                params = extract_node_params(child)
                if params:
                    node_data["parameters"] = params

                nodes.append(node_data)

                # Extract input connections for this child
                try:
                    for conn in child.inputConnections():
                        src_node = conn.inputNode()
                        if src_node is None:
                            continue

                        # Check if this connection comes from a subnet input
                        try:
                            subnet_input = conn.subnetIndirectInput()
                            if subnet_input is not None:
                                # Find which subnet input index this is
                                for si in subnet_inputs:
                                    try:
                                        indirect_inputs = inst.indirectInputs()
                                        if si["index"] < len(indirect_inputs):
                                            if subnet_input == indirect_inputs[si["index"]]:
                                                si["connections"].append({
                                                    "dest_node": child.name(),
                                                    "dest_input": conn.inputIndex(),
                                                })
                                                break
                                    except Exception:
                                        pass
                                continue
                        except Exception:
                            pass

                        # Regular node-to-node connection
                        # Note: hou connection naming is inverted:
                        #   conn.inputNode() = source node
                        #   conn.outputNode() = dest node
                        #   conn.outputIndex() = source output port
                        #   conn.inputIndex() = dest input port
                        connections.append({
                            "source_node": src_node.name(),
                            "source_output": conn.outputIndex(),
                            "dest_node": child.name(),
                            "dest_input": conn.inputIndex(),
                        })

                except Exception:
                    pass

            except Exception:
                continue

        graph["subnet_inputs"] = [si for si in subnet_inputs if si["connections"]]
        graph["nodes"] = nodes
        graph["connections"] = connections

        return graph

    except Exception:
        return None
    finally:
        if inst is not None:
            try:
                inst.destroy()
            except Exception:
                pass


def extract_category(category_name, category, library_filter):
    """Extract all Labs HDA graphs for a given category."""
    results = []
    errors = []

    labs_hdas = find_labs_hdas(category, library_filter)
    if not labs_hdas:
        return results, errors

    # Check if we can create temp parent for this category
    if category_name not in TEMP_PARENT_MAP:
        errors.append(
            f"Skipping category {category_name}: not in TEMP_PARENT_MAP"
        )
        return results, errors

    parent_node, cleanup_nodes = create_temp_parent(category_name)
    if parent_node is None:
        errors.append(f"Could not create temp parent for category {category_name}")
        return results, errors

    try:
        for type_name, node_type in labs_hdas:
            try:
                graph = extract_hda_graph(parent_node, type_name, node_type)
                if graph is not None:
                    results.append(graph)
                else:
                    errors.append(f"{category_name}/{type_name}: failed to extract")
            except Exception as e:
                errors.append(f"{category_name}/{type_name}: {e}")
    finally:
        for node in cleanup_nodes:
            try:
                node.destroy()
            except Exception:
                pass

    return results, errors


def main():
    parser = argparse.ArgumentParser(description="Extract Labs HDA internal graphs")
    parser.add_argument("output", help="Output JSON file path")
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated categories to extract (e.g. Sop,Dop). Default: all.",
    )
    parser.add_argument(
        "--library-filter",
        type=str,
        default="SideFXLabs",
        help="Filter HDAs by library file path containing this string (default: SideFXLabs)",
    )
    args = parser.parse_args()

    # Determine which categories to extract
    all_categories = hou.nodeTypeCategories()

    if args.categories:
        requested = [c.strip() for c in args.categories.split(",")]
        categories = {}
        for name in requested:
            if name in all_categories:
                categories[name] = all_categories[name]
            else:
                print(f"WARNING: Unknown category '{name}', skipping", file=sys.stderr)
    else:
        categories = {
            name: cat
            for name, cat in all_categories.items()
            if name in CATEGORY_TO_CONTEXT
        }

    # Houdini version
    houdini_version = f"{hou.applicationVersionString()}"

    # Extract
    all_graphs = []
    all_errors = []

    for cat_name in sorted(categories.keys()):
        # Skip categories we don't have a temp parent for
        if cat_name not in TEMP_PARENT_MAP:
            print(
                f"WARNING: Skipping category '{cat_name}': no temp parent mapping",
                file=sys.stderr,
            )
            continue

        cat = categories[cat_name]
        print(f"Scanning {cat_name} for Labs HDAs...", file=sys.stderr)

        graphs, errors = extract_category(cat_name, cat, args.library_filter)
        all_graphs.extend(graphs)
        all_errors.extend(errors)

        print(
            f"  {len(graphs)} HDAs extracted, {len(errors)} errors",
            file=sys.stderr,
        )

    # Build output
    output = {
        "houdini_version": houdini_version,
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "hda_count": len(all_graphs),
        "categories_extracted": sorted(categories.keys()),
        "library_filter": args.library_filter,
        "graphs": all_graphs,
    }
    if all_errors:
        output["errors"] = all_errors

    # Write output
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"Done: {len(all_graphs)} Labs HDAs written to {args.output}",
        file=sys.stderr,
    )
    if all_errors:
        print(
            f"Warnings: {len(all_errors)} errors (see 'errors' key in output)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
