#!/usr/bin/env python
"""Standalone node type schema extractor — runs inside hython.

This script is executed by hython (Houdini's Python interpreter) and
cannot import from this repo. It communicates via JSON output file.

Usage:
    hython _hython_extract.py <output_path> [--categories Sop,Dop] [--no-ports]
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
        geo = hou.node("/obj").createNode("geo", "_schema_vop_tmp")
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

    container = parent.createNode(container_type, "_schema_tmp")
    return container, [container]


def extract_parm_template(pt):
    """Extract schema dict from a single hou.ParmTemplate."""
    d = {
        "name": pt.name(),
        "label": pt.label(),
        "type": pt.type().name(),
        "size": pt.numComponents(),
    }

    # Default value
    try:
        default = pt.defaultValue()
        if default is not None:
            if isinstance(default, tuple):
                default = list(default)
            d["default"] = default
    except Exception:
        pass

    # Min/max
    try:
        min_val = pt.minValue()
        if min_val is not None:
            d["min_value"] = min_val
        max_val = pt.maxValue()
        if max_val is not None:
            d["max_value"] = max_val
        if pt.minIsStrict():
            d["min_is_strict"] = True
        if pt.maxIsStrict():
            d["max_is_strict"] = True
    except Exception:
        pass

    # Menu items
    try:
        items = pt.menuItems()
        if items:
            d["menu_items"] = list(items)
            labels = pt.menuLabels()
            if labels:
                d["menu_labels"] = list(labels)
    except Exception:
        pass

    # Visibility / help / tags
    try:
        if pt.isHidden():
            d["is_hidden"] = True
    except Exception:
        pass
    try:
        help_str = pt.help()
        if help_str:
            d["help"] = help_str
    except Exception:
        pass
    try:
        tags = pt.tags()
        if tags:
            d["tags"] = dict(tags)
    except Exception:
        pass

    # Conditionals
    try:
        conditionals = pt.conditionals()
        if conditionals:
            for cond_type, cond_str in conditionals.items():
                cond_name = cond_type.name()
                if cond_name == "DisableWhen":
                    d["disable_when"] = cond_str
                elif cond_name == "HideWhen":
                    d["hide_when"] = cond_str
    except Exception:
        pass

    return d


def flatten_parm_templates(group):
    """Recursively flatten a parm template group, skipping folder containers."""
    result = []
    for pt in group:
        if pt.type() == hou.parmTemplateType.Folder:
            result.extend(flatten_parm_templates(pt.parmTemplates()))
        elif pt.type() == hou.parmTemplateType.FolderSet:
            result.extend(flatten_parm_templates(pt.parmTemplates()))
        else:
            result.append(extract_parm_template(pt))
    return result


def extract_ports(node):
    """Extract input/output port schemas from a node instance."""
    inputs = []
    try:
        names = node.inputNames()
        labels = node.inputLabels()
        for i, name in enumerate(names):
            inputs.append({
                "index": i,
                "name": name,
                "label": labels[i] if i < len(labels) else "",
            })
    except Exception:
        pass

    outputs = []
    try:
        names = node.outputNames()
        labels = node.outputLabels()
        for i, name in enumerate(names):
            outputs.append({
                "index": i,
                "name": name,
                "label": labels[i] if i < len(labels) else "",
            })
    except Exception:
        pass

    return inputs, outputs


def extract_category(category_name, category, extract_ports_flag=True):
    """Extract all node type schemas for a given category."""
    results = []
    errors = []

    parent_node, cleanup_nodes = create_temp_parent(category_name)
    if parent_node is None and category_name not in ("Object", "Driver", "Shop", "Lop"):
        errors.append(f"Could not create temp parent for category {category_name}")
        return results, errors

    node_types = category.nodeTypes()

    for type_name, node_type in sorted(node_types.items()):
        try:
            schema = {
                "category": category_name,
                "type_name": type_name,
            }

            # Label / description
            try:
                schema["label"] = node_type.description()
            except Exception:
                schema["label"] = ""

            # Name components
            try:
                comps = node_type.nameComponents()
                if comps[0]:
                    schema["scope_namespace"] = comps[0]
                if comps[1]:
                    schema["namespace"] = comps[1]
                if comps[2]:
                    schema["base_type"] = comps[2]
                if comps[3]:
                    schema["version"] = comps[3]
            except Exception:
                pass

            # Icon
            try:
                icon = node_type.icon()
                if icon:
                    schema["icon"] = icon
            except Exception:
                pass

            # Input/output counts (from type definition, no instance needed)
            try:
                schema["min_inputs"] = node_type.minNumInputs()
            except Exception:
                schema["min_inputs"] = 0
            try:
                schema["max_inputs"] = node_type.maxNumInputs()
            except Exception:
                schema["max_inputs"] = 0
            try:
                schema["max_outputs"] = node_type.maxNumOutputs()
            except Exception:
                schema["max_outputs"] = 0

            # Flags
            try:
                if node_type.isGenerator():
                    schema["is_generator"] = True
            except Exception:
                pass
            try:
                if node_type.hasUnorderedInputs():
                    schema["unordered_inputs"] = True
            except Exception:
                pass
            try:
                if node_type.deprecated():
                    schema["deprecated"] = True
            except Exception:
                pass
            try:
                defn = node_type.definition()
                if defn is not None:
                    schema["is_hda"] = True
            except Exception:
                pass

            # Parameters (from parm template group — no instance needed)
            try:
                ptg = node_type.parmTemplateGroup()
                schema["parameters"] = flatten_parm_templates(ptg.entries())
            except Exception:
                schema["parameters"] = []

            # Ports (requires creating a temporary instance)
            if extract_ports_flag and parent_node is not None:
                try:
                    tmp_node = parent_node.createNode(type_name, "_port_probe")
                    inputs, outputs = extract_ports(tmp_node)
                    if inputs:
                        schema["inputs"] = inputs
                    if outputs:
                        schema["outputs"] = outputs
                    tmp_node.destroy()
                except Exception:
                    # Many node types can't be instantiated — that's fine
                    pass

            results.append(schema)

        except Exception as e:
            errors.append(f"{category_name}/{type_name}: {e}")

    # Clean up temp nodes
    for node in cleanup_nodes:
        try:
            node.destroy()
        except Exception:
            pass

    return results, errors


def main():
    parser = argparse.ArgumentParser(description="Extract Houdini node type schemas")
    parser.add_argument("output", help="Output JSON file path")
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Comma-separated categories to extract (e.g. Sop,Dop,Vop). Default: all.",
    )
    parser.add_argument(
        "--no-ports",
        action="store_true",
        help="Skip port extraction (faster, no node instantiation needed)",
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
    all_nodes = []
    all_errors = []

    for cat_name in sorted(categories.keys()):
        cat = categories[cat_name]
        print(f"Extracting {cat_name}...", file=sys.stderr)
        nodes, errors = extract_category(cat_name, cat, extract_ports_flag=not args.no_ports)
        all_nodes.extend(nodes)
        all_errors.extend(errors)
        print(f"  {len(nodes)} types extracted, {len(errors)} errors", file=sys.stderr)

    # Build output
    output = {
        "houdini_version": houdini_version,
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "node_count": len(all_nodes),
        "categories_extracted": sorted(categories.keys()),
        "nodes": all_nodes,
    }
    if all_errors:
        output["errors"] = all_errors

    # Write output
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Done: {len(all_nodes)} node types written to {args.output}", file=sys.stderr)
    if all_errors:
        print(f"Warnings: {len(all_errors)} errors (see 'errors' key in output)", file=sys.stderr)


if __name__ == "__main__":
    main()
