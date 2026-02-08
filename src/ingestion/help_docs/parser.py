"""Single-pass, line-by-line state machine parser for Houdini help docs."""

from __future__ import annotations

import re
from enum import Enum, auto

from .models import (
    HelpLocalVar,
    HelpMenuOption,
    HelpParameter,
    HelpParameterGroup,
    HelpPort,
    HelpSection,
    HelpTopAttribute,
    NodeHelpDoc,
)

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Header directives: #key: value  (may have extra whitespace)
_RE_DIRECTIVE = re.compile(r"^#(\w+)\s*:\s*(.+)$")

# Title line: = Title = or = Title<T> =
_RE_TITLE = re.compile(r"^=\s+(.+?)\s+=\s*$")

# Brief description: """..."""
_RE_BRIEF = re.compile(r'^"""(.*)"""$')

# Section header: == Section == or == Section == (anchor)
_RE_SECTION = re.compile(r"^==\s+(.+?)\s+==")

# Parameter group separator: ~~~ Label ~~~ or bare ~~~
_RE_PARAM_GROUP = re.compile(r"^~~~(?:\s+(.+?)\s+~~~)?$")

# Parameter label: "Label:" at indent 0, or 4-space indent with label
# Must end with colon, not be a directive or section marker.
_RE_PARAM_LABEL = re.compile(r"^(\w[^#\n]*?)\s*:\s*$")

# Parameter directive (indented): #id: value, #channels: value
_RE_PARAM_DIRECTIVE = re.compile(r"^\s+#(\w+)\s*:\s*(.+)$")

# Block markers: @parameters, @inputs, @outputs, @related, @locals, @top_attributes
_RE_BLOCK_MARKER = re.compile(r"^@(\w+)\s*$")

# Include directive: :include path:
_RE_INCLUDE = re.compile(r"^:include\s+(.+?):\s*$")

# Related node reference: - [Node:context/name]
_RE_RELATED = re.compile(r"^\s*-\s+\[Node:(.+?)\]")

# APEX/TOP port: ::name: or ::`name`:  (with optional #type: annotation on next line)
_RE_PORT = re.compile(r"^::(?:`?)(\w[\w.]*(?:<\w+>)?)(?:`?):\s*$")

# Local variable: single uppercase name followed by colon
_RE_LOCAL_VAR = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*:\s*$")


class _State(Enum):
    """Parser states."""

    HEADER = auto()
    BODY = auto()
    PARAMETERS = auto()
    INPUTS = auto()
    OUTPUTS = auto()
    RELATED = auto()
    LOCALS = auto()
    TOP_ATTRIBUTES = auto()


class HelpFileParser:
    """Parses a single Houdini help file into a NodeHelpDoc."""

    def parse(self, text: str, filename: str = "") -> NodeHelpDoc | None:
        """Parse a help file's text content.

        Returns None if the file is an include (not a node doc) or has
        no identifiable context/internal_name.
        """
        # Strip BOM
        if text.startswith("\ufeff"):
            text = text[1:]

        doc = NodeHelpDoc()
        state = _State.HEADER

        # Parser sub-state
        current_section: HelpSection | None = None
        current_param: HelpParameter | None = None
        current_param_group: HelpParameterGroup | None = None
        current_port: HelpPort | None = None
        current_local: HelpLocalVar | None = None
        current_top_attr: HelpTopAttribute | None = None
        current_menu_option: HelpMenuOption | None = None
        in_brief = False
        brief_lines: list[str] = []
        is_include_type = False
        param_desc_lines: list[str] = []
        port_desc_lines: list[str] = []
        local_desc_lines: list[str] = []
        top_attr_desc_lines: list[str] = []
        menu_desc_lines: list[str] = []
        section_lines: list[str] = []
        param_indent: int = -1  # indent level of parameter labels

        def _flush_param() -> None:
            nonlocal current_param, current_menu_option, param_desc_lines, menu_desc_lines
            if current_menu_option and current_param:
                current_menu_option.description = "\n".join(menu_desc_lines).strip()
                current_param.menu_options.append(current_menu_option)
                current_menu_option = None
                menu_desc_lines = []
            if current_param:
                current_param.description = "\n".join(param_desc_lines).strip()
                param_desc_lines = []
                if current_param_group is not None:
                    current_param_group.parameters.append(current_param)
                else:
                    doc.parameters.append(current_param)
                current_param = None

        def _flush_port() -> None:
            nonlocal current_port, port_desc_lines
            if current_port:
                current_port.description = "\n".join(port_desc_lines).strip()
                port_desc_lines = []
                if state == _State.INPUTS:
                    doc.inputs.append(current_port)
                else:
                    doc.outputs.append(current_port)
                current_port = None

        def _flush_local() -> None:
            nonlocal current_local, local_desc_lines
            if current_local:
                current_local.description = "\n".join(local_desc_lines).strip()
                local_desc_lines = []
                doc.locals.append(current_local)
                current_local = None

        def _flush_top_attr() -> None:
            nonlocal current_top_attr, top_attr_desc_lines
            if current_top_attr:
                current_top_attr.description = "\n".join(top_attr_desc_lines).strip()
                top_attr_desc_lines = []
                doc.top_attributes.append(current_top_attr)
                current_top_attr = None

        def _flush_section() -> None:
            nonlocal current_section, section_lines
            if current_section:
                current_section.content = "\n".join(section_lines).strip()
                doc.sections.append(current_section)
                current_section = None
                section_lines = []

        lines = text.split("\n")
        i = 0
        while i < len(lines):
            raw_line = lines[i]
            line = raw_line.rstrip()
            stripped = line.strip()
            i += 1

            # --- Multi-line brief handling ---
            if in_brief:
                if '"""' in line:
                    # End of multi-line brief
                    part = line.split('"""')[0]
                    brief_lines.append(part)
                    doc.brief = "\n".join(brief_lines).strip()
                    in_brief = False
                else:
                    brief_lines.append(line)
                continue

            # --- Block marker transitions (@parameters, @inputs, etc.) ---
            m = _RE_BLOCK_MARKER.match(stripped)
            if m:
                block_name = m.group(1)
                new_state = {
                    "parameters": _State.PARAMETERS,
                    "inputs": _State.INPUTS,
                    "outputs": _State.OUTPUTS,
                    "related": _State.RELATED,
                    "locals": _State.LOCALS,
                    "top_attributes": _State.TOP_ATTRIBUTES,
                }.get(block_name)
                if new_state is not None:
                    # Flush previous state
                    if state == _State.PARAMETERS:
                        _flush_param()
                    elif state in (_State.INPUTS, _State.OUTPUTS):
                        _flush_port()
                    elif state == _State.LOCALS:
                        _flush_local()
                    elif state == _State.TOP_ATTRIBUTES:
                        _flush_top_attr()
                    elif state == _State.BODY:
                        _flush_section()
                    state = new_state
                    continue
                # Other @-blocks (like @subtopics) — skip
                if block_name == "subtopics":
                    state = _State.BODY
                    continue

            # --- HEADER state ---
            if state == _State.HEADER:
                # Directive line
                dm = _RE_DIRECTIVE.match(stripped)
                if dm:
                    key, val = dm.group(1), dm.group(2).strip()
                    if key == "type":
                        if val == "include":
                            is_include_type = True
                    elif key == "context":
                        doc.context = val.lower()
                    elif key == "internal":
                        doc.internal_name = val
                    elif key == "namespace":
                        doc.namespace = val
                    elif key == "icon":
                        doc.icon = val
                    elif key == "tags":
                        doc.tags = [t.strip() for t in val.split(",") if t.strip()]
                    elif key == "since":
                        doc.since_version = val
                    elif key == "version":
                        doc.version = val
                    continue

                # Title line
                tm = _RE_TITLE.match(stripped)
                if tm:
                    doc.title = tm.group(1)
                    continue

                # Brief description
                bm = _RE_BRIEF.match(stripped)
                if bm:
                    doc.brief = bm.group(1)
                    state = _State.BODY
                    continue

                # Start of multi-line brief
                if stripped.startswith('"""') and not stripped.endswith('"""'):
                    in_brief = True
                    brief_lines = [stripped[3:]]
                    continue

                # Empty line in header — still header
                if not stripped:
                    continue

                # Non-directive, non-title, non-empty → transition to body
                state = _State.BODY
                # Fall through to body processing

            # --- BODY state ---
            if state == _State.BODY:
                # Section header
                sm = _RE_SECTION.match(stripped)
                if sm:
                    _flush_section()
                    current_section = HelpSection(title=sm.group(1))
                    continue

                # Include directive
                im = _RE_INCLUDE.match(stripped)
                if im:
                    doc.includes.append(im.group(1))
                    continue

                # Accumulate section content
                if current_section is not None:
                    section_lines.append(line)
                continue

            # --- PARAMETERS state ---
            if state == _State.PARAMETERS:
                # Include directive within parameters
                im = _RE_INCLUDE.match(stripped)
                if im:
                    doc.includes.append(im.group(1))
                    continue

                # Section header within parameters (== Section ==)
                sm = _RE_SECTION.match(stripped)
                if sm:
                    _flush_param()
                    param_indent = -1
                    continue

                # Parameter group (~~~ Label ~~~ or bare ~~~)
                gm = _RE_PARAM_GROUP.match(stripped)
                if gm:
                    _flush_param()
                    if current_param_group is not None:
                        doc.parameter_groups.append(current_param_group)
                    label = gm.group(1) or ""
                    if label:
                        current_param_group = HelpParameterGroup(label=label)
                    else:
                        current_param_group = None
                    # Reset param indent for new group context
                    param_indent = -1
                    continue

                # Parameter directive (#id:, #channels:)
                pdm = _RE_PARAM_DIRECTIVE.match(line)
                if pdm and current_param is not None:
                    key, val = pdm.group(1), pdm.group(2).strip()
                    if key == "id":
                        current_param.id = val
                    elif key == "channels":
                        current_param.channels = val
                    continue

                # Check for Label: pattern
                plm = _RE_PARAM_LABEL.match(stripped)
                if plm and stripped:
                    line_indent = len(line) - len(line.lstrip())

                    # First label we see establishes the param indent level
                    if param_indent < 0:
                        param_indent = line_indent

                    if line_indent <= param_indent:
                        # Same or lesser indent → new parameter
                        _flush_param()
                        current_param = HelpParameter(label=plm.group(1))
                        param_indent = line_indent
                        continue
                    elif current_param is not None:
                        # Deeper indent → menu option
                        if current_menu_option:
                            current_menu_option.description = "\n".join(menu_desc_lines).strip()
                            current_param.menu_options.append(current_menu_option)
                            menu_desc_lines = []
                        current_menu_option = HelpMenuOption(label=plm.group(1))
                        continue

                # If we're in a menu option, accumulate its description
                if current_menu_option is not None:
                    if stripped:
                        menu_desc_lines.append(stripped)
                    continue

                # Description text for current param (indented)
                if current_param is not None and stripped:
                    param_desc_lines.append(stripped)
                continue

            # --- INPUTS / OUTPUTS state ---
            if state in (_State.INPUTS, _State.OUTPUTS):
                # Port definition (::name: or ::`name`:)
                pm = _RE_PORT.match(stripped)
                if pm:
                    _flush_port()
                    current_port = HelpPort(name=pm.group(1))
                    continue

                # Port type annotation (#type: ...)
                pdm = _RE_PARAM_DIRECTIVE.match(line)
                if pdm and current_port is not None:
                    key, val = pdm.group(1), pdm.group(2).strip()
                    if key == "type":
                        current_port.type = val
                    continue

                # Description text for current port
                if current_port is not None and stripped:
                    port_desc_lines.append(stripped)
                continue

            # --- RELATED state ---
            if state == _State.RELATED:
                rm = _RE_RELATED.match(line)
                if rm:
                    doc.related.append(rm.group(1))
                continue

            # --- LOCALS state ---
            if state == _State.LOCALS:
                lm = _RE_LOCAL_VAR.match(stripped)
                if lm:
                    _flush_local()
                    current_local = HelpLocalVar(name=lm.group(1))
                    continue

                if current_local is not None and stripped:
                    local_desc_lines.append(stripped)
                continue

            # --- TOP_ATTRIBUTES state ---
            if state == _State.TOP_ATTRIBUTES:
                # Same port-like syntax as inputs/outputs
                pm = _RE_PORT.match(stripped)
                if pm:
                    _flush_top_attr()
                    current_top_attr = HelpTopAttribute(name=pm.group(1))
                    continue

                pdm = _RE_PARAM_DIRECTIVE.match(line)
                if pdm and current_top_attr is not None:
                    key, val = pdm.group(1), pdm.group(2).strip()
                    if key == "type":
                        current_top_attr.type = val
                    continue

                if current_top_attr is not None and stripped:
                    top_attr_desc_lines.append(stripped)
                continue

        # --- End of file: flush remaining state ---
        if state == _State.PARAMETERS:
            _flush_param()
            if current_param_group is not None:
                doc.parameter_groups.append(current_param_group)
        elif state in (_State.INPUTS, _State.OUTPUTS):
            _flush_port()
        elif state == _State.LOCALS:
            _flush_local()
        elif state == _State.TOP_ATTRIBUTES:
            _flush_top_attr()
        elif state == _State.BODY:
            _flush_section()

        # Skip include-type files
        if is_include_type:
            return None

        # Derive context/internal_name from filename if not in directives
        if filename and (not doc.context or not doc.internal_name):
            # Filename format: "context/name.txt" or "context/namespace--name.txt"
            parts = filename.replace("\\", "/").split("/")
            if len(parts) >= 2:
                if not doc.context:
                    doc.context = parts[-2]
                if not doc.internal_name:
                    doc.internal_name = parts[-1].removesuffix(".txt")

        # Must have both context and internal_name to be valid
        if not doc.context or not doc.internal_name:
            return None

        return doc
