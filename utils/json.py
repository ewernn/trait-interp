"""
Compact JSON serialization with readable structure but single-line arrays.

Usage:
    from utils.json import dump_compact

    with open(path, 'w') as f:
        dump_compact(data, f)
"""

import json
import re


def dump_compact(obj, f):
    """
    Write JSON with indented structure but single-line primitive arrays.

    Objects get normal 2-space indentation. Arrays of primitives (strings,
    numbers, bools, null) collapse to single lines.
    """
    s = json.dumps(obj, indent=2)
    # Collapse arrays that contain only primitives (no nested objects/arrays)
    # Match [...] where contents have no { or [
    s = re.sub(
        r'\[\s*\n\s*([^\[\]{}]*?)\s*\n\s*\]',
        lambda m: '[' + ', '.join(x.strip() for x in m.group(1).split(',')) + ']',
        s,
        flags=re.DOTALL
    )
    f.write(s)


def dumps_compact(obj) -> str:
    """Return compact JSON string."""
    s = json.dumps(obj, indent=2)
    s = re.sub(
        r'\[\s*\n\s*([^\[\]{}]*?)\s*\n\s*\]',
        lambda m: '[' + ', '.join(x.strip() for x in m.group(1).split(',')) + ']',
        s,
        flags=re.DOTALL
    )
    return s
