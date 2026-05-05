"""Framework auto-patchers.

Each module in this subpackage exposes a single ``apply()`` function that
returns ``True`` if it patched something, ``False`` if the framework isn't
installed (or already patched). ``init.py`` calls them in sequence.
"""

from __future__ import annotations

import importlib.util
import sys


def has_module(name: str) -> bool:
    """True if `name` can be imported.

    Falls back to checking ``sys.modules`` directly so test stubs (which may
    not have a ``__spec__``) still register as present.
    """
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False
