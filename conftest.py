"""Skip the a2a plugin module's doctest collection when a2a-sdk is not
installed (the core package has no dependencies)."""

import importlib.util

collect_ignore = []
if importlib.util.find_spec("a2a") is None:
    collect_ignore.append("src/agent_budget/a2a_plugin.py")
