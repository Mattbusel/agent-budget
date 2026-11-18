"""Skip the optional integration modules' doctest collection when their
framework is not installed (the core package has no dependencies)."""

import importlib.util

collect_ignore = []
if importlib.util.find_spec("a2a") is None:
    collect_ignore.append("src/agent_budget/a2a_plugin.py")
if importlib.util.find_spec("strands") is None:
    collect_ignore.append("src/agent_budget/strands.py")
