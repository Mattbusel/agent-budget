"""``python -m agent_budget audit calls.jsonl --cap 5``: see :mod:`agent_budget.audit`."""

import sys

from .audit import main

if __name__ == "__main__":
    sys.exit(main())
