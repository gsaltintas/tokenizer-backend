"""Per-tokenizer memoization of whole-vocabulary analyses.

Results are keyed on the adapter object rather than the tokenizer ID, so they
are dropped when the registry evicts or reloads that tokenizer.
"""

import weakref
from typing import Callable, TypeVar

T = TypeVar("T")

_results: "weakref.WeakKeyDictionary[object, dict[str, object]]" = weakref.WeakKeyDictionary()


def memo(adapter: object, key: str, compute: Callable[[], T]) -> T:
    """Return compute() for this adapter, computing it on first use only."""
    entries = _results.setdefault(adapter, {})
    if key not in entries:
        entries[key] = compute()
    return entries[key]  # type: ignore[return-value]
