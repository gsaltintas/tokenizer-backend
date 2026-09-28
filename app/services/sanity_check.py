"""
Single-tokenizer health report via tokenizer-intrinsic-evals' sanity check.

Runs the same 16 checks as the library's `tokenizer-sanity-check` CLI on the
built-in, offline probe corpus, and reports the exit code the CLI would have
returned (0 pass, 1 warn, 2 fail) so the UI can show whether it would gate.
"""

import logging
import math
from functools import lru_cache
from typing import Any

from app.services.adapter import TokenizerAdapter
from app.services.tokeval_wrapper import to_tokeval_wrapper


def sanity_check(adapter: TokenizerAdapter) -> dict[str, Any]:
    return _cached_sanity_check(adapter)


@lru_cache(maxsize=10)
def _cached_sanity_check(adapter: TokenizerAdapter) -> dict[str, Any]:
    from tokenizer_analysis.diagnostics.probe_corpus import builtin_probes
    from tokenizer_analysis.diagnostics.sanity_check import (
        TokenizerSanityChecker,
        severity_to_exit_code,
    )

    wrapper = to_tokeval_wrapper(adapter)
    with _CapturedWarnings() as warnings:
        report = TokenizerSanityChecker(wrapper, builtin_probes(), name=adapter.name).run()

    checks = list(report["checks"].values())
    return _jsonable({
        "overall_severity": report["overall_severity"],
        "exit_code": severity_to_exit_code(report["overall_severity"]),
        "n_fail": sum(c["severity"] == "fail" for c in checks),
        "n_warn": sum(c["severity"] in ("warn", "unverifiable") for c in checks),
        "checks": checks,
        "lossy_breakdown": report["lossy_breakdown"],
        "vocab_reachability": report["vocab_reachability"],
        "vocab_composition": report["vocab_composition"],
        "components": report["components"],
        "warnings": warnings,
    })


class _CapturedWarnings(logging.Handler):
    """Collect library warnings (e.g. the generic special-token fallback) for the report."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())

    def __enter__(self) -> list[str]:
        logging.getLogger("tokenizer_analysis").addHandler(self)
        return self.messages

    def __exit__(self, *exc) -> None:
        logging.getLogger("tokenizer_analysis").removeHandler(self)


def _jsonable(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    if isinstance(o, (set, frozenset)):
        return sorted((_jsonable(x) for x in o), key=repr)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, bytes):
        return o.hex()
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    return repr(o)
