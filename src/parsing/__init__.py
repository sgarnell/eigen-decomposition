"""Phase 01 -- Graphviz ``.gv`` parsing.

Public surface (imported lazily so that
``python -m src.parsing.parse_graphviz`` does not execute the module twice):

* :func:`parse_graphviz_file` -- parse a file into a :class:`ParsedGraph`
  (raises :class:`GraphvizValidationError` on any validation error).
* :func:`build_parsed_graph` -- parse and return ``(ParsedGraph, ValidationReport)``
  without raising, for callers that want to inspect warnings.
* :func:`write_artifact` / :func:`artifact_path` -- persist the canonical
  ``parsed_graph.json`` artifact.
* :func:`main` -- the command line interface.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import-time only for type checkers
    from src.parsing.parse_graphviz import (
        DEFAULT_ARTIFACT_NAME,
        FILTERED_ARTIFACT_NAME,
        GraphvizValidationError,
        Neuron,
        NeuronPair,
        ParsedGraph,
        RawEdge,
        ValidationIssue,
        ValidationReport,
        apply_filters,
        artifact_path,
        build_argument_parser,
        build_parsed_graph,
        main,
        parse_graphviz_file,
        validate_payload_schema,
        write_artifact,
    )

_MODULE_NAME = "src.parsing.parse_graphviz"

__all__ = [
    "DEFAULT_ARTIFACT_NAME",
    "FILTERED_ARTIFACT_NAME",
    "GraphvizValidationError",
    "Neuron",
    "NeuronPair",
    "ParsedGraph",
    "RawEdge",
    "ValidationIssue",
    "ValidationReport",
    "apply_filters",
    "artifact_path",
    "build_argument_parser",
    "build_parsed_graph",
    "main",
    "parse_graphviz_file",
    "validate_payload_schema",
    "write_artifact",
]


def __getattr__(name: str) -> Any:
    """Resolve the public names from :mod:`src.parsing.parse_graphviz` on demand."""
    if name in __all__:
        module = importlib.import_module(_MODULE_NAME)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)