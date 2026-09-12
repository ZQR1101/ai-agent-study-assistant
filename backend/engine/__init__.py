"""Review engine: playbook-driven document review pipeline.

Public entry: :func:`backend.engine.orchestrator.run_review`.
"""

from backend.engine.citation_gate import validate_citations
from backend.engine.scorecard import build_scorecard

__all__ = ["validate_citations", "build_scorecard"]
