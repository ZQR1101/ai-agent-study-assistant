"""Playbook registry: the only place new business verticals get registered.

Adding a vertical = append a PlaybookSpec to ``_PLAYBOOKS`` (or register at
runtime in tests via :func:`register_playbook`). The engine reads everything
else from the database.
"""

from __future__ import annotations

from backend.playbooks.base import PlaybookSpec
from backend.playbooks.contract_compliance import CONTRACT_COMPLIANCE_PLAYBOOK
from backend.playbooks.dpa_review import DPA_REVIEW_PLAYBOOK
from backend.playbooks.delivery_intake import DELIVERY_INTAKE_PLAYBOOK

_PLAYBOOKS: dict[str, PlaybookSpec] = {
    spec.id: spec
    for spec in (CONTRACT_COMPLIANCE_PLAYBOOK, DELIVERY_INTAKE_PLAYBOOK, DPA_REVIEW_PLAYBOOK)
}


def register_playbook(spec: PlaybookSpec, *, replace: bool = False) -> None:
    """Register a playbook at runtime (used by tests and future verticals)."""

    spec.validate()
    if spec.id in _PLAYBOOKS and not replace:
        raise ValueError(f"playbook already registered: {spec.id}")
    _PLAYBOOKS[spec.id] = spec


def list_playbook_ids() -> tuple[str, ...]:
    return tuple(_PLAYBOOKS.keys())


def list_playbooks() -> tuple[PlaybookSpec, ...]:
    return tuple(_PLAYBOOKS.values())


def get_playbook(playbook_id: str) -> PlaybookSpec:
    spec = _PLAYBOOKS.get(playbook_id)
    if spec is None:
        raise KeyError(f"unknown playbook: {playbook_id}")
    return spec


def playbook_summary(spec: PlaybookSpec) -> dict:
    return {
        "id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "dimensions": list(spec.dimensions),
        "rule_count": len(spec.rule_seeds),
        "deliverables": list(spec.deliverables),
    }
