"""Playbook package: business verticals as data."""

from backend.playbooks.base import PlaybookSpec
from backend.playbooks.registry import (
    get_playbook,
    list_playbook_ids,
    list_playbooks,
    playbook_summary,
    register_playbook,
)

__all__ = [
    "PlaybookSpec",
    "get_playbook",
    "list_playbook_ids",
    "list_playbooks",
    "playbook_summary",
    "register_playbook",
]
