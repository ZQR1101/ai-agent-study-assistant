"""Playbook specification: the data contract every business vertical implements.

A playbook is DATA, not code: adding a new business vertical means adding a
spec instance (dimensions + rulebook seeds + deliverable template), never
touching the engine. The registry seeds rules into the database on first run;
after that, admins edit rules in the database and changes take effect on the
next document without redeployment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlaybookSpec:
    id: str
    name: str
    description: str
    friendly_id_prefix: str
    dimensions: tuple[str, ...]
    rule_seeds: tuple[dict, ...] = ()
    # Phase-3 deliverable descriptors (rendered by engine.export)
    deliverables: tuple[str, ...] = ()
    scoring_instructions: str = ""

    def validate(self) -> None:
        if not self.id or not self.friendly_id_prefix:
            raise ValueError("playbook requires id and friendly_id_prefix")
        seen = set()
        for seed in self.rule_seeds:
            if seed["dimension"] not in self.dimensions:
                raise ValueError(
                    f"rule '{seed['name']}' dimension '{seed['dimension']}' "
                    f"not in playbook dimensions {self.dimensions}"
                )
            key = (seed["dimension"], seed["name"])
            if key in seen:
                raise ValueError(f"duplicate rule seed: {key}")
            seen.add(key)
