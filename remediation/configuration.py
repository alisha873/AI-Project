"""
Phase 12 - infrastructure / configuration context.

Configuration facts are statements about how the environment is SET UP (for example "container X is
attached to network Y"). They are not observed activity, they are not hypotheses and they are not
reference knowledge, so they get their own category and their own identifiers (`CFG-1`, `CFG-2`, ...).

Where the facts come from
-------------------------
Only from what the pipeline already produced: the `graph_relationships.configured_only` list of the
Phase 8 reconstruction, in which every edge carries `basis: "configured"` and the artifacts it was read
from. Callers may add further VERIFIED facts explicitly (`ConfigurationFactInput`). This module reads no
file, no Docker artifact and no `ground_truth/` data.

What this module will not do
----------------------------
It does not invent facts and it does not interpret the lab's compose or container files. It therefore says
nothing about image versions, user accounts, ports, capabilities or file systems unless a caller supplies
such a fact. For `incident_001` the Phase 8 edges only describe network membership and start-up processes.

Status of the configuration context
-----------------------------------
available    at least one usable fact and no conflict.
unavailable  no usable fact. Proposals must then say they depend on configuration details nobody supplied.
conflicting  two facts disagree about a SINGLE-VALUED attribute of the same subject. Both are withheld
             (a model must not pick a winner) and the disagreement is reported as an open question.

A fact is single-valued only when its supplier says so (`single_valued=True`): a container may be attached
to several networks, so the Phase 8 relationships are never treated as exclusive.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from retrieval.query_builder import FORBIDDEN_INPUT_KEYS

ConfigurationStatus = Literal["available", "unavailable", "conflicting"]
FACT_ID_PREFIX = "CFG-"


class ConfigurationError(ValueError):
    """The supplied configuration data is malformed."""


@dataclass(frozen=True)
class ConfigurationFactInput:
    """A verified configuration fact supplied by a caller (before an ID is assigned)."""

    subject: str
    attribute: str
    value: str
    source_artifacts: tuple[str, ...] = ()
    single_valued: bool = False

    def __post_init__(self) -> None:
        for name in ("subject", "attribute", "value"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class ConfigurationFact:
    fact_id: str
    subject: str
    attribute: str
    value: str
    detail: str | None
    source_artifacts: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    single_valued: bool
    basis: str = "configured"


@dataclass(frozen=True)
class ConfigurationConflict:
    subject: str
    attribute: str
    fact_ids: tuple[str, ...]
    values: tuple[str, ...]


@dataclass(frozen=True)
class ConfigurationContext:
    status: ConfigurationStatus
    facts: tuple[ConfigurationFact, ...]            # usable facts only (conflicting ones are withheld)
    conflicts: tuple[ConfigurationConflict, ...]
    notes: tuple[str, ...]

    @property
    def fact_ids(self) -> tuple[str, ...]:
        return tuple(fact.fact_id for fact in self.facts)

    @property
    def conflicting_fact_ids(self) -> tuple[str, ...]:
        return tuple(sorted({fact_id for conflict in self.conflicts for fact_id in conflict.fact_ids}))


UNAVAILABLE = ConfigurationContext(
    status="unavailable",
    facts=(),
    conflicts=(),
    notes=("No configuration fact was supplied.",),
)


def _clean_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(sorted({item.strip() for item in value if isinstance(item, str) and item.strip()}))


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _edges_from_reconstruction(reconstruction: Mapping) -> tuple[list[dict], list[str]]:
    """Raw fact dictionaries from the Phase 8 configured-only edges, and notes about skipped entries."""
    relationships = reconstruction.get("graph_relationships")
    notes: list[str] = []
    if relationships is None:
        return [], ["The reconstruction has no graph_relationships section."]
    if not isinstance(relationships, Mapping):
        raise ConfigurationError("reconstruction.graph_relationships must be an object")
    found = sorted(FORBIDDEN_INPUT_KEYS.intersection(str(key) for key in relationships))
    if found:
        raise ConfigurationError(f"graph_relationships contains evaluation-answer key(s) {found}")

    edges = relationships.get("configured_only")
    if edges is None:
        return [], ["The reconstruction lists no configured relationships."]
    if not isinstance(edges, (list, tuple)):
        raise ConfigurationError("graph_relationships.configured_only must be a list")

    facts: list[dict] = []
    skipped = 0
    for edge in edges:
        if not isinstance(edge, Mapping) or edge.get("basis") != "configured":
            skipped += 1  # only edges that declare themselves configured are trusted as configuration
            continue
        subject, attribute, value = _text(edge.get("source")), _text(edge.get("relationship")), _text(edge.get("target"))
        if not (subject and attribute and value):
            skipped += 1
            continue
        attributes = edge.get("attributes") if isinstance(edge.get("attributes"), Mapping) else {}
        facts.append(
            {
                "subject": subject,
                "attribute": attribute,
                "value": value,
                "detail": _text(attributes.get("relationship_basis")),
                "source_artifacts": _clean_tuple(edge.get("source_artifacts")),
                "source_event_ids": _clean_tuple(edge.get("evidence_ids")),
                "single_valued": False,
            }
        )
    if skipped:
        notes.append(f"{skipped} configured-relationship entr{'y was' if skipped == 1 else 'ies were'} skipped as malformed or not marked configured.")
    return facts, notes


def build_configuration_context(
    reconstruction: Mapping | None,
    extra_facts: tuple[ConfigurationFactInput, ...] = (),
) -> ConfigurationContext:
    """
    Configuration facts from the Phase 8 `configured_only` edges plus any caller-supplied facts.

    Identical (subject, attribute, value) facts are merged. IDs are assigned in sorted order so the same
    input always yields the same IDs. Conflicting single-valued facts are withheld and reported.
    """
    raw: list[dict] = []
    notes: list[str] = []
    if reconstruction is not None:
        if not isinstance(reconstruction, Mapping):
            raise ConfigurationError("reconstruction must be an object or None")
        raw, notes = _edges_from_reconstruction(reconstruction)
    for extra in extra_facts:
        if not isinstance(extra, ConfigurationFactInput):
            raise ConfigurationError("extra_facts must contain ConfigurationFactInput items")
        raw.append(
            {
                "subject": extra.subject.strip(),
                "attribute": extra.attribute.strip(),
                "value": extra.value.strip(),
                "detail": None,
                "source_artifacts": _clean_tuple(extra.source_artifacts),
                "source_event_ids": (),
                "single_valued": extra.single_valued,
            }
        )

    merged: dict[tuple[str, str, str], dict] = {}
    for item in raw:
        key = (item["subject"], item["attribute"], item["value"])
        if key in merged:
            previous = merged[key]
            previous["source_artifacts"] = tuple(sorted({*previous["source_artifacts"], *item["source_artifacts"]}))
            previous["source_event_ids"] = tuple(sorted({*previous["source_event_ids"], *item["source_event_ids"]}))
            previous["single_valued"] = previous["single_valued"] or item["single_valued"]
        else:
            merged[key] = dict(item)

    everything = [
        ConfigurationFact(fact_id=f"{FACT_ID_PREFIX}{number}", **merged[key])
        for number, key in enumerate(sorted(merged), start=1)
    ]

    groups: dict[tuple[str, str], list[ConfigurationFact]] = {}
    for fact in everything:
        if fact.single_valued:
            groups.setdefault((fact.subject, fact.attribute), []).append(fact)
    conflicts = tuple(
        ConfigurationConflict(
            subject=subject,
            attribute=attribute,
            fact_ids=tuple(f.fact_id for f in group),
            values=tuple(f.value for f in group),
        )
        for (subject, attribute), group in sorted(groups.items())
        if len({f.value for f in group}) > 1
    )
    withheld = {fact_id for conflict in conflicts for fact_id in conflict.fact_ids}
    usable = tuple(fact for fact in everything if fact.fact_id not in withheld)

    if conflicts:
        status: ConfigurationStatus = "conflicting"
    elif usable:
        status = "available"
    else:
        status = "unavailable"
        notes.append("No usable configuration fact is available.")
    return ConfigurationContext(status=status, facts=usable, conflicts=conflicts, notes=tuple(notes))
