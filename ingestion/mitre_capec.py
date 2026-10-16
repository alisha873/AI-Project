"""
Phase 10 — Milestone 4B: adapter for the native CAPEC XML (official `capec_vX.Y.zip`).

`parse_capec_archive(archive_path, spec, selection)` reads ONE pinned local archive
and returns validated `SourceDocument`s for the curated attack patterns only.
No database work, chunking or network access.

Source format (CAPEC schema, namespace `http://capec.mitre.org/capec-3`)
------------------------------------------------------------------------
This is NOT the CWE format. `<Attack_Pattern_Catalog Name="CAPEC" Version=".." Date="..">`
holds `<Attack_Patterns>` (`<Attack_Pattern ID Name Abstraction Status>`),
`<Categories>`, `<Views>` and `<External_References>`. The official ZIP also
ships an `.xsd` schema file next to the one `.xml`; only the XML is read.

Content sections written (each only when the source supplies it)
----------------------------------------------------------------
Description, Extended Description, Alternate Terms, Likelihood of Attack, Typical
Severity, Prerequisites, Skills Required, Resources Required, Indicators,
Execution Flow (steps, phases and techniques), Consequences, Mitigations,
Related Attack Patterns, Related Weaknesses, Taxonomy Mappings, Notes, References.

NOT included (bounded on purpose): Example_Instances and the full Content_History
(only its latest Modification_Date is used, as `source_modified`).

Relationships kept as `ExternalReference`s, exactly as the SOURCE states them:
- `<Related_Attack_Pattern Nature=X CAPEC_ID=N>` -> relation X, target capec `CAPEC-N`
- `<Related_Weakness CWE_ID=N>`                  -> relation `RelatedWeakness`, target cwe `CWE-N`
- `<Taxonomy_Mapping>` (WASC, OWASP Attacks, ATTACK, ...) -> relation `TaxonomyMapping`
- `<Reference External_Reference_ID=REF-n>`      -> relation `Reference`, target `CAPEC-REF`
Targets outside the curated subset are kept. Nothing is inferred by AutoShield.

CAPEC `Status` values include Stable, Draft, Usable, Deprecated and Obsolete. The
status is written into the document text; a deprecated entry keeps the source's
own "DEPRECATED:" name prefix as its title.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

from ingestion.mitre_xml import (
    ArtifactSpec,
    SelectedRecord,
    SourceFormatError,
    alternate_term_lines,
    assemble_content,
    child_text,
    children,
    citation_entries,
    citation_table,
    collapse,
    consequence_lines,
    latest_modification_date,
    local_target_reference,
    make_document,
    note_lines,
    one_line,
    optional_child,
    parse_catalog_root,
    read_xml_member,
    require_digits,
    select_records,
    taxonomy_entries,
)
from knowledge.schema import SourceDocument

NAMESPACE = "http://capec.mitre.org/capec-3"
ADAPTER_NAME = "mitre_capec"
ADAPTER_VERSION = "mitre-capec-1"


def parse_capec_archive(
    archive_path: str | Path, spec: ArtifactSpec, selection: Sequence[SelectedRecord]
) -> list[SourceDocument]:
    """
    Verify the pinned CAPEC archive and return one `SourceDocument` per selected attack
    pattern (in selection order).

    Raises a `MitreSourceError` subclass for: a missing/unreadable/ambiguous archive,
    checksum mismatch, malformed XML, wrong namespace, unexpected version or date,
    missing/duplicate selected IDs, a name or status that differs from the pin, a
    record missing its name or description, or any schema violation.
    """
    xml_bytes, archive_sha = read_xml_member(
        archive_path, expected_sha256=spec.archive_sha256, expected_member=spec.xml_member
    )
    root = parse_catalog_root(
        xml_bytes,
        namespace=NAMESPACE,
        root_tag="Attack_Pattern_Catalog",
        catalog_name="CAPEC",
        expected_version=spec.source_version,
        expected_date=spec.catalog_date,
    )
    container = optional_child(root, NAMESPACE, "Attack_Patterns", label="CAPEC catalog")
    chosen = select_records(container, NAMESPACE, "Attack_Pattern", selection, catalog="CAPEC")
    citations = citation_table(root, NAMESPACE, catalog="CAPEC")
    return [
        _pattern_document(item, element, citations, spec, archive_sha)
        for item, element in chosen
    ]


def _list_lines(record: ET.Element, container_tag: str, item_tag: str, label: str) -> list[str]:
    """Plain list sections: `Prerequisites/Prerequisite`, `Mitigations/Mitigation`, ..."""
    container = optional_child(record, NAMESPACE, container_tag, label=label)
    return [
        f"- {text}"
        for text in (one_line(e) for e in children(container, NAMESPACE, item_tag))
        if text
    ]


def _skill_lines(record: ET.Element, label: str) -> list[str]:
    container = optional_child(record, NAMESPACE, "Skills_Required", label=label)
    lines: list[str] = []
    for element in children(container, NAMESPACE, "Skill"):
        text = one_line(element)
        level = collapse(element.get("Level"))
        if text:
            lines.append(f"- ({level}) {text}" if level else f"- {text}")
    return lines


def _execution_flow_lines(record: ET.Element, label: str) -> list[str]:
    container = optional_child(record, NAMESPACE, "Execution_Flow", label=label)
    lines: list[str] = []
    for step in children(container, NAMESPACE, "Attack_Step"):
        number = collapse(child_text(step, NAMESPACE, "Step", label=label))
        phase = collapse(child_text(step, NAMESPACE, "Phase", label=label))
        description = collapse(child_text(step, NAMESPACE, "Description", label=label))
        head = f"- Step {number}" if number else "- Step"
        if phase:
            head += f" [{phase}]"
        lines.append(f"{head}: {description}" if description else head)
        for technique in children(step, NAMESPACE, "Technique"):
            text = one_line(technique)
            if text:
                lines.append(f"  - Technique: {text}")
    return lines


def _pattern_document(
    item: SelectedRecord,
    record: ET.Element,
    citations,
    spec: ArtifactSpec,
    archive_sha: str,
) -> SourceDocument:
    label = f"CAPEC-{item.record_id}"
    name = collapse(record.get("Name"))
    if not name:
        raise SourceFormatError(f"{label}: the record has no Name")
    description = child_text(record, NAMESPACE, "Description", label=label)
    if not description:
        raise SourceFormatError(f"{label}: the record has no Description")

    references: list[dict[str, str | None]] = []

    pattern_lines: list[str] = []
    related = optional_child(record, NAMESPACE, "Related_Attack_Patterns", label=label)
    for pattern in children(related, NAMESPACE, "Related_Attack_Pattern"):
        nature = collapse(pattern.get("Nature"))
        target = require_digits(pattern.get("CAPEC_ID"), what="Related_Attack_Pattern CAPEC_ID", label=label)
        if not nature:
            raise SourceFormatError(f"{label}: a Related_Attack_Pattern to CAPEC-{target} has no Nature")
        pattern_lines.append(f"- {nature} CAPEC-{target}")
        references.append(local_target_reference(nature, "capec", target))

    weakness_lines: list[str] = []
    weaknesses = optional_child(record, NAMESPACE, "Related_Weaknesses", label=label)
    for weakness in children(weaknesses, NAMESPACE, "Related_Weakness"):
        target = require_digits(weakness.get("CWE_ID"), what="Related_Weakness CWE_ID", label=label)
        weakness_lines.append(f"- CWE-{target}")
        references.append(local_target_reference("RelatedWeakness", "cwe", target))

    taxonomy_lines, taxonomy_refs = taxonomy_entries(record, NAMESPACE, label=label)
    reference_lines, citation_refs = citation_entries(
        record, NAMESPACE, citations, ref_source="CAPEC-REF", label=label
    )
    references += taxonomy_refs + citation_refs

    consequences = optional_child(record, NAMESPACE, "Consequences", label=label)
    content = assemble_content(
        f"{label}: {name}",
        [
            f"Status: {collapse(record.get('Status'))}" if record.get("Status") else "",
            f"Abstraction: {collapse(record.get('Abstraction'))}" if record.get("Abstraction") else "",
        ],
        [
            ("Description", description),
            ("Extended Description", child_text(record, NAMESPACE, "Extended_Description", label=label) or ""),
            ("Alternate Terms", alternate_term_lines(record, NAMESPACE, label=label)),
            ("Likelihood of Attack", collapse(child_text(record, NAMESPACE, "Likelihood_Of_Attack", label=label))),
            ("Typical Severity", collapse(child_text(record, NAMESPACE, "Typical_Severity", label=label))),
            ("Prerequisites", _list_lines(record, "Prerequisites", "Prerequisite", label)),
            ("Skills Required", _skill_lines(record, label)),
            ("Resources Required", _list_lines(record, "Resources_Required", "Resource", label)),
            ("Indicators", _list_lines(record, "Indicators", "Indicator", label)),
            ("Execution Flow", _execution_flow_lines(record, label)),
            ("Consequences", consequence_lines(consequences, NAMESPACE, label=label)),
            ("Mitigations", _list_lines(record, "Mitigations", "Mitigation", label)),
            ("Related Attack Patterns", pattern_lines),
            ("Related Weaknesses", weakness_lines),
            ("Taxonomy Mappings", taxonomy_lines),
            ("Notes", note_lines(record, NAMESPACE, label=label)),
            ("References", reference_lines),
        ],
    )
    return make_document(
        source="capec",
        record_id=item.record_id,
        doc_type="attack_pattern",
        title=name,
        content=content,
        source_modified=latest_modification_date(record, NAMESPACE, label=label),
        references=references,
        spec=spec,
        adapter=ADAPTER_NAME,
        adapter_version=ADAPTER_VERSION,
        archive_sha256=archive_sha,
    )
