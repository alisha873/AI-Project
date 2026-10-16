"""
Phase 10 — Milestone 4B: adapter for the native CWE XML (official `cwec_vX.Y.xml.zip`).

`parse_cwe_archive(archive_path, spec, selection)` reads ONE pinned local archive
and returns validated `SourceDocument`s for the curated weaknesses only. It
does no database work, chunking or network access.

Source format (CWE schema v7, namespace `http://cwe.mitre.org/cwe-7`)
---------------------------------------------------------------------
`<Weakness_Catalog Name="CWE" Version=".." Date="..">` holds `<Weaknesses>`
(`<Weakness ID Name Abstraction Structure Status>`), `<Categories>`, `<Views>`
and `<External_References>`. Only `<Weakness>` records are read.

Content sections written (each only when the source supplies it)
----------------------------------------------------------------
Description, Extended Description, Background Details, Alternate Terms, Modes of
Introduction, Applicable Platforms, Likelihood of Exploit, Common Consequences,
Detection Methods, Potential Mitigations, Relationships, Related Attack
Patterns, Taxonomy Mappings, Notes, References.

NOT included (bounded on purpose): Demonstrative_Examples and Observed_Examples
(long code and CVE lists), Weakness_Ordinalities, Mapping_Notes,
Functional_Areas, Affected_Resources and the full Content_History (only its
latest Modification_Date is used, as `source_modified`).

Relationships kept as `ExternalReference`s, exactly as the SOURCE states them:
- `<Related_Weakness Nature=X CWE_ID=N>`  -> relation X, target cwe `CWE-N`
- `<Related_Attack_Pattern CAPEC_ID=N>`   -> relation `RelatedAttackPattern`, target capec `CAPEC-N`
- `<Taxonomy_Mapping>`                    -> relation `TaxonomyMapping`
- `<Reference External_Reference_ID=REF-n>` -> relation `Reference`, target `CWE-REF`
Targets outside the curated subset are kept. The View_ID/Ordinal of a
Related_Weakness cannot be stored in an `ExternalReference`; it appears in the
document text. Nothing is inferred by AutoShield.

Known fidelity limits (the text always keeps every item; only the structured
references are affected):
- A taxonomy mapping is identified by its `Entry_ID` (or `Entry_Name` when
  there is no ID). Two mappings with the same taxonomy and `Entry_ID` but
  different names (e.g. ISA/IEC 62443 "Part 3-3": "Req SR 1.1" and "Req SR 1.2")
  collapse into ONE `ExternalReference`; both stay in the "Taxonomy Mappings" text.
- `<Reference>` entries that occur only inside the excluded Demonstrative_Examples
  are not collected; only the record's own top-level `<References>` are.

The record's `Status` (Stable, Draft, Incomplete, Deprecated, ...) is written
into the text of the document; deprecated entries also keep the source's own
"DEPRECATED:" name prefix as their title.
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
    rich_text,
    select_records,
    taxonomy_entries,
)
from knowledge.schema import SourceDocument

NAMESPACE = "http://cwe.mitre.org/cwe-7"
ADAPTER_NAME = "mitre_cwe"
ADAPTER_VERSION = "mitre-cwe-1"
_PLATFORM_KINDS = ("Language", "Operating_System", "Architecture", "Paradigm", "Technology")


def parse_cwe_archive(
    archive_path: str | Path, spec: ArtifactSpec, selection: Sequence[SelectedRecord]
) -> list[SourceDocument]:
    """
    Verify the pinned CWE archive and return one `SourceDocument` per selected weakness
    (in selection order).

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
        root_tag="Weakness_Catalog",
        catalog_name="CWE",
        expected_version=spec.source_version,
        expected_date=spec.catalog_date,
    )
    container = optional_child(root, NAMESPACE, "Weaknesses", label="CWE catalog")
    chosen = select_records(container, NAMESPACE, "Weakness", selection, catalog="CWE")
    citations = citation_table(root, NAMESPACE, catalog="CWE")
    return [
        _weakness_document(item, element, citations, spec, archive_sha)
        for item, element in chosen
    ]


def _platform_lines(record: ET.Element, label: str) -> list[str]:
    container = optional_child(record, NAMESPACE, "Applicable_Platforms", label=label)
    lines: list[str] = []
    for kind in _PLATFORM_KINDS:
        for element in children(container, NAMESPACE, kind):
            name = collapse(element.get("Name")) or collapse(element.get("Class"))
            if not name:
                continue
            prevalence = collapse(element.get("Prevalence"))
            suffix = f" ({prevalence})" if prevalence else ""
            lines.append(f"- {kind.replace('_', ' ')}: {name}{suffix}")
    return lines


def _introduction_lines(record: ET.Element, label: str) -> list[str]:
    container = optional_child(record, NAMESPACE, "Modes_Of_Introduction", label=label)
    lines: list[str] = []
    for element in children(container, NAMESPACE, "Introduction"):
        phases = ", ".join(one_line(p) for p in children(element, NAMESPACE, "Phase"))
        note = child_text(element, NAMESPACE, "Note", label=label)
        if phases and note:
            lines.append(f"- {phases}: {collapse(note)}")
        elif phases or note:
            lines.append(f"- {phases or collapse(note)}")
    return lines


def _detection_lines(record: ET.Element, label: str) -> list[str]:
    container = optional_child(record, NAMESPACE, "Detection_Methods", label=label)
    lines: list[str] = []
    for element in children(container, NAMESPACE, "Detection_Method"):
        method = collapse(child_text(element, NAMESPACE, "Method", label=label))
        description = collapse(child_text(element, NAMESPACE, "Description", label=label))
        effectiveness = collapse(child_text(element, NAMESPACE, "Effectiveness", label=label))
        eff_notes = collapse(child_text(element, NAMESPACE, "Effectiveness_Notes", label=label))
        if not (method or description):
            continue
        line = f"- {method or 'Detection method'}"
        if effectiveness:
            line += f" (effectiveness: {effectiveness})"
        if description:
            line += f": {description}"
        if eff_notes:
            line += f" Effectiveness notes: {eff_notes}"
        lines.append(line)
    return lines


def _mitigation_lines(record: ET.Element, label: str) -> list[str]:
    container = optional_child(record, NAMESPACE, "Potential_Mitigations", label=label)
    lines: list[str] = []
    for element in children(container, NAMESPACE, "Mitigation"):
        phases = ", ".join(one_line(p) for p in children(element, NAMESPACE, "Phase"))
        strategy = collapse(child_text(element, NAMESPACE, "Strategy", label=label))
        description = collapse(child_text(element, NAMESPACE, "Description", label=label))
        effectiveness = collapse(child_text(element, NAMESPACE, "Effectiveness", label=label))
        eff_notes = collapse(child_text(element, NAMESPACE, "Effectiveness_Notes", label=label))
        if not description:
            continue
        tags = [t for t in (phases, strategy) if t]
        line = "- " + (f"[{'; '.join(tags)}] " if tags else "") + description
        if effectiveness:
            line += f" (effectiveness: {effectiveness})"
        if eff_notes:
            line += f" Effectiveness notes: {eff_notes}"
        lines.append(line)
    return lines


def _weakness_document(
    item: SelectedRecord,
    record: ET.Element,
    citations,
    spec: ArtifactSpec,
    archive_sha: str,
) -> SourceDocument:
    label = f"CWE-{item.record_id}"
    name = collapse(record.get("Name"))
    if not name:
        raise SourceFormatError(f"{label}: the record has no Name")
    description = child_text(record, NAMESPACE, "Description", label=label)
    if not description:
        raise SourceFormatError(f"{label}: the record has no Description")

    references: list[dict[str, str | None]] = []

    relationship_lines: list[str] = []
    for related in children(optional_child(record, NAMESPACE, "Related_Weaknesses", label=label), NAMESPACE, "Related_Weakness"):
        nature = collapse(related.get("Nature"))
        target = require_digits(related.get("CWE_ID"), what="Related_Weakness CWE_ID", label=label)
        if not nature:
            raise SourceFormatError(f"{label}: a Related_Weakness to CWE-{target} has no Nature")
        details = [
            f"{key} {collapse(related.get(attr))}"
            for key, attr in (("view", "View_ID"), ("ordinal", "Ordinal"))
            if collapse(related.get(attr))
        ]
        suffix = f" ({', '.join(details)})" if details else ""
        relationship_lines.append(f"- {nature} CWE-{target}{suffix}")
        references.append(local_target_reference(nature, "cwe", target))

    pattern_lines: list[str] = []
    patterns = optional_child(record, NAMESPACE, "Related_Attack_Patterns", label=label)
    for pattern in children(patterns, NAMESPACE, "Related_Attack_Pattern"):
        target = require_digits(pattern.get("CAPEC_ID"), what="Related_Attack_Pattern CAPEC_ID", label=label)
        pattern_lines.append(f"- CAPEC-{target}")
        references.append(local_target_reference("RelatedAttackPattern", "capec", target))

    taxonomy_lines, taxonomy_refs = taxonomy_entries(record, NAMESPACE, label=label)
    reference_lines, citation_refs = citation_entries(
        record, NAMESPACE, citations, ref_source="CWE-REF", label=label
    )
    references += taxonomy_refs + citation_refs

    background = optional_child(record, NAMESPACE, "Background_Details", label=label)
    background_text = "\n\n".join(
        rich_text(detail) for detail in children(background, NAMESPACE, "Background_Detail")
    )
    consequences = optional_child(record, NAMESPACE, "Common_Consequences", label=label)

    content = assemble_content(
        f"{label}: {name}",
        [
            f"Status: {collapse(record.get('Status'))}" if record.get("Status") else "",
            f"Abstraction: {collapse(record.get('Abstraction'))}" if record.get("Abstraction") else "",
            f"Structure: {collapse(record.get('Structure'))}" if record.get("Structure") else "",
        ],
        [
            ("Description", description),
            ("Extended Description", child_text(record, NAMESPACE, "Extended_Description", label=label) or ""),
            ("Background Details", background_text),
            ("Alternate Terms", alternate_term_lines(record, NAMESPACE, label=label)),
            ("Modes of Introduction", _introduction_lines(record, label)),
            ("Applicable Platforms", _platform_lines(record, label)),
            ("Likelihood of Exploit", collapse(child_text(record, NAMESPACE, "Likelihood_Of_Exploit", label=label))),
            ("Common Consequences", consequence_lines(consequences, NAMESPACE, label=label)),
            ("Detection Methods", _detection_lines(record, label)),
            ("Potential Mitigations", _mitigation_lines(record, label)),
            ("Relationships", relationship_lines),
            ("Related Attack Patterns", pattern_lines),
            ("Taxonomy Mappings", taxonomy_lines),
            ("Notes", note_lines(record, NAMESPACE, label=label)),
            ("References", reference_lines),
        ],
    )
    return make_document(
        source="cwe",
        record_id=item.record_id,
        doc_type="weakness",
        title=name,
        content=content,
        source_modified=latest_modification_date(record, NAMESPACE, label=label),
        references=references,
        spec=spec,
        adapter=ADAPTER_NAME,
        adapter_version=ADAPTER_VERSION,
        archive_sha256=archive_sha,
    )
