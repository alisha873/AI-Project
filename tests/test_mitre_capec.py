"""
Tests for `ingestion.mitre_capec`.

Inputs are of two labelled kinds (see tests/mitre_helpers.py):
- VERBATIM EXTRACT of the official CAPEC v3.9 XML (CAPEC-664, -137, -106, -242);
- SYNTHETIC XML: invented, minimal, valid or deliberately broken. Not official records.

Archive-level checks (checksum, ZIP members, DTD refusal) are shared code and are
tested thoroughly in `test_mitre_cwe.py`; here only what is CAPEC-specific, plus the
CAPEC ZIP shape (an `.xsd` next to the single `.xml`).
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from ingestion.mitre_capec import ADAPTER_NAME, ADAPTER_VERSION, parse_capec_archive
from ingestion.mitre_xml import (
    ArchiveError,
    ChecksumMismatchError,
    SelectedRecord,
    SelectionError,
    SourceFormatError,
    SourceVersionError,
)
from knowledge.schema import SourceDocument
from tests.mitre_helpers import (
    FIXTURE_DIR,
    INGESTED_AT,
    RETRIEVED_AT,
    TEST_LICENSE,
    attack_pattern,
    capec_archive,
    capec_catalog_xml,
    capec_extract_archive,
    cwe_catalog_xml,
)

SSRF = SelectedRecord("664", "Server Side Request Forgery")
PARAM = SelectedRecord("137", "Parameter Injection")
CODE = SelectedRecord("242", "Code Injection")
DEPRECATED = SelectedRecord("106", "DEPRECATED: XSS through Log Files", "Deprecated")
SYNTH = SelectedRecord("1", "Synthetic Pattern")


def parse(archive, selection):
    return parse_capec_archive(archive.path, archive.spec, selection)


def record_slice(record_id: str) -> str:
    text = (FIXTURE_DIR / "capec_v3.9_extract.xml").read_text(encoding="utf-8")
    return re.search(rf'<Attack_Pattern ID="{record_id}"[\s\S]*?</Attack_Pattern>', text).group(0)


# ============================================================
# VERBATIM OFFICIAL EXTRACT
# ============================================================

class TestOfficialExtract:
    @pytest.fixture()
    def docs(self, tmp_path):
        return {d.source_id: d for d in parse(capec_extract_archive(tmp_path), [SSRF, PARAM, DEPRECATED, CODE])}

    def test_identity_and_metadata(self, docs):
        doc = docs["CAPEC-664"]
        assert doc.doc_id == "capec:CAPEC-664"
        assert doc.source == "capec"
        assert doc.source_version == "3.9"
        assert doc.doc_type == "attack_pattern"
        assert doc.title == "Server Side Request Forgery"
        assert doc.source_url == "https://capec.mitre.org/data/definitions/664.html"
        assert doc.license == TEST_LICENSE
        assert doc.content_origin == "source_text"

    def test_provenance_describes_the_real_artifact(self, tmp_path):
        archive = capec_extract_archive(tmp_path)
        prov = parse(archive, [SSRF])[0].provenance
        assert prov.origin_kind == "local_file"
        assert prov.artifact_path == "capec_v3.9.zip"
        assert prov.artifact_sha256 == archive.sha256
        assert (prov.adapter, prov.adapter_version) == (ADAPTER_NAME, ADAPTER_VERSION)
        assert (prov.retrieved_at_utc, prov.ingested_at_utc) == (RETRIEVED_AT, INGESTED_AT)

    def test_source_modified_is_the_latest_modification_date(self, docs):
        for source_id, doc in docs.items():
            dates = re.findall(
                r"<Modification_Date>([0-9-]+)</Modification_Date>", record_slice(source_id.split("-")[1])
            )
            assert doc.source_modified == max(dates)

    def test_all_the_capec_sections_are_preserved_in_order(self, docs):
        content = docs["CAPEC-664"].content
        assert content.startswith("# CAPEC-664: Server Side Request Forgery\n")
        assert "Status: Stable" in content and "Abstraction: Standard" in content
        assert re.findall(r"^## (.+)$", content, re.MULTILINE) == [
            "Description", "Likelihood of Attack", "Typical Severity", "Prerequisites",
            "Skills Required", "Resources Required", "Execution Flow", "Consequences",
            "Mitigations", "Related Attack Patterns", "Related Weaknesses", "References",
        ]
        assert "Server Side Request Forgery attacks differ from Cross Site Request Forgery" in content
        assert re.search(r"^- Step 1 \[[A-Za-z]+\]: ", content, re.MULTILINE)
        assert re.search(r"^  - Technique: ", content, re.MULTILINE)
        assert "xhtml" not in content and "<" not in content

    def test_sections_absent_from_the_source_are_omitted(self, docs):
        assert "<Execution_Flow" not in record_slice("137")
        assert "Execution Flow" not in docs["CAPEC-137"].content
        assert "<Indicators" not in record_slice("664") and "Indicators" not in docs["CAPEC-664"].content

    def test_relationships_are_explicit_source_facts(self, docs):
        refs = {(r.relation, r.target_source, r.target_source_id): r.url for r in docs["CAPEC-664"].references}
        assert refs[("ChildOf", "capec", "CAPEC-115")] == "https://capec.mitre.org/data/definitions/115.html"
        assert refs[("RelatedWeakness", "cwe", "CWE-918")] == "https://cwe.mitre.org/data/definitions/918.html"
        assert ("RelatedWeakness", "cwe", "CWE-20") in refs
        assert any(key[0] == "Reference" and key[1] == "CAPEC-REF" for key in refs)

    def test_references_outside_the_selection_are_retained(self, docs):
        # CWE-88 is not in this extract (nor selected); the source names it, so it is kept.
        refs = [(r.relation, r.target_source_id) for r in docs["CAPEC-137"].references]
        assert refs == [("RelatedWeakness", "CWE-88")]

    def test_taxonomy_mapping_without_entry_id_uses_the_entry_name(self, docs):
        refs = {(r.relation, r.target_source, r.target_source_id) for r in docs["CAPEC-242"].references}
        assert ("TaxonomyMapping", "OWASP Attacks", "Code Injection") in refs
        assert "- OWASP Attacks: Code Injection" in docs["CAPEC-242"].content

    def test_deprecated_status_is_preserved(self, docs):
        doc = docs["CAPEC-106"]
        assert doc.title.startswith("DEPRECATED:")
        assert "Status: Deprecated" in doc.content
        assert "has been deprecated" in doc.content

    def test_documents_pass_schema_revalidation_and_are_stable(self, tmp_path):
        archive = capec_extract_archive(tmp_path)
        first = parse(archive, [SSRF, PARAM, DEPRECATED, CODE])
        second = parse(archive, [SSRF, PARAM, DEPRECATED, CODE])
        assert first == second
        assert [d.document_sha256 for d in first] == [d.document_sha256 for d in second]
        for doc in first:
            assert SourceDocument.model_validate(doc.model_dump(mode="json")) == doc

    def test_the_shipped_xsd_member_does_not_make_the_archive_ambiguous(self, tmp_path):
        xml = (FIXTURE_DIR / "capec_v3.9_extract.xml").read_bytes()
        archive = capec_archive(tmp_path, xml, extra_members={"ap_schema_v3.5.xsd": b"<xs:schema/>"})
        assert parse(archive, [SSRF])[0].source_id == "CAPEC-664"


# ============================================================
# SYNTHETIC CONTENT
# ============================================================

def synth(tmp_path, record_body: str = "<Description>Synthetic description.</Description>", **kwargs):
    xml = capec_catalog_xml(f"<Attack_Patterns>{attack_pattern(body=record_body)}</Attack_Patterns>")
    return parse(capec_archive(tmp_path, xml, **kwargs), [SYNTH])


class TestSyntheticContent:
    def test_minimal_record(self, tmp_path):
        doc = synth(tmp_path)[0]
        assert doc.content == (
            "# CAPEC-1: Synthetic Pattern\n\nStatus: Draft\nAbstraction: Standard\n\n"
            "## Description\nSynthetic description."
        )
        assert doc.references == () and doc.source_modified is None

    def test_execution_flow_prerequisites_skills_and_mitigations(self, tmp_path):
        body = (
            "<Description>D.</Description>"
            "<Prerequisites><Prerequisite>Needs input.</Prerequisite><Prerequisite>  </Prerequisite></Prerequisites>"
            '<Skills_Required><Skill Level="Low">Basic.</Skill></Skills_Required>'
            "<Execution_Flow><Attack_Step><Step>1</Step><Phase>Explore</Phase><Description>Look.</Description>"
            "<Technique>Spider</Technique><Technique>Sniff</Technique></Attack_Step></Execution_Flow>"
            "<Mitigations><Mitigation>Validate <xhtml:b>all</xhtml:b> input.</Mitigation></Mitigations>"
        )
        content = synth(tmp_path, body)[0].content
        assert "## Prerequisites\n- Needs input.\n\n" in content  # blank prerequisite is dropped
        assert "## Skills Required\n- (Low) Basic." in content
        assert "- Step 1 [Explore]: Look.\n  - Technique: Spider\n  - Technique: Sniff" in content
        assert "## Mitigations\n- Validate all input." in content

    def test_related_patterns_and_weaknesses(self, tmp_path):
        body = (
            "<Description>D.</Description>"
            '<Related_Attack_Patterns><Related_Attack_Pattern Nature="CanPrecede" CAPEC_ID="7"/></Related_Attack_Patterns>'
            '<Related_Weaknesses><Related_Weakness CWE_ID="89"/></Related_Weaknesses>'
        )
        doc = synth(tmp_path, body)[0]
        assert {(r.relation, r.target_source, r.target_source_id) for r in doc.references} == {
            ("CanPrecede", "capec", "CAPEC-7"),
            ("RelatedWeakness", "cwe", "CWE-89"),
        }
        assert "- CanPrecede CAPEC-7" in doc.content and "- CWE-89" in doc.content

    @pytest.mark.parametrize(
        "relation",
        [
            '<Related_Attack_Patterns><Related_Attack_Pattern CAPEC_ID="7"/></Related_Attack_Patterns>',  # no Nature
            '<Related_Attack_Patterns><Related_Attack_Pattern Nature="ChildOf" CAPEC_ID="x"/></Related_Attack_Patterns>',
            '<Related_Weaknesses><Related_Weakness CWE_ID="CWE-89"/></Related_Weaknesses>',
            "<Related_Weaknesses><Related_Weakness/></Related_Weaknesses>",
        ],
    )
    def test_malformed_relationships_fail(self, tmp_path, relation):
        with pytest.raises(SourceFormatError):
            synth(tmp_path, "<Description>D.</Description>" + relation)

    @pytest.mark.parametrize("body", ["", "<Description/>"])
    def test_missing_description_fails(self, tmp_path, body):
        with pytest.raises(SourceFormatError, match="no Description"):
            synth(tmp_path, body)

    def test_example_instances_are_not_part_of_the_content(self, tmp_path):
        body = "<Description>D.</Description><Example_Instances><Example>SECRETEXAMPLE</Example></Example_Instances>"
        assert "SECRETEXAMPLE" not in synth(tmp_path, body)[0].content


# ============================================================
# VERSION, NAMESPACE, SELECTION, CHECKSUM
# ============================================================

class TestVersionNamespaceSelection:
    def test_unexpected_version_is_rejected(self, tmp_path):
        xml = capec_catalog_xml(f"<Attack_Patterns>{attack_pattern()}</Attack_Patterns>", version="3.8")
        with pytest.raises(SourceVersionError, match="3.8.*3.9"):
            parse(capec_archive(tmp_path, xml), [SYNTH])

    def test_unexpected_date_is_rejected(self, tmp_path):
        xml = capec_catalog_xml(f"<Attack_Patterns>{attack_pattern()}</Attack_Patterns>", date="2023-01-25")
        with pytest.raises(SourceVersionError, match="date"):
            parse(capec_archive(tmp_path, xml), [SYNTH])

    def test_the_cwe_format_is_not_accepted_as_capec(self, tmp_path):
        xml = cwe_catalog_xml("<Weaknesses/>", version="3.9", date="2023-01-24")
        with pytest.raises(SourceFormatError, match="unexpected root element"):
            parse(capec_archive(tmp_path, xml), [SYNTH])

    def test_malformed_xml_is_rejected(self, tmp_path):
        with pytest.raises(SourceFormatError, match="malformed XML"):
            parse(capec_archive(tmp_path, b"<Attack_Pattern_Catalog"), [SYNTH])

    def test_missing_duplicate_and_renamed_selected_ids_fail(self, tmp_path):
        xml = capec_catalog_xml(
            f"<Attack_Patterns>{attack_pattern('1')}{attack_pattern('2')}{attack_pattern('2')}"
            f"{attack_pattern('3', name='Renamed')}</Attack_Patterns>"
        )
        archive = capec_archive(tmp_path, xml)
        selection = [
            SYNTH,
            SelectedRecord("2", "Synthetic Pattern"),
            SelectedRecord("3", "Synthetic Pattern"),
            SelectedRecord("4", "Absent"),
        ]
        with pytest.raises(SelectionError) as info:
            parse(archive, selection)
        message = str(info.value)
        assert "ID 2 occurs 2 times" in message
        assert "ID 3: name is 'Renamed'" in message
        assert "ID 4 is missing" in message

    def test_status_pin_is_enforced(self, tmp_path):
        xml = capec_catalog_xml(f"<Attack_Patterns>{attack_pattern(status='Deprecated')}</Attack_Patterns>")
        pinned = SelectedRecord("1", "Synthetic Pattern", "Stable")
        with pytest.raises(SelectionError, match="pinned status"):
            parse(capec_archive(tmp_path, xml), [pinned])

    def test_dangling_citation_fails(self, tmp_path):
        body = '<Description>D.</Description><References><Reference External_Reference_ID="REF-5"/></References>'
        with pytest.raises(SourceFormatError, match="REF-5"):
            synth(tmp_path, body)

    def test_checksum_mismatch_and_missing_archive_fail(self, tmp_path):
        archive = capec_extract_archive(tmp_path)
        wrong = dataclasses.replace(archive.spec, archive_sha256="f" * 64)
        with pytest.raises(ChecksumMismatchError):
            parse_capec_archive(archive.path, wrong, [SSRF])
        with pytest.raises(ArchiveError, match="not found"):
            parse_capec_archive(tmp_path / "nope.zip", archive.spec, [SSRF])
