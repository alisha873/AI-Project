"""
Tests for `ingestion.mitre_cwe` (and the shared archive/XML checks in `ingestion.mitre_xml`).

Inputs are of two labelled kinds (see tests/mitre_helpers.py):
- VERBATIM EXTRACT of the official CWE v4.20 XML (CWE-918, CWE-1187): tests the
  real source structure, namespaces and wording;
- SYNTHETIC XML: invented, minimal, valid or deliberately broken. Not official records.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import zipfile

import pytest

from ingestion import mitre_xml
from ingestion.mitre_cwe import ADAPTER_NAME, ADAPTER_VERSION, parse_cwe_archive
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
    cwe_archive,
    cwe_catalog_xml,
    cwe_extract_archive,
    make_spec,
    weakness,
    write_zip,
)

SSRF = SelectedRecord("918", "Server-Side Request Forgery (SSRF)", "Incomplete")
DEPRECATED = SelectedRecord("1187", "DEPRECATED: Use of Uninitialized Resource", "Deprecated")
SYNTH = SelectedRecord("1", "Synthetic Weakness")


def parse(archive, selection):
    return parse_cwe_archive(archive.path, archive.spec, selection)


def fixture_text() -> str:
    return (FIXTURE_DIR / "cwe_v4.20_extract.xml").read_text(encoding="utf-8")


def record_slice(record_id: str) -> str:
    return re.search(rf'<Weakness ID="{record_id}"[\s\S]*?</Weakness>', fixture_text()).group(0)


# ============================================================
# VERBATIM OFFICIAL EXTRACT
# ============================================================

class TestOfficialExtract:
    @pytest.fixture()
    def docs(self, tmp_path):
        return parse(cwe_extract_archive(tmp_path), [SSRF, DEPRECATED])

    def test_identity_and_metadata(self, tmp_path):
        archive = cwe_extract_archive(tmp_path)
        doc = parse(archive, [SSRF])[0]
        assert doc.doc_id == "cwe:CWE-918"
        assert doc.source == "cwe"
        assert doc.source_id == "CWE-918"
        assert doc.source_version == "4.20"
        assert doc.doc_type == "weakness"
        assert doc.title == "Server-Side Request Forgery (SSRF)"
        assert doc.source_url == "https://cwe.mitre.org/data/definitions/918.html"
        assert doc.license == TEST_LICENSE
        assert doc.content_origin == "source_text"

    def test_provenance_describes_the_real_artifact(self, tmp_path):
        archive = cwe_extract_archive(tmp_path)
        prov = parse(archive, [SSRF])[0].provenance
        assert prov.origin_kind == "local_file"
        assert prov.artifact_path == "cwec_v4.20.xml.zip"
        assert prov.artifact_sha256 == archive.sha256
        assert prov.artifact_sha256 != "0" * 64
        assert (prov.adapter, prov.adapter_version) == (ADAPTER_NAME, ADAPTER_VERSION)
        assert prov.retrieved_at_utc == RETRIEVED_AT
        assert prov.ingested_at_utc == INGESTED_AT

    def test_source_modified_is_the_latest_modification_date_not_a_clock(self, docs):
        for doc, record_id in zip(docs, ("918", "1187")):
            dates = re.findall(r"<Modification_Date>([0-9-]+)</Modification_Date>", record_slice(record_id))
            assert dates
            assert doc.source_modified == max(dates)

    def test_sections_and_distinctive_text_are_preserved(self, docs):
        content = docs[0].content
        assert content.startswith("# CWE-918: Server-Side Request Forgery (SSRF)\n")
        assert "Status: Incomplete" in content
        headings = re.findall(r"^## (.+)$", content, re.MULTILINE)
        assert headings == [
            "Description", "Alternate Terms", "Modes of Introduction", "Applicable Platforms",
            "Common Consequences", "Detection Methods", "Relationships",
            "Related Attack Patterns", "Notes", "References",
        ]
        assert "Cross Site Port Attack" in content and "XSPA" in content
        assert "ChildOf CWE-441 (view 1000, ordinal Primary)" in content
        assert "- CAPEC-664" in content
        assert "xhtml" not in content and "<" not in content

    def test_missing_source_sections_are_omitted_not_invented(self, docs):
        # The v4.20 record CWE-918 has no Potential_Mitigations; none may appear.
        assert "<Potential_Mitigations" not in record_slice("918")
        assert "Potential Mitigations" not in docs[0].content
        assert "Extended Description" not in docs[0].content

    def test_relationships_and_references_are_explicit_source_facts(self, docs):
        refs = {(r.relation, r.target_source, r.target_source_id): r.url for r in docs[0].references}
        assert refs[("ChildOf", "cwe", "CWE-441")] == "https://cwe.mitre.org/data/definitions/441.html"
        assert ("ChildOf", "cwe", "CWE-610") in refs
        assert refs[("RelatedAttackPattern", "capec", "CAPEC-664")] == "https://capec.mitre.org/data/definitions/664.html"
        assert refs[("Reference", "CWE-REF", "REF-1511")].endswith("Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html")
        # CWE-441 / CWE-610 / CAPEC-664 need not be ingested for the reference to be kept.
        assert len(docs[0].references) == 3 + 10  # 2 ChildOf + 1 CAPEC + 10 External_Reference ids

    def test_deprecated_status_is_preserved(self, docs):
        doc = docs[1]
        assert doc.title.startswith("DEPRECATED:")
        assert "Status: Deprecated" in doc.content
        assert "deprecated because it was a duplicate of CWE-908" in doc.content

    def test_documents_pass_schema_revalidation(self, docs):
        for doc in docs:
            assert SourceDocument.model_validate(doc.model_dump(mode="json")) == doc

    def test_identity_and_hashes_are_stable_across_repeated_parsing(self, tmp_path):
        archive = cwe_extract_archive(tmp_path)
        first = parse(archive, [SSRF, DEPRECATED])
        second = parse(archive, [SSRF, DEPRECATED])
        assert first == second
        assert [d.document_sha256 for d in first] == [d.document_sha256 for d in second]
        assert [d.doc_id for d in first] == ["cwe:CWE-918", "cwe:CWE-1187"]

    def test_selection_order_is_the_output_order(self, tmp_path):
        docs = parse(cwe_extract_archive(tmp_path), [DEPRECATED, SSRF])
        assert [d.source_id for d in docs] == ["CWE-1187", "CWE-918"]


# ============================================================
# SYNTHETIC CONTENT
# ============================================================

def synth(tmp_path, record_body: str = "<Description>Synthetic description.</Description>", *, extra: str = "", selection=(SYNTH,)):
    # `extra` goes where the real format puts it: a sibling of <Weaknesses> under the catalog root.
    archive = cwe_archive(tmp_path, cwe_catalog_xml(f"<Weaknesses>{weakness(body=record_body)}</Weaknesses>{extra}"))
    return parse(archive, list(selection))


class TestSyntheticContent:
    def test_minimal_record_has_only_what_the_source_supplies(self, tmp_path):
        doc = synth(tmp_path)[0]
        assert doc.content == (
            "# CWE-1: Synthetic Weakness\n\nStatus: Draft\nAbstraction: Base\nStructure: Simple\n\n"
            "## Description\nSynthetic description."
        )
        assert doc.references == ()
        assert doc.source_modified is None

    def test_rich_text_is_flattened_readably(self, tmp_path):
        body = (
            "<Description><xhtml:p>First   paragraph\n   continues.</xhtml:p>"
            "<xhtml:p>Second paragraph:</xhtml:p>"
            "<xhtml:ul><xhtml:li>alpha</xhtml:li><xhtml:li>beta<xhtml:br/>gamma</xhtml:li></xhtml:ul></Description>"
        )
        doc = synth(tmp_path, body)[0]
        assert "## Description\nFirst paragraph continues.\n\nSecond paragraph:\n\n- alpha\n- beta\ngamma" in doc.content

    def test_consequences_mitigations_detection_and_introduction(self, tmp_path):
        body = (
            "<Description>D.</Description>"
            "<Modes_Of_Introduction><Introduction><Phase>Implementation</Phase><Note>Why.</Note></Introduction></Modes_Of_Introduction>"
            "<Common_Consequences><Consequence><Scope>Confidentiality</Scope><Scope>Integrity</Scope>"
            "<Impact>Read Application Data</Impact><Note>Leaks.</Note></Consequence></Common_Consequences>"
            "<Detection_Methods><Detection_Method><Method>Manual Analysis</Method>"
            "<Description>Review code.</Description><Effectiveness>Moderate</Effectiveness></Detection_Method></Detection_Methods>"
            "<Potential_Mitigations><Mitigation><Phase>Implementation</Phase><Strategy>Input Validation</Strategy>"
            "<Description><xhtml:p>Validate input.</xhtml:p><xhtml:p>Use allowlists.</xhtml:p></Description></Mitigation></Potential_Mitigations>"
        )
        content = synth(tmp_path, body)[0].content
        assert "## Modes of Introduction\n- Implementation: Why." in content
        assert "- Scope: Confidentiality, Integrity | Impact: Read Application Data | Note: Leaks." in content
        assert "- Manual Analysis (effectiveness: Moderate): Review code." in content
        assert "- [Implementation; Input Validation] Validate input. Use allowlists." in content

    def test_related_weakness_and_attack_pattern_become_references(self, tmp_path):
        body = (
            "<Description>D.</Description>"
            '<Related_Weaknesses><Related_Weakness Nature="ChildOf" CWE_ID="74" View_ID="1000" Ordinal="Primary"/>'
            '<Related_Weakness Nature="PeerOf" CWE_ID="75" View_ID="1000"/></Related_Weaknesses>'
            '<Related_Attack_Patterns><Related_Attack_Pattern CAPEC_ID="66"/></Related_Attack_Patterns>'
        )
        doc = synth(tmp_path, body)[0]
        refs = {(r.relation, r.target_source, r.target_source_id) for r in doc.references}
        assert refs == {
            ("ChildOf", "cwe", "CWE-74"),
            ("PeerOf", "cwe", "CWE-75"),
            ("RelatedAttackPattern", "capec", "CAPEC-66"),
        }
        assert "- ChildOf CWE-74 (view 1000, ordinal Primary)" in doc.content
        assert "- PeerOf CWE-75 (view 1000)" in doc.content

    @pytest.mark.parametrize(
        "relation",
        [
            '<Related_Weaknesses><Related_Weakness CWE_ID="74"/></Related_Weaknesses>',  # no Nature
            '<Related_Weaknesses><Related_Weakness Nature="ChildOf" CWE_ID="abc"/></Related_Weaknesses>',
            '<Related_Weaknesses><Related_Weakness Nature="ChildOf"/></Related_Weaknesses>',
            '<Related_Attack_Patterns><Related_Attack_Pattern CAPEC_ID="x1"/></Related_Attack_Patterns>',
        ],
    )
    def test_malformed_relationships_fail(self, tmp_path, relation):
        with pytest.raises(SourceFormatError):
            synth(tmp_path, "<Description>D.</Description>" + relation)

    def test_taxonomy_mappings(self, tmp_path):
        body = (
            "<Description>D.</Description><Taxonomy_Mappings>"
            '<Taxonomy_Mapping Taxonomy_Name="WASC"><Entry_ID>19</Entry_ID><Entry_Name>SQL Injection</Entry_Name></Taxonomy_Mapping>'
            '<Taxonomy_Mapping Taxonomy_Name="PLOVER"><Entry_Name>SQL injection</Entry_Name></Taxonomy_Mapping>'
            "</Taxonomy_Mappings>"
        )
        doc = synth(tmp_path, body)[0]
        refs = {(r.relation, r.target_source, r.target_source_id) for r in doc.references}
        assert refs == {("TaxonomyMapping", "WASC", "19"), ("TaxonomyMapping", "PLOVER", "SQL injection")}
        assert "- WASC 19: SQL Injection" in doc.content and "- PLOVER: SQL injection" in doc.content

    def test_taxonomy_mappings_sharing_an_entry_id_collapse_in_references_but_not_in_text(self, tmp_path):
        # Observed in the official CWE-287 (ISA/IEC 62443): a documented fidelity limit.
        body = (
            "<Description>D.</Description><Taxonomy_Mappings>"
            '<Taxonomy_Mapping Taxonomy_Name="ISA/IEC 62443"><Entry_ID>Part 3-3</Entry_ID><Entry_Name>Req SR 1.1</Entry_Name></Taxonomy_Mapping>'
            '<Taxonomy_Mapping Taxonomy_Name="ISA/IEC 62443"><Entry_ID>Part 3-3</Entry_ID><Entry_Name>Req SR 1.2</Entry_Name></Taxonomy_Mapping>'
            "</Taxonomy_Mappings>"
        )
        doc = synth(tmp_path, body)[0]
        assert [(r.target_source, r.target_source_id) for r in doc.references] == [("ISA/IEC 62443", "Part 3-3")]
        assert "- ISA/IEC 62443 Part 3-3: Req SR 1.1" in doc.content
        assert "- ISA/IEC 62443 Part 3-3: Req SR 1.2" in doc.content

    def test_references_inside_excluded_demonstrative_examples_are_not_collected(self, tmp_path):
        body = (
            "<Description>D.</Description><Demonstrative_Examples><Demonstrative_Example><References>"
            '<Reference External_Reference_ID="REF-1"/></References></Demonstrative_Example></Demonstrative_Examples>'
        )
        doc = synth(tmp_path, body, extra=self.CITATIONS)[0]
        assert doc.references == () and "REF-1" not in doc.content

    def test_taxonomy_mapping_without_any_entry_fails(self, tmp_path):
        body = '<Description>D.</Description><Taxonomy_Mappings><Taxonomy_Mapping Taxonomy_Name="WASC"/></Taxonomy_Mappings>'
        with pytest.raises(SourceFormatError, match="neither Entry_ID nor Entry_Name"):
            synth(tmp_path, body)

    CITATIONS = (
        '<External_References><External_Reference Reference_ID="REF-1"><Author>A. Author</Author><Title>A Book</Title>'
        "<URL>https://example.org/book</URL></External_Reference>"
        '<External_Reference Reference_ID="REF-2"><Title>No Link</Title></External_Reference></External_References>'
    )

    def test_citations_resolve_through_the_external_references_table(self, tmp_path):
        body = (
            "<Description>D.</Description><References>"
            '<Reference External_Reference_ID="REF-1" Section="Chapter 1"/>'
            '<Reference External_Reference_ID="REF-1" Section="Chapter 2"/>'
            '<Reference External_Reference_ID="REF-2"/></References>'
        )
        doc = synth(tmp_path, body, extra=self.CITATIONS)[0]
        assert [(r.target_source_id, r.url) for r in doc.references] == [
            ("REF-1", "https://example.org/book"),
            ("REF-2", None),
        ]
        assert "- REF-1: A Book (A. Author), Chapter 1" in doc.content
        assert "- REF-1: A Book (A. Author), Chapter 2" in doc.content
        assert "- REF-2: No Link" in doc.content

    def test_dangling_citation_fails(self, tmp_path):
        body = '<Description>D.</Description><References><Reference External_Reference_ID="REF-9"/></References>'
        with pytest.raises(SourceFormatError, match="REF-9"):
            synth(tmp_path, body, extra=self.CITATIONS)

    def test_duplicate_citation_id_fails(self, tmp_path):
        duplicate = (
            '<External_References><External_Reference Reference_ID="REF-1"><Title>A</Title></External_Reference>'
            '<External_Reference Reference_ID="REF-1"><Title>B</Title></External_Reference></External_References>'
        )
        with pytest.raises(SourceFormatError, match="more than once"):
            synth(tmp_path, extra=duplicate)

    def test_latest_modification_date_wins_regardless_of_order(self, tmp_path):
        history = (
            "<Content_History><Submission><Submission_Date>2001-01-01</Submission_Date></Submission>"
            "<Modification><Modification_Date>2020-05-05</Modification_Date></Modification>"
            "<Modification><Modification_Date>2024-02-29</Modification_Date></Modification>"
            "<Modification><Modification_Date>2019-12-31</Modification_Date></Modification></Content_History>"
        )
        assert synth(tmp_path, "<Description>D.</Description>" + history)[0].source_modified == "2024-02-29"

    @pytest.mark.parametrize(
        "modification",
        [
            "<Modification><Modification_Date>2024-13-40</Modification_Date></Modification>",
            "<Modification><Modification_Date>yesterday</Modification_Date></Modification>",
            "<Modification><Modification_Comment>no date</Modification_Comment></Modification>",
        ],
    )
    def test_bad_modification_dates_fail(self, tmp_path, modification):
        body = f"<Description>D.</Description><Content_History>{modification}</Content_History>"
        with pytest.raises(SourceFormatError):
            synth(tmp_path, body)

    @pytest.mark.parametrize("body", ["", "<Description>   </Description>"])
    def test_missing_or_blank_description_fails(self, tmp_path, body):
        with pytest.raises(SourceFormatError, match="no Description"):
            synth(tmp_path, body)

    def test_unlisted_sections_are_ignored_and_never_leak_into_content(self, tmp_path):
        body = "<Description>D.</Description><Demonstrative_Examples><Demonstrative_Example>SECRETCODE</Demonstrative_Example></Demonstrative_Examples>"
        assert "SECRETCODE" not in synth(tmp_path, body)[0].content


# ============================================================
# STRUCTURE, NAMESPACE, VERSION
# ============================================================

def parse_xml(tmp_path, xml: bytes, **kwargs):
    return parse(cwe_archive(tmp_path, xml, **kwargs), [SYNTH])


class TestStructureAndVersion:
    def test_prefixed_namespace_is_handled_like_the_default_namespace(self, tmp_path):
        xml = (
            f'<?xml version="1.0"?><cwe:Weakness_Catalog xmlns:cwe="http://cwe.mitre.org/cwe-7" Name="CWE" '
            f'Version="4.20" Date="2026-04-30"><cwe:Weaknesses><cwe:Weakness ID="1" Name="Synthetic Weakness" Status="Draft">'
            f"<cwe:Description>Prefixed.</cwe:Description></cwe:Weakness></cwe:Weaknesses></cwe:Weakness_Catalog>"
        ).encode()
        assert "Prefixed." in parse_xml(tmp_path, xml)[0].content

    def test_wrong_root_namespace_is_rejected(self, tmp_path):
        xml = cwe_catalog_xml(f"<Weaknesses>{weakness()}</Weaknesses>", ns="http://capec.mitre.org/capec-3")
        with pytest.raises(SourceFormatError, match="unexpected root element"):
            parse_xml(tmp_path, xml)

    def test_wrong_root_element_is_rejected(self, tmp_path):
        with pytest.raises(SourceFormatError, match="unexpected root element"):
            parse_xml(tmp_path, cwe_catalog_xml("", root="Attack_Pattern_Catalog"))

    def test_wrong_catalog_name_is_rejected(self, tmp_path):
        xml = cwe_catalog_xml(f"<Weaknesses>{weakness()}</Weaknesses>", name="CAPEC")
        with pytest.raises(SourceFormatError, match="root Name"):
            parse_xml(tmp_path, xml)

    def test_records_in_a_foreign_namespace_are_not_found(self, tmp_path):
        body = f'<Weaknesses xmlns="http://example.org/other">{weakness()}</Weaknesses>'
        with pytest.raises(SourceFormatError, match="no container element"):
            parse_xml(tmp_path, cwe_catalog_xml(body))

    def test_unexpected_version_is_rejected(self, tmp_path):
        xml = cwe_catalog_xml(f"<Weaknesses>{weakness()}</Weaknesses>", version="4.21")
        with pytest.raises(SourceVersionError, match="4.21.*4.20"):
            parse_xml(tmp_path, xml)

    def test_unexpected_date_is_rejected(self, tmp_path):
        xml = cwe_catalog_xml(f"<Weaknesses>{weakness()}</Weaknesses>", date="2026-05-01")
        with pytest.raises(SourceVersionError, match="date"):
            parse_xml(tmp_path, xml)

    def test_missing_version_attribute_is_rejected(self, tmp_path):
        xml = (
            b'<?xml version="1.0"?><Weakness_Catalog xmlns="http://cwe.mitre.org/cwe-7" Name="CWE" Date="2026-04-30"/>'
        )
        with pytest.raises(SourceVersionError, match="no Version"):
            parse_xml(tmp_path, xml)

    def test_the_version_inside_the_file_beats_the_filename(self, tmp_path):
        # Named like 4.20, declares 4.19: must be refused.
        xml = cwe_catalog_xml(f"<Weaknesses>{weakness()}</Weaknesses>", version="4.19")
        with pytest.raises(SourceVersionError):
            parse(cwe_archive(tmp_path, xml, filename="cwec_v4.20.xml.zip"), [SYNTH])

    @pytest.mark.parametrize("xml", [b"", b"<not-xml", b"plain text", b"<a><b></a>"])
    def test_malformed_xml_is_rejected(self, tmp_path, xml):
        with pytest.raises(SourceFormatError, match="malformed XML"):
            parse_xml(tmp_path, xml)

    @pytest.mark.parametrize(
        "prolog",
        ['<!DOCTYPE x [<!ENTITY a "b">]>', '<!DOCTYPE Weakness_Catalog>', '<!ENTITY a "b">'],
    )
    def test_dtd_and_entity_declarations_are_refused(self, tmp_path, prolog):
        xml = prolog.encode() + cwe_catalog_xml(f"<Weaknesses>{weakness()}</Weaknesses>")
        with pytest.raises(SourceFormatError, match="DOCTYPE or ENTITY"):
            parse_xml(tmp_path, xml)

    def test_escaped_doctype_text_inside_content_is_fine(self, tmp_path):
        # Official CWE-611 text contains escaped markup like this; it is plain text, not a declaration.
        doc = synth(tmp_path, "<Description>Example &lt;!DOCTYPE foo&gt; is escaped text.</Description>")[0]
        assert "<!DOCTYPE foo>" in doc.content


# ============================================================
# SELECTION
# ============================================================

class TestSelection:
    def make(self, tmp_path, *records: str):
        return cwe_archive(tmp_path, cwe_catalog_xml(f"<Weaknesses>{''.join(records)}</Weaknesses>"))

    def test_missing_selected_id_fails(self, tmp_path):
        with pytest.raises(SelectionError, match="ID 2 is missing"):
            parse(self.make(tmp_path, weakness("1")), [SYNTH, SelectedRecord("2", "Other")])

    def test_duplicate_id_in_the_source_fails(self, tmp_path):
        with pytest.raises(SelectionError, match="ID 1 occurs 2 times"):
            parse(self.make(tmp_path, weakness("1"), weakness("1")), [SYNTH])

    def test_duplicate_id_in_the_selection_fails(self, tmp_path):
        with pytest.raises(SelectionError, match="repeats"):
            parse(self.make(tmp_path, weakness("1")), [SYNTH, SYNTH])

    def test_name_mismatch_fails(self, tmp_path):
        with pytest.raises(SelectionError, match="pinned name"):
            parse(self.make(tmp_path, weakness("1", name="Renamed")), [SYNTH])

    def test_status_mismatch_fails_when_a_status_is_pinned(self, tmp_path):
        pinned = SelectedRecord("1", "Synthetic Weakness", "Stable")
        with pytest.raises(SelectionError, match="pinned status"):
            parse(self.make(tmp_path, weakness("1", status="Deprecated")), [pinned])

    def test_all_problems_are_reported_together(self, tmp_path):
        selection = [SelectedRecord("1", "Wrong"), SelectedRecord("7", "Absent")]
        with pytest.raises(SelectionError) as info:
            parse(self.make(tmp_path, weakness("1")), selection)
        message = str(info.value)
        assert "pinned name" in message and "ID 7 is missing" in message

    def test_nothing_is_returned_when_one_record_is_bad(self, tmp_path):
        records = [weakness("1"), weakness("2", body="")]
        with pytest.raises(SourceFormatError):
            parse(self.make(tmp_path, *records), [SYNTH, SelectedRecord("2", "Synthetic Weakness")])

    def test_unselected_records_are_never_read(self, tmp_path):
        broken_but_unselected = weakness("99", body="")
        docs = parse(self.make(tmp_path, weakness("1"), broken_but_unselected), [SYNTH])
        assert [d.source_id for d in docs] == ["CWE-1"]

    def test_non_numeric_selected_id_is_refused(self):
        with pytest.raises(SelectionError):
            SelectedRecord("CWE-89", "Name")


# ============================================================
# ARCHIVE HANDLING
# ============================================================

class TestArchive:
    def test_missing_archive_fails_clearly(self, tmp_path):
        archive = cwe_extract_archive(tmp_path)
        with pytest.raises(ArchiveError, match="not found"):
            parse_cwe_archive(tmp_path / "absent.zip", archive.spec, [SSRF])

    def test_not_a_zip_fails(self, tmp_path):
        path = tmp_path / "fake.zip"
        path.write_bytes(b"this is not a zip file")
        spec = make_spec(hashlib.sha256(path.read_bytes()).hexdigest(), version="4.20", date="2026-04-30",
                         member="cwec_v4.20.xml", filename="fake.zip")
        with pytest.raises(ArchiveError, match="not a valid ZIP"):
            parse_cwe_archive(path, spec, [SSRF])

    def test_checksum_mismatch_is_actionable(self, tmp_path):
        archive = cwe_extract_archive(tmp_path)
        wrong = dataclasses.replace(archive.spec, archive_sha256="0" * 64)
        with pytest.raises(ChecksumMismatchError) as info:
            parse_cwe_archive(archive.path, wrong, [SSRF])
        message = str(info.value)
        assert "0" * 64 in message and archive.sha256 in message
        assert "official URL" in message and "update the manifest" in message

    def test_a_modified_archive_no_longer_matches_its_pin(self, tmp_path):
        archive = cwe_extract_archive(tmp_path)
        xml = (FIXTURE_DIR / "cwe_v4.20_extract.xml").read_bytes().replace(b"XSPA", b"XSPB")
        write_zip(archive.path, {"cwec_v4.20.xml": xml})  # same name, different content
        with pytest.raises(ChecksumMismatchError):
            parse_cwe_archive(archive.path, archive.spec, [SSRF])

    def test_sibling_non_xml_members_are_ignored(self, tmp_path):
        xml = cwe_catalog_xml(f"<Weaknesses>{weakness()}</Weaknesses>")
        archive = cwe_archive(tmp_path, xml, extra_members={"schema.xsd": b"<xs:schema/>", "NOTES.txt": b"x"})
        assert parse(archive, [SYNTH])[0].source_id == "CWE-1"

    def test_two_xml_members_are_ambiguous(self, tmp_path):
        xml = cwe_catalog_xml(f"<Weaknesses>{weakness()}</Weaknesses>")
        archive = cwe_archive(tmp_path, xml, extra_members={"other.xml": xml})
        with pytest.raises(ArchiveError, match="exactly one .xml member"):
            parse(archive, [SYNTH])

    def test_no_xml_member_fails(self, tmp_path):
        path = tmp_path / "a.zip"
        sha = write_zip(path, {"readme.txt": b"hello"})
        spec = make_spec(sha, version="4.20", date="2026-04-30", member="cwec_v4.20.xml", filename="a.zip")
        with pytest.raises(ArchiveError, match="exactly one .xml member"):
            parse_cwe_archive(path, spec, [SYNTH])

    def test_unexpected_member_name_fails(self, tmp_path):
        xml = cwe_catalog_xml(f"<Weaknesses>{weakness()}</Weaknesses>")
        archive = cwe_archive(tmp_path, xml, member="renamed.xml")
        wrong_pin = make_spec(archive.sha256, version="4.20", date="2026-04-30",
                              member="cwec_v4.20.xml", filename="cwec_v4.20.xml.zip")
        with pytest.raises(ArchiveError, match="expected XML member"):
            parse_cwe_archive(archive.path, wrong_pin, [SYNTH])

    @pytest.mark.parametrize("name", ["../evil.xml", "sub/cwec.xml", "/abs.xml", "C:evil.xml", "a\\b.xml"])
    def test_unsafe_member_names_are_refused_not_extracted(self, tmp_path, name):
        xml = cwe_catalog_xml(f"<Weaknesses>{weakness()}</Weaknesses>")
        path = tmp_path / "unsafe.zip"
        sha = write_zip(path, {name: xml})
        spec = make_spec(sha, version="4.20", date="2026-04-30", member=name, filename="unsafe.zip")
        with pytest.raises(ArchiveError, match="unsafe member name"):
            parse_cwe_archive(path, spec, [SYNTH])
        assert not (tmp_path.parent / "evil.xml").exists()

    def test_oversized_member_is_refused(self, tmp_path, monkeypatch):
        archive = cwe_extract_archive(tmp_path)
        monkeypatch.setattr(mitre_xml, "MAX_XML_BYTES", 100)
        with pytest.raises(ArchiveError, match="declares"):
            parse(archive, [SSRF])

    def test_spec_rejects_a_malformed_checksum(self):
        with pytest.raises(ArchiveError, match="64 lowercase"):
            make_spec("ABC", version="4.20", date="2026-04-30", member="m.xml", filename="f.zip")

    def test_zip_directory_entries_do_not_count_as_members(self, tmp_path):
        xml = cwe_catalog_xml(f"<Weaknesses>{weakness()}</Weaknesses>")
        path = tmp_path / "d.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("docs/", b"")
            zf.writestr("cwec_v4.20.xml", xml)
        spec = make_spec(hashlib.sha256(path.read_bytes()).hexdigest(), version="4.20", date="2026-04-30",
                         member="cwec_v4.20.xml", filename="d.zip")
        assert parse_cwe_archive(path, spec, [SYNTH])[0].source_id == "CWE-1"
