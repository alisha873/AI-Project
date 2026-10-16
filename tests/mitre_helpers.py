"""
Shared helpers for the MITRE adapter and corpus tests (not a test module).

Two kinds of test input are used, and they are labelled accordingly:
- SYNTHETIC XML built here (`*_catalog_xml`, `weakness`, `attack_pattern`):
  invented, minimal and valid or deliberately broken. NOT official records.
- VERBATIM EXTRACTS of the official MITRE v4.20 CWE / v3.9 CAPEC XML in
  `tests/fixtures/mitre/` (see the attribution comment inside each file).
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import NamedTuple

from ingestion.mitre_xml import ArtifactSpec

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mitre"
CWE_NS = "http://cwe.mitre.org/cwe-7"
CAPEC_NS = "http://capec.mitre.org/capec-3"
XHTML_NS = "http://www.w3.org/1999/xhtml"
RETRIEVED_AT = "2026-10-15T22:53:27Z"
INGESTED_AT = "2026-10-15T22:53:30Z"
TEST_LICENSE = "TEST LICENSE: MITRE Terms of Use (test stand-in)"


class Archive(NamedTuple):
    path: Path
    spec: ArtifactSpec
    sha256: str


def write_zip(path: Path, members: dict[str, bytes]) -> str:
    """Write a ZIP with the given members and return the SHA-256 of the ZIP file."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_spec(
    sha256: str, *, version: str, date: str, member: str, filename: str, license: str = TEST_LICENSE
) -> ArtifactSpec:
    return ArtifactSpec(
        source_version=version,
        catalog_date=date,
        archive_sha256=sha256,
        xml_member=member,
        artifact_filename=filename,
        license=license,
        retrieved_at_utc=RETRIEVED_AT,
        ingested_at_utc=INGESTED_AT,
    )


def _catalog(root: str, name: str, ns: str, version: str, date: str, body: str) -> bytes:
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<{root} xmlns="{ns}" xmlns:xhtml="{XHTML_NS}" Name="{name}" Version="{version}" Date="{date}">'
        f"{body}</{root}>"
    ).encode("utf-8")


def cwe_catalog_xml(
    body: str, *, version: str = "4.20", date: str = "2026-04-30", ns: str = CWE_NS,
    root: str = "Weakness_Catalog", name: str = "CWE",
) -> bytes:
    return _catalog(root, name, ns, version, date, body)


def capec_catalog_xml(
    body: str, *, version: str = "3.9", date: str = "2023-01-24", ns: str = CAPEC_NS,
    root: str = "Attack_Pattern_Catalog", name: str = "CAPEC",
) -> bytes:
    return _catalog(root, name, ns, version, date, body)


def weakness(
    record_id: str = "1", name: str = "Synthetic Weakness", status: str = "Draft",
    body: str = "<Description>Synthetic description.</Description>",
) -> str:
    return f'<Weakness ID="{record_id}" Name="{name}" Abstraction="Base" Structure="Simple" Status="{status}">{body}</Weakness>'


def attack_pattern(
    record_id: str = "1", name: str = "Synthetic Pattern", status: str = "Draft",
    body: str = "<Description>Synthetic description.</Description>",
) -> str:
    return f'<Attack_Pattern ID="{record_id}" Name="{name}" Abstraction="Standard" Status="{status}">{body}</Attack_Pattern>'


def cwe_archive(
    tmp_path: Path, xml: bytes, *, version: str = "4.20", date: str = "2026-04-30",
    member: str = "cwec_v4.20.xml", filename: str = "cwec_v4.20.xml.zip",
    extra_members: dict[str, bytes] | None = None,
) -> Archive:
    """A CWE-style ZIP holding `xml` (plus optional extra members) and a matching spec."""
    path = tmp_path / filename
    sha = write_zip(path, {member: xml, **(extra_members or {})})
    return Archive(path, make_spec(sha, version=version, date=date, member=member, filename=filename), sha)


def capec_archive(
    tmp_path: Path, xml: bytes, *, version: str = "3.9", date: str = "2023-01-24",
    member: str = "capec_v3.9.xml", filename: str = "capec_v3.9.zip",
    extra_members: dict[str, bytes] | None = None,
) -> Archive:
    path = tmp_path / filename
    sha = write_zip(path, {member: xml, **(extra_members or {})})
    return Archive(path, make_spec(sha, version=version, date=date, member=member, filename=filename), sha)


def cwe_extract_archive(tmp_path: Path) -> Archive:
    """A ZIP built from the verbatim official CWE v4.20 extract (CWE-918, CWE-1187)."""
    return cwe_archive(tmp_path, (FIXTURE_DIR / "cwe_v4.20_extract.xml").read_bytes())


def capec_extract_archive(tmp_path: Path) -> Archive:
    """A ZIP built from the verbatim official CAPEC v3.9 extract (CAPEC-664, -137, -106, -242)."""
    return capec_archive(tmp_path, (FIXTURE_DIR / "capec_v3.9_extract.xml").read_bytes())
