"""
Build or update AutoShield's pinned MITRE reference corpus in a SQLite store.

Example (Windows, from the repository root):

    .\\.venv\\Scripts\\python.exe -m scripts.ingest_mitre_corpus ^
        --cwe-archive data\\knowledge_sources\\raw\\cwec_v4.20.xml.zip ^
        --capec-archive data\\knowledge_sources\\raw\\capec_v3.9.zip ^
        --database data\\knowledge_db\\mitre_corpus.sqlite

What it does: loads data/knowledge_sources/manifest.json; verifies each
archive's SHA-256 and the version/date declared inside its XML; parses ONLY the
curated records; then ingests them through the existing pipeline (canonical
chunker -> KnowledgeStore, FTS5 keyword index). It works on local files only
and never touches the network.

The raw archives are not in the repository. Download them yourself from the
official MITRE locations pinned in the manifest, e.g. in PowerShell:

    Invoke-WebRequest https://cwe.mitre.org/data/xml/cwec_v4.20.xml.zip -OutFile data\\knowledge_sources\\raw\\cwec_v4.20.xml.zip
    Invoke-WebRequest https://capec.mitre.org/data/archive/capec_v3.9.zip -OutFile data\\knowledge_sources\\raw\\capec_v3.9.zip

If a file does not match the pinned checksum, the build fails; it never
substitutes another release.

Exit status: 0 only when the complete corpus was built; 1 on any failure
(manifest, archive, parsing, store or ingestion); 2 for command-line misuse.

Batch semantics: both archives are fully verified and parsed BEFORE any write.
After that each document is its own store transaction (the batch is not
atomic). If a write fails midway, documents already written stay committed and
the failure is reported as INCOMPLETE; simply rerun the same command, which is
safe: finished documents come back "unchanged".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:  # lets `python scripts\ingest_mitre_corpus.py` work too
    sys.path.insert(0, str(_REPO_ROOT))

from ingestion.mitre_corpus import (  # noqa: E402
    CorpusManifest,
    CorpusReport,
    build_corpus_documents,
    ingest_corpus_documents,
    load_manifest,
)
from ingestion.mitre_xml import MitreSourceError  # noqa: E402
from ingestion.pipeline import BatchIngestionError, IngestionError  # noqa: E402
from knowledge.store import KnowledgeStore, KnowledgeStoreError  # noqa: E402

DEFAULT_MANIFEST = _REPO_ROOT / "data" / "knowledge_sources" / "manifest.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.ingest_mitre_corpus",
        description="Verify the pinned MITRE CWE/CAPEC archives and ingest the curated records.",
    )
    parser.add_argument("--cwe-archive", required=True, type=Path, help="path to cwec_v4.20.xml.zip")
    parser.add_argument("--capec-archive", required=True, type=Path, help="path to capec_v3.9.zip")
    parser.add_argument("--database", required=True, type=Path, help="SQLite file to create or update")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="pinned corpus manifest (default: data/knowledge_sources/manifest.json)",
    )
    return parser


def _print_header(manifest: CorpusManifest, args: argparse.Namespace) -> None:
    print("AutoShield MITRE corpus build")
    print(f"  manifest : {manifest.path}")
    for entry in (manifest.cwe, manifest.capec):
        print(
            f"  {entry.key.upper():5s} pinned version {entry.spec.source_version} "
            f"(catalog date {entry.spec.catalog_date}), {len(entry.selection)} curated record(s)"
        )
        print(f"        sha256 {entry.spec.archive_sha256}")
    print(f"  database : {args.database}")


def _print_success(report: CorpusReport, store: KnowledgeStore) -> None:
    summary = report.summary
    print()
    print("Result: OK - complete corpus build")
    print(f"  CWE version {report.cwe_version}, CAPEC version {report.capec_version}")
    print(
        f"  selected {report.selected_cwe + report.selected_capec} "
        f"({report.selected_cwe} CWE + {report.selected_capec} CAPEC); ingested {report.ingested}"
    )
    print(
        f"  created {len(summary.created)}, updated {len(summary.updated)}, "
        f"unchanged {len(summary.unchanged)}, duplicates skipped {len(summary.duplicates_skipped)}"
    )
    print(f"  store now holds {store.count_documents()} document(s), {store.count_chunks()} chunk(s)")
    print("  errors: none")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
    except MitreSourceError as exc:
        print(f"Result: FAILED - manifest error: {exc}", file=sys.stderr)
        return 1
    _print_header(manifest, args)

    try:
        # Verify checksums/versions and parse BOTH archives before the database
        # is opened, so a bad archive never creates or touches the database file.
        documents = build_corpus_documents(
            manifest, cwe_archive=args.cwe_archive, capec_archive=args.capec_archive
        )
        with KnowledgeStore(args.database) as store:
            report = ingest_corpus_documents(manifest, documents, store=store)
            _print_success(report, store)
    except BatchIngestionError as exc:
        done = exc.completed
        print(
            f"\nResult: INCOMPLETE - the corpus build did NOT finish: {exc}\n"
            f"  committed before the failure: created {len(done.created)}, "
            f"updated {len(done.updated)}, unchanged {len(done.unchanged)}\n"
            "  rerunning the same command is safe",
            file=sys.stderr,
        )
        return 1
    except (MitreSourceError, IngestionError, KnowledgeStoreError, OSError) as exc:
        print(f"\nResult: FAILED - {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
