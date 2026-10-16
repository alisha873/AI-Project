# Attribution and licence notice — MITRE CWE and CAPEC reference data

This notice covers the reference data AutoShield's security knowledge base builds from two MITRE resources.
It sits next to `manifest.json`, which pins the exact releases and records the checksums.
Keep this file with any copy of the corpus or of data derived from it.

AutoShield is not affiliated with, sponsored by or endorsed by The MITRE Corporation. The records are
**reformatted** (rich text flattened, selected sections only, see the adapter docstrings in `ingestion/mitre_cwe.py`
and `ingestion/mitre_capec.py`) and only a **curated subset** is stored (`manifest.json`,
`"is_complete_catalogue": false`). The stored text is therefore a derivative of the official records and
is not a substitute for them. Consult the official source for authoritative content.

## CWE™ — Common Weakness Enumeration

- Release used: **CWE List Version 4.20** (XML root `Date="2026-04-30"`).
- Official downloads: <https://cwe.mitre.org/data/downloads.html>
- Official archive of releases: <https://cwe.mitre.org/data/archive.html>
- Pinned archive: <https://cwe.mitre.org/data/xml/cwec_v4.20.xml.zip> (checksum in `manifest.json`)
- Terms of Use: <https://cwe.mitre.org/about/termsofuse.html> (read on 2026-10-16)

Copyright designation shown on the official site when the archive was retrieved:

> Copyright © 2006–2026, The MITRE Corporation. CWE, CWSS, CWRAF, and the CWE logo are trademarks of The MITRE Corporation.

Licence grant, quoted from the Terms of Use page:

> CWE™ is free to use by any organization or individual for any research, development, and/or commercial purposes, per these CWE Terms of Use. Accordingly, The MITRE Corporation hereby grants you a non-exclusive, royalty-free license to use CWE for research, development, and commercial purposes. Any copy you make for such purposes is authorized on the condition that you reproduce MITRE’s copyright designation and this license in any such copy. CWE is a trademark of The MITRE Corporation. Please contact cwe@mitre.org if you require further clarification on this issue.

Disclaimer, quoted in part from the same page:

> By accessing information through this site you (as “the user”) hereby agrees the site and the information is provided on an “as is” basis only without warranty of any kind, express or implied, including but not limited to implied warranties of merchantability, availability, accuracy, noninfringement, or fitness for a particular purpose. […] The user shall comply with all applicable laws, rules, and regulations, and the data source’s restrictions, when using the site.

## CAPEC™ — Common Attack Pattern Enumeration and Classification

- Release used: **CAPEC List Version 3.9** (XML root `Date="2023-01-24"`).
- Official downloads: <https://capec.mitre.org/data/downloads.html>
- Official archive of releases: <https://capec.mitre.org/data/archive.html>
- Pinned archive: <https://capec.mitre.org/data/archive/capec_v3.9.zip> (checksum in `manifest.json`)
- Terms of Use: <https://capec.mitre.org/about/termsofuse.html> (read on 2026-10-16)

Copyright designation shown on the official site when the archive was retrieved:

> Copyright © 2007–2026, The MITRE Corporation. CAPEC and the CAPEC logo are trademarks of The MITRE Corporation.

Licence grant, quoted from the Terms of Use page:

> The MITRE Corporation (MITRE) hereby grants you a non-exclusive, royalty-free license to use Common Attack Pattern Enumeration and Classification (CAPEC™) for research, development, and commercial purposes. Any copy you make for such purposes is authorized provided that you reproduce MITRE’s copyright designation and this license in any such copy.

Disclaimer, quoted from the same page:

> ALL DOCUMENTS AND THE INFORMATION CONTAINED THEREIN ARE PROVIDED ON AN "AS IS" BASIS AND THE CONTRIBUTOR, THE ORGANIZATION HE/SHE REPRESENTS OR IS SPONSORED BY (IF ANY), THE MITRE CORPORATION, ITS BOARD OF TRUSTEES, OFFICERS, AGENTS, AND EMPLOYEES, DISCLAIM ALL WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTY THAT THE USE OF THE INFORMATION THEREIN WILL NOT INFRINGE ANY RIGHTS OR ANY IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.

## Where the attribution lives

The terms require that **any copy reproduces MITRE’s copyright designation and the licence**. To make that hard to lose:

1. **This file** holds the designation and the licence text above.
2. **`manifest.json`** holds, for each source, the version, official URLs, the terms URL, the archive checksum
   and the `license` string that is copied onto every document.
3. **Every stored document** carries that `license` string (copyright designation, terms URL, pointer to this
   file), its per-record official `source_url`, and a provenance record naming the pinned archive and its SHA-256.
4. **The test fixtures** in `tests/fixtures/mitre/` begin with an attribution comment.

**Open point (not decided here):** the per-document `license` field holds the designation and the terms URL,
not the full licence text. Whether that alone satisfies the “reproduce … this license” condition for every
kind of redistribution was **not** determined. Anything that redistributes the stored text (a database export,
a published report, a dashboard) should include this file as well. This notice is not legal advice.

## Third-party material and other caveats

- CWE and CAPEC records cite **external references** (books, papers, web pages) and **taxonomy mappings** (for example
  WASC, OWASP, SEI CERT). AutoShield stores only identifiers, titles, authors, section hints and URLs of those
  items as the source lists them, never the cited works themselves. The MITRE terms pages quoted above do not
  state terms for those third-party works, so **no claim** is made about them; their own terms apply.
  Some record text may itself quote or paraphrase third-party material (the CWE page speaks of “the data source’s
  restrictions”). Whether any such passage has separate terms was not examined and is **left ambiguous**.
- **Record URLs** follow the official patterns `https://cwe.mitre.org/data/definitions/<id>.html` and
  `https://capec.mitre.org/data/definitions/<id>.html`. They were checked (HTTP 200) on 2026-10-16 for sample
  records. They address the **current** site, which may show a newer release than the pinned 4.20 / 3.9 text.
- The **CAPEC archive page** (`/data/archive.html`) rendered with server-side-include errors on 2026-10-16 and did not list
  versioned downloads. The pinned CAPEC URL follows the `/data/archive/capec_v<version>.zip` pattern and was
  confirmed by downloading the file and reading `Version="3.9"` from the XML itself, not from the filename.
- Deprecated or otherwise non-standard `Status` values reported by the source (for example Deprecated, Obsolete,
  Draft, Incomplete) are kept in each document’s text; deprecated entries keep their “DEPRECATED:” name prefix.
- Records are reference knowledge. A weakness or attack pattern being present in this corpus is not evidence that
  it occurred in any incident, and correlation is not attribution.

## Obtaining the raw archives

The raw archives are **not** stored in version control (`data/knowledge_sources/raw/` is git-ignored).
Download them from the official locations above, then verify them with the build command, which refuses any file
whose SHA-256 differs from `manifest.json`. See `scripts/ingest_mitre_corpus.py` for the exact command.
