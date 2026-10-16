# Annotation guide for the retrieval benchmark v2

Applies to `retrieval_benchmark_v2.json`. Version 1 (`retrieval_benchmark_v1.json`) is kept unchanged.

## What you are labelling

For each question, decide which documents of the 23-record corpus a good retrieval system should return.
The corpus is a small curated subset of MITRE CWE 4.20 (weaknesses) and CAPEC 3.9 (attack patterns); the
valid document IDs are listed in `annotation_sheet_v2.json` (`corpus_documents`). Judge from the record content
(title, description, prerequisites, execution flow, indicators, consequences, detection, mitigations). Do **not**
run any search system, and do not look at any retrieval result, before labelling.

## A document is RELEVANT when

1. Its **central subject** (what the record is about) is the behaviour, weakness, attack technique or guidance
   the question asks about, **or**
2. The question asks about one **facet** of a topic (consequences, prerequisites, execution flow, indicators,
   detection, mitigation) and the record contains that facet **for that same topic**.

## A document is NOT relevant when

- It is only a parent, child or sibling of the right record and the question does not ask for the general or
  related class. (Example: a question about the general restricted-directory weakness does not make the specific
  relative-path record relevant by itself; a question that asks for the specific variant does not make the
  general class relevant.)
- It mentions a word or technique only in passing (for example a denial-of-service consequence listed as a side
  effect of a record about something else).
- It shares vocabulary with the question but concerns a different weakness or attack.

## Rules

- Use only the listed document IDs. List **every** relevant document; most questions have one to three.
- A question about an **identifier** (for example a record number) is relevant to the record(s) named.
- If you believe **no** document is relevant (the topic is not in the corpus), say so explicitly with an empty list.
- Do not guess from outside knowledge about records that are not in the corpus.
- Add a short note if a case was a close call.

## Review process recorded in the benchmark

Each case stores `author_labels` (the author's original judgement), the list of `reviews` (reviewer identity
and method, labels given without seeing the author's labels or any retrieval result), an `adjudication` (final
label and how it was decided) and a `review_status`:

| review_status | meaning |
|---|---|
| `author_only_awaiting_review` | only the author has labelled it |
| `ai_review_agreed` | a separate AI review agreed exactly with the author |
| `ai_review_adjudicated` | the AI review disagreed; the final label was decided by the stated adjudication method |
| `human_review_agreed` / `human_review_adjudicated` | same, for a human reviewer (**none has taken place yet**) |
| `not_applicable_no_labels` | behaviour cases without labels |

An AI review is **not** an independent human annotation. Disagreements are kept in the file, not erased.
A human reviewer can use `annotation_sheet_v2.json` (questions only, no author labels) and the corpus records.

## Amendments recorded after review pass 1 (before any v2 retrieval was run)

The first review pass (an AI reviewer working from the guide above) disagreed with the author on 7 of 53 items.
The author adjudicated them using the clauses below. Because the author (also an AI) adjudicated, this is **not**
independent; every adjudicated case is marked so that a human can re-decide it. Clauses 1-2 were not in the
guide text the reviewer saw.

1. **Cross-lane links.** A record that the source itself lists as the Related Weakness or Related Attack Pattern
   of a labelled record, and that describes the same behaviour (for example blind SQL injection and the SQL
   injection weakness), stays relevant. The system is meant to return weakness and attack-pattern knowledge
   together. Cases kept under this clause are flagged `contested`, and the evaluator can also score the strict
   "agreed by author and every reviewer" label set as a sensitivity check.
2. **Conjunctions.** If a question asks for several things together ("library calls and a sandbox"), a record
   is relevant only if it contains all of them. If a facet is stated in only one record's text, only that record
   is relevant (checked by searching the full source records, not only the digest).
3. **Evidence hidden by truncation.** If the reviewer could not see the evidence (the digest was truncated),
   the full source record decides.
