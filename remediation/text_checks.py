"""
Phase 12 - text-level screens for model-written remediation statements.

These are deterministic regular-expression screens. They catch a short list of high-risk patterns; they
do NOT understand meaning, can miss paraphrases and can flag unusual wording. Passing them proves nothing
about correctness. They are heuristic safeguards, not a test of whether a statement is entailed by the
evidence, and no false-positive or false-negative rate has been measured for them.

How hedging is handled (compromise and "presented as confirmed" screens)
------------------------------------------------------------------------
A match is allowed when it is negated ("no evidence that the database was breached") or when it sits in
a clause that explicitly frames it as a possibility or an open question: if, whether, unless, may,
might, could, would, possible, potential, suspect, hypothesis, unconfirmed, unknown, unclear. The clause
is the text since the last comma, full stop, semicolon or colon. This is a keyword window, not
understanding: "the database was breached, which may be unconfirmed" is rejected (the hedge comes
after), and a hedge word in the same clause can excuse an unrelated claim. Success claims are NOT
excused by hedging words (a hedged "this will prevent all attacks" is still an overclaim); only explicit
negation ("this does not guarantee that ...") excuses them.

Reused from Phase 11D (`rag.citations`, imported rather than copied so the two phases cannot drift):
the evidence/reference token patterns, the CWE/CAPEC/CVE identifier extractor, the compromise-claim and
"presented as confirmed" patterns and the negation handling.

New in Phase 12:
- remediation-success claims ("this fix guarantees...", "the vulnerability has been fixed"),
- ungrounded literals: a path or a back-ticked command/setting that does not occur anywhere in the
  material the model was shown. This is how invented configuration paths, commands and asset names are
  refused. A command written without back-ticks or a path-less asset name is NOT detected (limitation).
"""

from __future__ import annotations

import re

from rag.citations import (
    _CONFIRMATION,
    _EVIDENCE_TOKEN_RE,
    _HIGH_IMPACT,
    _PATH_RE,
    _REFERENCE_TOKEN_RE,
    _identifiers,
    _is_negated,
)

CONFIG_TOKEN_RE = re.compile(r"\bCFG-[0-9]+\b")
_BACKTICK_RE = re.compile(r"`([^`\n]{1,200})`")

_SUCCESS_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bguarantee[sd]?\b|\bguaranteeing\b",
        r"\b(?:fully|completely|entirely|totally|permanently)\s+(?:eliminat|prevent|mitigat|remediat|resolv|fix|secur|protect|block|remov|neutrali|stop)\w*",
        r"\belimina(?:te|tes|ted|ting)\s+(?:the\s+|this\s+|that\s+|all\s+|any\s+|every\s+|entire\s+)?(?:vulnerabilit\w+|weakness\w*|risk\w*|threat\w*|attack\w*)\b",
        r"\b(?:has|have|had)\s+been\s+(?:successfully\s+)?(?:fixed|remediated|patched|mitigated|resolved|eliminated|secured|closed)\b",
        r"\b(?:is|are)\s+now\s+(?:secure|safe|fixed|patched|protected|mitigated|remediated)\b",
        r"\bno\s+longer\s+(?:vulnerable|exploitable|at\s+risk)\b",
        r"\bsuccessfully\s+(?:remediat|fix|patch|mitigat|resolv)\w*",
        r"\bremediation\s+(?:succeeded|was\s+successful|is\s+complete|was\s+completed)\b",
        r"\b(?:immune|invulnerable)\b",
        r"\bzero\s+(?:residual\s+)?risk\b",
        r"\b100\s*%\s*(?:secure|safe|protected|effective)",
    )
)
_NEGATING_VERBS = r"(?:guarantee|ensure|imply|mean|prove|show|establish|confirm|demonstrate|claim|assure|promise|state|say)"
_EXTRA_NEGATION_BEFORE = re.compile(
    r"(?:n't|never|nor)\s+(?:\w+\s+)?$"
    # "does not guarantee that <claim>", "cannot say whether <claim>", "can't confirm <claim>"
    rf"|(?:(?:does|do|did|would|will|should)\s+not|cannot|can\s+not|could\s+not|may\s+not|might\s+not)\s+"
    rf"{_NEGATING_VERBS}\w*\s+(?:(?:that|whether|if)\s+)?(?:\w+\s+){{0,3}}$"
    # "no evidence shows that <claim>", "no sign that <claim>"
    r"|\bno\s+(?:evidence|indication|sign|proof)s?\s+(?:\w+\s+){0,3}$"
    # "no claim that <claim>", "not guaranteed that <claim>"
    r"|\bno\s+(?:claim|guarantee|assurance|promise)\s+(?:is\s+made\s+|can\s+be\s+made\s+)?(?:that|of)\s+(?:\w+\s+){0,3}$"
    r"|\bnot\s+(?:been\s+)?(?:guaranteed|claimed|proven|shown|established|demonstrated)\s+(?:that\s+|to\s+)?(?:\w+\s+){0,3}$",
    re.IGNORECASE,
)

# A clause that frames what follows as a possibility, suspicion or open question (see the module docstring).
_HEDGE = re.compile(
    r"\b(?:if|whether|unless|suspect(?:ed|s|ing)?|possib(?:le|ly|ility)|potential(?:ly)?|might|may|could|would"
    r"|hypothes[ie]s|hypothesi[sz]ed|unconfirmed|unverified|unclear|unknown|uncertain|alleged(?:ly)?"
    r"|not\s+(?:known|clear|certain))\b",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(r"[.;:!?,\n]")

_COMPROMISED_THINGS = (
    r"(?:database|databases|db|data|records?|rows?|tables?|credentials?|passwords?|accounts?|server|servers"
    r"|host|hosts|application|app|site|website|network|system|systems|endpoint|service|sessions?|files?|information)"
)
# Compromise claims added in Phase 12 (the Phase 11D list is reused unchanged, see `compromise_claims`).
_MORE_COMPROMISE = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "The database was breached.", "Records were exfiltrated.", "The table was dumped.", "data has been stolen"
        rf"\b{_COMPROMISED_THINGS}\s+(?:was|were|has\s+been|have\s+been|got)\s+(?:successfully\s+|already\s+|fully\s+)?"
        r"(?:breached|hacked|compromised|exfiltrated|stolen|leaked|dumped|extracted|taken\s+over|infiltrated|penetrated|defaced|wiped)\b",
        # "The attacker dumped the users table.", "an intruder gained access", "the attackers exfiltrated ..."
        r"\b(?:the\s+|an?\s+)?(?:attackers?|intruders?|adversar(?:y|ies)|hackers?|threat\s+actors?)\s+"
        r"(?:successfully\s+|then\s+|later\s+)?"
        r"(?:dumped|exfiltrated|stole|extracted|downloaded|obtained|gained|compromised|breached|exploited|bypassed|escalated|deleted|tampered)\b",
        # "The attack succeeded.", "the injection attempts were successful", "the exploit worked"
        r"\b(?:attacks?|attempts?|exploit(?:ation)?|injections?|intrusion|breach)\s+"
        r"(?:was\s+|were\s+|has\s+been\s+|have\s+been\s+)?(?:successful(?:ly)?|succeeded|worked)\b",
        # "A breach occurred.", "exfiltration took place"
        r"\b(?:breach|compromise|intrusion|exfiltration|exploitation|attack)\s+(?:has\s+|had\s+)?(?:occurred|happened|took\s+place)\b",
        # "SQL injection was used to dump the database"
        r"\b(?:used|leveraged)\s+to\s+(?:dump|exfiltrate|steal|extract)\b",
    )
)

# Claims that a proposal works, added in Phase 12 (the earlier patterns stay in `_SUCCESS_PATTERNS`).
_MORE_SUCCESS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # "This will prevent all SQL injection.", "should eliminate every risk"
        r"\b(?:will|would|shall|does|can|should)\s+(?:completely\s+|fully\s+|entirely\s+|totally\s+|permanently\s+)?"
        r"(?:prevent|stop|block|eliminate|eradicate|remove|fix|solve|resolve|remediate|neutrali[sz]e|close)\s+"
        r"(?:all|every|any|each|the\s+entire)\b",
        # "This will prevent SQL injection attacks.", "will block attacks", "will fix the vulnerability"
        r"\b(?:will|would|shall)\s+(?:completely\s+|fully\s+)?"
        r"(?:prevent|stop|block|eliminate|eradicate|fix|solve|resolve|remediate|neutrali[sz]e)\s+"
        r"(?:the\s+|this\s+|that\s+|these\s+|such\s+|\w+\s+)?"
        r"(?:sql\s+injection|injection|attacks?|vulnerabilit\w+|exploit\w*|threats?|breach\w*|compromise|data\s+loss|exfiltration)\b",
        # "prevents all injection"
        r"\b(?:prevents?|stops?|blocks?|eliminates?|eradicates?)\s+(?:all|every)\b",
        # "This makes the application safe.", "keeps the endpoint secure" (not "makes logging more secure")
        r"\b(?:makes?|made|keeps?|renders?)\s+(?:(?:the|this|that|your|our|an?|it)\s+)?"
        r"(?:(?!more\b|less\b|much\b|far\b|too\b)[\w\-]+\s+){0,2}"
        r"(?:safe|secure|immune|invulnerable|unexploitable|impenetrable|bulletproof)\b",
        # "The endpoint is not exploitable.", "will be no longer vulnerable"
        r"\b(?:is|are|becomes?|remains?|will\s+be|will\s+become|would\s+be|would\s+become)\s+(?:no\s+longer|not)\s+"
        r"(?:exploitable|vulnerable|injectable|attackable|at\s+risk)\b",
        # "The endpoint cannot be exploited."
        r"\b(?:cannot|can\s?not|can't|could\s+not|couldn't)\s+(?:be\s+)?(?:exploited|attacked|compromised|breached|injected)\b",
    )
)

# Statements of confirmation added in Phase 12 ("it was established that the app is injectable").
_MORE_CONFIRMATION = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:established|demonstrated|proved|proven|verified|determined)\s+that\s+[^.;\n]{0,60}?"
        r"\b(?:vulnerable|injectable|exploitable|exploited|compromised)\b",
        r"\b(?:is|are|was|were)\s+(?:definitely|certainly|clearly|undoubtedly)\s+(?:vulnerable|injectable|exploitable)\b",
    )
)


def identifiers(text: str) -> set[str]:
    return _identifiers(text)


def evidence_tokens(text: str) -> set[str]:
    return set(_EVIDENCE_TOKEN_RE.findall(text))


def reference_tokens(text: str) -> set[str]:
    return set(_REFERENCE_TOKEN_RE.findall(text))


def configuration_tokens(text: str) -> set[str]:
    return set(CONFIG_TOKEN_RE.findall(text))


def _negated(text: str, start: int, end: int | None = None) -> bool:
    if _is_negated(text, start, end):
        return True
    return _EXTRA_NEGATION_BEFORE.search(text[max(0, start - 40):start]) is not None


def _hedged(text: str, start: int) -> bool:
    """True if the clause containing `start` frames the match as a possibility or open question."""
    prefix = text[max(0, start - 80):start]
    breaks = list(_CLAUSE_BREAK.finditer(prefix))
    clause = prefix[breaks[-1].end():] if breaks else prefix
    return _HEDGE.search(clause) is not None


def compromise_claims(text: str) -> list[str]:
    """
    Positive claims of exploitation, data access or compromise.

    Negated wording ("no evidence that ...") and clearly hedged wording ("it is possible that ...",
    "if records were exfiltrated, ...") are allowed. Keyword screen only; see the module docstring.
    """
    return [
        match.group(0)
        for pattern in (*_HIGH_IMPACT, *_MORE_COMPROMISE)
        for match in pattern.finditer(text)
        if not _negated(text, match.start(), match.end()) and not _hedged(text, match.start())
    ]


def confirmation_claims(text: str) -> list[str]:
    """Statements presenting a root cause or vulnerability as confirmed (negated or hedged wording is allowed)."""
    older = [
        match.group(0)
        for pattern in _CONFIRMATION
        for match in pattern.finditer(text)
        if not _negated(text, match.start())
    ]
    newer = [
        match.group(0)
        for pattern in _MORE_CONFIRMATION
        for match in pattern.finditer(text)
        if not _negated(text, match.start()) and not _hedged(text, match.start())
    ]
    return older + newer


def success_claims(text: str) -> list[str]:
    """
    Claims that a fix works, is complete or guarantees safety.

    Only explicit negation excuses a match ("this does not guarantee that ..."); hedging words do not,
    because a hedged overclaim ("could prevent all attacks") is still an overclaim.
    """
    return [
        match.group(0)
        for pattern in (*_SUCCESS_PATTERNS, *_MORE_SUCCESS)
        for match in pattern.finditer(text)
        if not _negated(text, match.start(), match.end())
    ]


def literals(text: str) -> list[str]:
    """Back-ticked spans and path-like tokens: the things a model could invent and present as real."""
    found = [match.group(1).strip() for match in _BACKTICK_RE.finditer(text)]
    for raw in _PATH_RE.findall(text):
        path = raw.rstrip(".,;:)")
        if path not in {"", "/", "//"}:
            found.append(path)
    return [item for item in dict.fromkeys(found) if item]


def ungrounded_literals(text: str, corpus: str) -> list[str]:
    """Literals of `text` that occur nowhere in `corpus` (the material the model was shown)."""
    return [item for item in literals(text) if item not in corpus]
