"""
Phase 13 live-execution support, kept OUTSIDE `validation/`.

The `validation` package is pinned by AST tests to have no network, process or filesystem access. A real
HTTP transport therefore cannot live there without weakening those pins. This sibling package is the only
place a live request may be constructed, and only after every approval, scope and opt-in gate has passed.

Nothing here is imported by `validation/` so a fake-transport unit test cannot accidentally become live
evidence. Reports produced through this package set `execution.transport_kind == "live"`; mocked reports
keep `fake`.
"""
