# Contributing to StrataRAG

Thanks for considering a contribution! This document is short on purpose —
the codebase is designed to be easy to extend.

## Ground rules

1. **The core stays dependency-free.** Nothing under `stratarag/` may import
   a third-party package at module level. External services live behind lazy
   imports that raise `MissingDependencyError` with the exact
   `pip install stratarag[extra]` fix. CI enforces this by running the whole
   suite with nothing installed.
2. **Everything offline-testable gets a behavioral test.** Use the built-in
   `EchoProvider` (scripted or synthetic), the hashing embedder, and the
   local stores; tests must pass with no network and no API keys.
3. **Honest failures.** If something can't work, fail loudly with the fix in
   the error message. Never silently degrade.

## Development setup

```bash
git clone https://github.com/<org>/stratarag && cd stratarag
pip install -e .                       # zero deps — this is instant
python -m unittest discover -s tests   # full suite, < 1s, no network
```

There is no lint/format gate beyond byte-compilation in CI; match the
surrounding style (PEP 8-ish, ~90 col, docstrings that explain *why*).

## Adding things

**A vector store adapter** — implement the five methods of
`stratarag.stores.base.VectorStore` (`add`, `query`, `get`, `delete`,
`count`), lazy-import the client inside `__init__`, raise
`MissingDependencyError(package, extra, feature)` on ImportError, register a
spec string in `stratarag/stores/__init__.py::resolve_store`, add the extra
to `pyproject.toml`, and add a missing-dependency-hint test plus (if the
service has an embedded/local mode) a live contract test guarded by
`unittest.skipUnless`.

**An embedding provider** — subclass `Embedder`, L2-normalize output, batch
requests, register the spec in `resolve_embedder`, same error/test rules.

**A pipeline stage** — subclass `Stage`, implement `run(ctx) -> ctx`, call
`self._trace(ctx, t0, **detail)` so runs stay observable, and add behavioral
tests (assert what the stage *does*, not just that it runs).

**An LLM provider** — subclass `LLMProvider`; only `complete()` is required
(async and streaming have defaults).

## Pull requests

- One logical change per PR; include tests; update `CHANGELOG.md` under an
  `Unreleased` heading.
- All matrix jobs in CI must pass (Linux/macOS/Windows, Python 3.9-3.13).
- By contributing you agree your work is licensed under Apache-2.0.

## Reporting security issues

Please do not open public issues for vulnerabilities — see `SECURITY.md`.
