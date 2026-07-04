## What & why

<!-- One paragraph: the problem, and how this PR solves it.
     Link the issue: Fixes #123 -->

## Checklist

- [ ] Tests added/updated — **behavioral** (assert what the change *does*), runnable offline with `python -m unittest discover -s tests`
- [ ] Core stays dependency-free: no module-level third-party imports under `stratarag/` (CI's hygiene test enforces this)
- [ ] New optional backends: lazy import + `MissingDependencyError` with the `pip install stratarag[extra]` hint, extra added to `pyproject.toml`, hint-path test added
- [ ] New stages call `self._trace(...)` so runs stay observable
- [ ] `CHANGELOG.md` updated under `## Unreleased`
- [ ] Docs/README updated if the public API changed

## How I tested it

<!-- Paste the test names or a snippet. If this touches a live backend
     (Pinecone, Redis, ...), say which service+version you verified against. -->

## Breaking changes

<!-- None expected? say "None". Otherwise describe the migration. -->
