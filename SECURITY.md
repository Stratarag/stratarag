# Security Policy

## Supported versions

The latest minor release receives security fixes.

## Reporting a vulnerability

Email the maintainers (see `pyproject.toml` authors) or use GitHub's
private vulnerability reporting on the repository. Please include a
reproduction. You will receive an acknowledgement within 72 hours.

Please do not open public issues for security reports.

## Design notes relevant to security

- The core makes **no network calls**; only explicitly constructed
  components (API providers, store adapters, `CloudStore`) touch the
  network, and none are constructed by default.
- The playground dashboard binds to `127.0.0.1` by default and has no
  authentication — do not expose it publicly.
- SQL identifiers in the SQLite/pgvector backends are validated; values are
  always parameterized.
