# Releasing StrataRAG

## One-time setup (in this order, same day, so names stay consistent)

1. **GitHub**: create the organization (e.g. `stratarag-ai` or `stratarag`)
   and the `stratarag` repository. Push this codebase. CI runs on push.
2. **PyPI** (claims the Python name):
   - Create an account at pypi.org, enable 2FA.
   - Go to *Your account -> Publishing -> Add a new pending publisher*:
     project `stratarag`, owner `<your-org>`, repository `stratarag`,
     workflow `release.yml`, environment `pypi`.
     A pending publisher reserves the name for your first trusted release —
     no API token ever exists.
   - (Optional) Repeat on test.pypi.org with environment `testpypi`.
3. **GitHub environments**: repo *Settings -> Environments* -> create `pypi`
   (add required reviewers if you want a human gate on releases) and
   `testpypi`.
4. **npm** (reserves the name for a future TS SDK): from `npm-placeholder/`
   run `npm login && npm publish --access public`. The placeholder clearly
   states it reserves the name and points to the Python project.

## Every release

```bash
# 1. bump version in pyproject.toml and stratarag/__init__.py (keep equal)
# 2. move CHANGELOG "Unreleased" notes under the new version
# 3. dry-run against TestPyPI (optional): Actions -> Release -> Run workflow
git commit -am "release: v0.5.0"
git tag v0.5.0
git push && git push --tags        # tag push triggers build -> publish-pypi
```

The workflow refuses to publish if the tag and `pyproject.toml` version
disagree, twine-checks the artifacts, smoke-tests the wheel in a clean venv
(via CI), and publishes with Sigstore attestations.

## Verify after the first release

```bash
pip install stratarag
python -c "import stratarag; print(stratarag.__version__)"
```
