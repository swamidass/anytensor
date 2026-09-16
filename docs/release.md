# Release

AnyTensor follows [**Semantic Versioning**](https://semver.org/)
(`MAJOR.MINOR.PATCH`):

| Bump | When |
|---|---|
| **MAJOR** | Breaking changes to the public API or documented semantics (signatures, promote defaults, empty-segment identities, …) |
| **MINOR** | Backward-compatible features |
| **PATCH** | Bug fixes, docs, tests |

We do **not** ship breaking changes in minor or patch releases. Prefer a
deprecation + changelog note when behavior must evolve; remove or flip it on
the next major.

Package version comes from git tags
([hatch-vcs](https://github.com/ofek/hatch-vcs)), not a static `version` in
`pyproject.toml`. Do not bump a version field. Do not run `towncrier build`
locally for a real release (hatch-vcs would write a `.devN` version into
`CHANGELOG.md`).

## During development

User-facing changes need a towncrier fragment in `changelog.d/` in the same
unit of work:

```bash
uv run towncrier create --no-edit -c "Short description." added.md
```

Types: `added` | `changed` | `fixed` | `removed` | `deprecated` | `security`.
Skip fragments for internal tests, refactors, and tooling.

## Cut a release

On `main`, with fragments committed:

```bash
git tag -a v0.1.0 -m "0.1.0"
git push origin v0.1.0
```

Pushing `v*` runs `.github/workflows/release.yml`:

1. Test matrix (3.11–3.13) with the **coverage gate** (`fail_under=100`,
   non-fuzz) and bounded fuzz
2. `uv build` (sdist + wheel)
3. Towncrier compiles `CHANGELOG.md` onto the default branch
4. GitHub Release with the dist artifacts
5. **PyPI** via Trusted Publishing (OIDC, environment `pypi`)
