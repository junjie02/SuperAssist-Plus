# Source Tree Technical Documentation

IMPORTANT: Any change to the `src` tree that affects package structure, import
boundaries, public APIs, or runtime behavior must update this document.

## Purpose

The `src` directory contains the installable Python package for
SuperAssist-Plus. Packaging is configured in `pyproject.toml` with setuptools
package discovery rooted at this directory.

## Structure

- `superassist_plus/`: the application package.

No executable logic should live directly in `src`; implementation belongs under
`superassist_plus`.

## Import Boundary

Code outside the package should import from `superassist_plus`, not from deeper
private implementation paths unless tests need module-level coverage.

## Maintenance Notes

- Keep the `src` layout compatible with editable installs.
- Add new top-level packages only when there is a clear ownership boundary.
- Update this document and the package-level `AGENT.md` when adding or moving
  packages.

