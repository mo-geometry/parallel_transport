# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Versioning Scheme

- **Major (X)**: Breaking API changes, architectural rewrites
- **Minor (Y)**: New features, backwards-compatible additions
- **Patch (Z)**: Bug fixes, minor improvements, dependency updates

## [Unreleased]

## [0.1.0] - 2026-05-24

### Added
- Project scaffolding: src layout, pyproject.toml with hatchling build backend
- Quaternion algebra module: multiplication, conjugate, axis-angle conversion, SO(3) mapping
- Sphere geometry module: Delaunay triangulation, stereographic projection, spherical triangles
- GitHub Actions CI/CD: lint (ruff), typecheck (mypy strict), test matrix (3.11/3.12/3.13)
- Pre-commit hooks: ruff lint + format, mypy, trailing whitespace, YAML/TOML checks
- Makefile with dev, lint, format, typecheck, test, clean targets
- Test suite covering quaternion operations and sphere geometry
- CLAUDE.md with project context and roadmap

[Unreleased]: https://github.com/mo-geometry/parallel_transport/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mo-geometry/parallel_transport/releases/tag/v0.1.0
