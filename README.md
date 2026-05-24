# Parallel Transport

Interactive visualisation of parallel transport, quaternion geometry, and fibre
bundles on the 2-sphere.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
make test
```

## Project Structure

```
src/parallel_transport/
  core/       — Quaternion algebra, transforms
  geometry/   — Sphere mesh, stereographic projection, spherical triangles
  physics/    — Parallel transport, Hamiltonian, S1 bundle phases
  ui/         — PyQt5 application, matplotlib subplots (future)
```
