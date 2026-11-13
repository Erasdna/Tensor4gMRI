#!/bin/bash
# Moves the gmritensor pin in uv.lock to the head of the branch named in pyproject.toml, syncs
# the local environment, runs the tests and pre-commit, and rebuilds the container. Run from the
# repo root on the host:
#   bash code/scripts/update_gmritensor.sh
# Revert with: git checkout uv.lock && uv sync

set -euo pipefail

uv lock --upgrade-package gmritensor
uv sync --extra dev

# Check that nothing broke
uv run pytest
uv run --with pre-commit pre-commit run --all-files

apptainer build --force apptainer/app.sif apptainer/app.def
