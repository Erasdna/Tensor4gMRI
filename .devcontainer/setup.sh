#!/usr/bin/env bash
# postCreateCommand: install the project and run the local CI checks. Failures only warn.
set -uo pipefail

warn() { echo "WARNING: $*" >&2; }

uv sync --frozen --all-extras || warn "uv sync failed"
uv run pre-commit run --all-files || warn "pre-commit reported problems"
SANDBOX_TESTS_REQUIRED=1 uv run pytest -q -p no:cacheprovider .devcontainer/tests \
    || warn "devcontainer isolation/workflow tests failed"

# PROJECT: tacl
if [ ! -s "$HOME/.config/tacl/config" ]; then
    echo "tacl is not registered in this container yet: run 'tacl p2386 --register'"
fi
if [ ! -x .git/hooks/pre-commit ]; then
    echo "Install the git hook on the host (.git/hooks is read-only here): uv run pre-commit install"
fi
echo "Container setup complete."
