ARG PYTHON_VERSION=3.12
FROM mcr.microsoft.com/devcontainers/python:1-${PYTHON_VERSION}-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /uvx /usr/local/bin/

# PROJECT: tacl (TSD upload client)
RUN UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin \
    uv tool install --python /usr/local/bin/python tsd-api-client==3.9.6

# Mount targets owned by vscode, so named volumes inherit the ownership
RUN install -d -o vscode -g vscode -m 700 /home/vscode/.ssh \
    && install -d -o vscode -g vscode /home/vscode/.cache /home/vscode/.config/tacl \
        /home/vscode/.local/share/tacl

# No sudo for the container user
RUN rm -f /etc/sudoers.d/*

USER vscode
