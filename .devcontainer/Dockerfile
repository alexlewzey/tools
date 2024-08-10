# hadolint global ignore=DL4001,SC1091,DL3008,DL3016,DL3047
FROM mcr.microsoft.com/vscode/devcontainers/python:3.10-bullseye

WORKDIR /workspaces/setup

# Install python dependencies
RUN pip install poetry==1.8.3


# Install hadolint (docker linter)
RUN wget -O /tmp/hadolint https://github.com/hadolint/hadolint/releases/download/v2.12.0/hadolint-Linux-x86_64 && \
    mv /tmp/hadolint /usr/local/bin/hadolint && \
    chmod +x /usr/local/bin/hadolint
