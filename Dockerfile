FROM python:3.11.9-slim

ENV PATH="/root/.local/bin:$PATH"

RUN apt-get update && apt-get install -y \
    liblzma-dev pipx git git-lfs \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system
RUN pipx install poetry

WORKDIR /app

COPY pyproject.toml .
COPY poetry.lock .