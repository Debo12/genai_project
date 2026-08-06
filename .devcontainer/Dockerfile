FROM mcr.microsoft.com/devcontainers/python:3.12

ENV POETRY_VERSION=2.1.3

RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    build-essential \
    gcc \
    g++ && \
    pip install --upgrade pip && \
    pip install poetry==$POETRY_VERSION

ENV POETRY_VIRTUALENVS_IN_PROJECT=true

WORKDIR /workspace

CMD ["sleep", "infinity"]