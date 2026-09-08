FROM python:3.11-slim-bookworm

ARG LLVM_MAJOR=20

ENV LLVM_ROOT=/usr/lib/llvm-$LLVM_MAJOR
ENV PATH=$LLVM_ROOT/bin:/opt/uv:$PATH
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        libc6-dev \
        ninja-build \
        gnupg \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /etc/apt/trusted.gpg.d/apt.llvm.org.gpg \
    && echo "deb [signed-by=/etc/apt/trusted.gpg.d/apt.llvm.org.gpg] https://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-$LLVM_MAJOR main" \
        > /etc/apt/sources.list.d/llvm.list \
    && apt-get update \
    && apt-get install --no-install-recommends -y \
        clang-$LLVM_MAJOR \
        llvm-$LLVM_MAJOR \
        llvm-$LLVM_MAJOR-tools \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

WORKDIR /workspace

COPY pyproject.toml .python-version README.md ./
RUN uv lock

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN uv sync --group dev

ENTRYPOINT ["uv", "run", "pipedream-smoke"]
