# n4m dev shell — pinned to the tools n4m's CI targets.
#
# Goals (per docs/REFACTOR_PLAN.md §2.12):
#   * C++17 toolchain + CMake + Ninja + OpenBLAS
#   * Python 3.12 + uv (the package manager n4m's Python pipeline uses)
#   * R 4.4 + system deps for the in-tree R binding
#   * Octave 9 + statistics package (CI exercises Octave, not MATLAB)
#   * Node 22 + Emscripten 3.x (for the JS / WASM binding)
#   * Docker CLI (host docker socket mounted by docker-compose) — needed
#     because parity-snapshots.yml builds reference env images
#
# Optional toolchains (Rust, Julia, .NET) are installed by the
# `post-create.sh` hook so the base image stays small for the people
# who never need them.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
    PATH=/usr/local/bin:/root/.cargo/bin:/root/.juliaup/bin:/opt/emsdk:/opt/emsdk/upstream/emscripten:$PATH \
    EMSDK=/opt/emsdk \
    EMSDK_VERSION=3.1.74

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget gnupg lsb-release locales sudo \
        git jq tree less ripgrep fd-find \
        build-essential gcc-12 g++-12 clang-16 lld-16 \
        cmake ninja-build make pkg-config \
        libopenblas-dev liblapacke-dev \
        gfortran \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        r-base r-base-dev libxml2-dev libssl-dev libcurl4-openssl-dev \
        libfontconfig1-dev libharfbuzz-dev libfribidi-dev \
        libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev \
        octave octave-statistics liboctave-dev \
        nodejs npm \
        docker.io \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-12 100 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-16 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-16 100 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 100 \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# uv — fast Python package manager (n4m's Python pipeline standardises on it).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && mv /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv --version

# Emscripten — pinned to 3.1.74 (matches the JS binding's Dockerfile).
RUN git clone --depth=1 https://github.com/emscripten-core/emsdk.git ${EMSDK} \
    && cd ${EMSDK} \
    && ./emsdk install ${EMSDK_VERSION} \
    && ./emsdk activate ${EMSDK_VERSION} \
    && rm -rf ${EMSDK}/.git

# Node global tools used by the binding test pipelines.
RUN npm install -g pnpm@10

# Make Ubuntu's "fdfind" available as "fd" (matches docs).
RUN ln -s /usr/bin/fdfind /usr/local/bin/fd

# Convenience: a real shell prompt for `make shell` users.
RUN echo 'PS1="[n4m-dev \W]\\$ "' >> /root/.bashrc \
    && echo 'source /opt/emsdk/emsdk_env.sh > /dev/null 2>&1 || true' >> /root/.bashrc

WORKDIR /workspaces/nirs4all-methods
