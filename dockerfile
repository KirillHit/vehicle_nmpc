FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

# Install core build tools and dependencies
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    wget \
    python3.12 \
    python3.12-dev \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

ENV UV_PROJECT_ENVIRONMENT=/workspace/env
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_LINK_MODE=copy
ENV MPLBACKEND=Agg
ENV PATH="/workspace/env/bin:$PATH"

# Clone acados repository
ARG ACADOS_TAG=v0.5.4
RUN git clone -b ${ACADOS_TAG} --recursive https://github.com/acados/acados.git

# Download t_renderer v0.2.0 for Linux AMD64
RUN mkdir -p /workspace/acados/bin \
    && wget -O /workspace/acados/bin/t_renderer https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-amd64 \
    && chmod +x /workspace/acados/bin/t_renderer

# Build acados
RUN mkdir acados/build && cd acados/build && \
    cmake .. \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    make install -j"$(nproc)"

# Install Python dependencies from the locked uv environment
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.python-version,target=.python-version \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev --no-install-project

# Install editable acados_template Python package
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /workspace/env/bin/python -e acados/interfaces/acados_template

# Export environment variables for acados libs
ENV ACADOS_SOURCE_DIR=/workspace/acados
ENV LD_LIBRARY_PATH=/workspace/acados/lib

CMD ["/bin/bash"]
