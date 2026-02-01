FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Install core build tools and dependencies
RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    python3-tk \
    curl \
    unzip \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone acados repository
ARG ACADOS_TAG=v0.5.3
RUN git clone -b ${ACADOS_TAG} --recursive https://github.com/acados/acados.git

# Download t_renderer v0.2.0 for Linux AMD64
RUN wget -O /workspace/acados/bin/t_renderer https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-amd64 \
    && chmod +x /workspace/acados/bin/t_renderer

# Build acados
RUN mkdir acados/build && cd acados/build && \
    cmake .. \
      -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_BUILD_TYPE=Release && \
    make install -j2

# Create Python virtual environment
RUN python3 -m venv env
ENV PATH="/workspace/env/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install Python requirements
COPY requirements.txt .
RUN pip install -r ./requirements.txt

# Install editable acados_template Python package
RUN pip install -e acados/interfaces/acados_template

# Export environment variables for acados libs
ENV ACADOS_SOURCE_DIR=/workspace/acados
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/acados/lib
