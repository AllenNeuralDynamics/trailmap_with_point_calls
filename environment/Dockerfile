# hash:sha256:8465457141b0c3d3214cdcebf037d4b41f68728e10cd58d0318cd298be85e5da
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/tensorflow:2.11-python3.10-cuda11.7.0-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libcublas-12-1=12.1.3.1-1 \
        libgl1=1.4.0-1 \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
