# syntax=docker/dockerfile:1
# Hafizify API server -- runs api/main.py via uvicorn only (no Gradio UI).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# --- System deps ---
# libsndfile1: required by the `soundfile` package (audio I/O) at runtime.
# curl: used by the HEALTHCHECK below.
# Everything in requirements-server.txt ships prebuilt manylinux wheels for
# cp310 -- no compiler needed. build-essential/cmake/python3-dev/
# libeigen3-dev were only ever here to compile kenlm from source, and kenlm
# itself never actually loads in this image (see the comment on it in
# requirements-server.txt, 2026-08-02) -- removed along with it rather than
# leaving unused build tooling bloating every image layer and every build.
# libglib2.0-0: provides libgthread-2.0.so.0, a runtime dependency of
# pygame's bundled SDL2_mixer library. Without it, `import pygame` succeeds
# but pygame.mixer.init() (called in correction_engine.py) raises
# NotImplementedError at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libsndfile1 \
    libglib2.0-0 \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Docker Desktop/WSL2's container networking hands out IPv6 addresses for
# hosts like download.pytorch.org's CDN, but doesn't actually route IPv6
# traffic -- so pip tries the (broken) IPv6 address first and stalls/fails
# instead of falling back to the working IPv4 one. This tells glibc's
# getaddrinfo() (what pip/curl use for DNS) to prefer IPv4 when both exist.
RUN echo "precedence ::ffff:0:0/96 100" >> /etc/gai.conf

# This container has no real sound card, so pygame.mixer.init() would still
# fail trying to open one. SDL_AUDIODRIVER=dummy tells SDL (which pygame
# wraps) to use a no-op virtual audio device instead -- mixer.init()
# succeeds and playback calls silently no-op rather than raising.
ENV SDL_AUDIODRIVER=dummy

WORKDIR /app

# --- Python deps first, for layer caching ---
COPY requirements-server.txt .
# --mount=type=cache persists pip's download cache in a BuildKit cache
# volume OUTSIDE the image layers (so it doesn't bloat the final image),
# separately from Docker's normal layer cache. This is the fix that
# actually matters: layer caching alone only helps when NOTHING earlier in
# this Dockerfile changes -- one edit to the apt-get line above (which
# happened three times in one afternoon while debugging kenlm) invalidates
# this whole layer and forces torch+torchvision+torchaudio (multi-GB, CUDA
# 12.4 build) to re-download from scratch every time. The cache mount
# survives that: even when this RUN has to re-execute, pip pulls from the
# persistent cache instead of the network. Dropped --no-cache-dir
# everywhere below -- it was telling pip to never use a download cache at
# all, which defeats the mount entirely; with the mount instead, cached
# wheels still never end up IN the final image (the cache dir isn't part
# of any image layer), so there's no image-size tradeoff for removing it.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
    pip install -r requirements-server.txt

# --- App code -- only what api/main.py's import chain actually needs ---
COPY api/ ./api/
COPY realtime_streamer.py session_manager.py hybrid_diacritic_pipeline.py \
     harakaat_error_detector.py surah_detector.py correction_engine.py \
     groq_transcriber.py quran_audio_provider.py quran_trie.py \
     quran_trie_cache.pkl quran_lm.txt quran_5gram.arpa ./
COPY fyp_model/ ./fyp_model/

# --- Model weights ---
# Left OUT of the image deliberately -- docker-compose.yml mounts this dir
# as a read-only volume instead, so rebuilding the image during development
# doesn't mean re-copying 1.5GB+ every time. Switch this to a COPY for a
# final, single-artifact deploy image once you're done iterating.
# COPY whisper-l-v3-turbo-quran-lora-dataset-mix/ ./whisper-l-v3-turbo-quran-lora-dataset-mix/

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
