# DeepStream 7.1 on Jetson (L4T R36.x / JetPack 6.x)
# Build:  docker build -t person-tracker .
# Run:    docker run --runtime=nvidia -v /path/to/video:/data person-tracker --input /data/input.mp4
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

WORKDIR /app

# ── Fix 1: Jetson CUDA library path ─────────────────────────────────────
ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:${LD_LIBRARY_PATH}

# ── Fix 2: Install real NVIDIA DeepStream Python bindings (pyds 1.1.11) ─
# The pip package named "pyds" is a different unrelated project.
RUN wget -q -O /tmp/pyds.whl \
    "https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.11/pyds-1.1.11-py3-none-linux_aarch64.whl" \
    && pip3 install /tmp/pyds.whl \
    && rm /tmp/pyds.whl

# Copy source and assets
# .dockerignore excludes *.engine (device-specific, auto-rebuilt on first run)
COPY main.py .
COPY pipeline/ ./pipeline/
COPY config/ ./config/
COPY models/ ./models/

RUN mkdir -p /app/output

# First run auto-builds TRT engines (~9 min total), cached in models/:
#   - PeopleNet:  resnet34_peoplenet_int8.onnx_b1_gpu0_int8.engine  (~5 min)
#   - OSNet ReID: osnet_x0_25_msmt17.onnx_b32_gpu0_fp16.engine     (~4 min)
# Mount models/ to persist engines across container restarts:
#   docker run ... -v $(pwd)/models:/app/models ...

ENTRYPOINT ["python3", "main.py"]
CMD ["--input", "/data/input.mp4", "--output", "/app/output/tracked.mp4"]
