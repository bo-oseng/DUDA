FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel@sha256:a55ff10111eb11f998884327d37361592e632899edd24fce99886b69289e33e6

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt requirements-iqa-runtime.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip install --no-cache-dir --no-deps pyiqa==0.1.14.1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
CMD ["python", "restoration/infer.py", "--help"]
