ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base-python:3.11-bookworm
FROM ${BUILD_FROM}

ENV LANG=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py run.sh ./
COPY templates ./templates

RUN chmod +x run.sh

CMD [ "/app/run.sh" ]
