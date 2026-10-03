FROM python:3.11-slim-bookworm

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

# Install system runtime & build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py run.sh ./
COPY templates ./templates

RUN chmod +x run.sh

CMD [ "/app/run.sh" ]
