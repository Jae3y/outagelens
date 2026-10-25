# OutageLens: web control room + API in one container.
#   docker build -t outagelens .
#   docker run -p 8000:8000 outagelens        ->  http://localhost:8000
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# libgomp: OpenMP runtime used by the OpenDSS (DSS C-API) binaries
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY core ./core
COPY server ./server
COPY web ./web
COPY scripts/warm_cache.py ./scripts/warm_cache.py
COPY run.py .

# Train the GNNs, calibrate, and run the default benchmarks at build time, so the container serves within
# seconds of starting even on a 0.1-CPU free tier.
RUN python scripts/warm_cache.py

RUN useradd --create-home outagelens && chown -R outagelens /app
USER outagelens

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/health', timeout=4)"
CMD ["python", "run.py", "--host", "0.0.0.0"]
