# Isolated runtime for executing LLM-generated pipeline code.
# Library versions are pinned to match the host so scores are reproducible.
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Run as an unprivileged user, never root
RUN useradd --create-home --uid 1000 sandbox
USER sandbox
WORKDIR /work

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg

CMD ["python"]
