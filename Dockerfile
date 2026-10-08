FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRID_CONFIG=configs/final.yaml

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY configs /app/configs
COPY artifacts /app/artifacts
COPY artifacts_merged_demo /app/artifacts_merged_demo
COPY results /app/results
COPY docs /app/docs
COPY app /app/app

RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Render injects PORT (default 10000); the 7860 default keeps HF compatibility.
ENV GRID_CONFIG=configs/merged_demo.yaml
EXPOSE 7860
CMD ["sh", "-c", "python -m uvicorn app.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
