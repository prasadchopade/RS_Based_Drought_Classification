FROM python:3.12-slim

# The rasterio wheel needs this and the slim image leaves it out
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Runs the tests, then the demo. Outputs land in /app/out, mount a volume there to keep them.
CMD ["sh", "-c", "python -m pytest -q && python -m drought demo --outdir out"]
