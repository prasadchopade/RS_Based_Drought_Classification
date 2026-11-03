FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Runs the tests, then the demo. Outputs land in /app/out, mount a volume there to keep them.
CMD ["sh", "-c", "python -m pytest -q && python -m drought demo --outdir out"]
