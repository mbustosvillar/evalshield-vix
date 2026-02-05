FROM python:3.12-slim

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r devalshield && useradd -r -g devalshield devalshield

# MANUAL TORCH CPU INSTALL (Priority 1)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY --chown=devalshield:devalshield requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=devalshield:devalshield . .

USER devalshield

CMD ["python", "collective_bot.py"]
