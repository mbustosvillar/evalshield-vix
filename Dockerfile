FROM python:3.10-slim

# Nivel Palantir: Hardening de Imagen
WORKDIR /app

# 1. Instalar dependencias mínimas y limpiar
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Gestión de usuarios (No-Root Protocol)
RUN groupadd -r devalshield && useradd -r -g devalshield devalshield
RUN chown -R devalshield:devalshield /app

# 3. Instalación de dependencias
COPY --chown=devalshield:devalshield requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copia de código y modelos (Integrity Protected)
COPY --chown=devalshield:devalshield . .

# 5. Switch to non-root user
USER devalshield

# Setting environment variables (To be injected via Railway Secrets)
ENV TELEGRAM_BOT_TOKEN=""
ENV TELEGRAM_GROUP_ID=""
ENV RAILWAY_ENVIRONMENT="production"

# Zero Trust Boot
CMD ["python", "collective_bot.py"]
