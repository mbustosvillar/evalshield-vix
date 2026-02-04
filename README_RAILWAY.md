# 🛡️ DevalShield Railway Deployment Guide

Sigue estos pasos para poner tu bot online en 2 minutos:

### 1. Preparar el Repositorio
Asegúrate de que estos archivos estén en la misma carpeta:
- `collective_bot.py`
- `integrated_orchestrator.py`
- `signal_engine.py` (y demás dependencias de lógica)
- `requirements.txt`
- `Dockerfile`

### 2. Despliegue en Railway
1. Ve a [Railway.app](https://railway.app/) y crea un nuevo proyecto.
2. Selecciona **"Deploy from GitHub repo"** (o usa el Railway CLI si lo tienes instalado).
3. En la configuración del servicio, ve a la pestaña **Variables** y agrega:
   - `TELEGRAM_BOT_TOKEN`: `8121906722:AAHLk4YaEUShOAcy_Eb86GXasPLeo-UZha8` (vix10bot)
   - `TELEGRAM_GROUP_ID`: (El ID de tu grupo de Telegram*)

### 3. Cómo obtener el Group ID
1. Agrega a tu bot al grupo de Telegram y nómbralo **Administrador**.
2. Envía un mensaje cualquiera al grupo.
3. Abre esta URL en tu navegador (reemplazando el token):
   `https://api.telegram.org/bot8121906722:AAHLk4YaEUShOAcy_Eb86GXasPLeo-UZha8/getUpdates`
4. Busca el campo `"chat":{"id": -100XXXXXXXXXX}`. Ese número (incluyendo el signo menos) es tu `TELEGRAM_GROUP_ID`.

### 4. Producción
Una vez configuradas las variables, Railway detectará el `Dockerfile` y lanzará el bot automáticamente. Podrás ver los logs en vivo desde el dashboard de Railway.

---
**Nota sobre modelos de ML**: Los archivos `.pth` (Transformer y PPO) deben estar en el repositorio para que el orquestador funcione correctamente en modo producción.
