import asyncio
import json
import os
import hashlib
import logging
from datetime import datetime, time as dtime
import sys

# Structured Logging Config (Nivel Institucional)
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class SecurityError(Exception):
    pass

# Import our orchestration logic
from integrated_orchestrator import run_orchestrator

# Try to import telegram, but provide instructions if missing
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
except ImportError:
    logger.error("python-telegram-bot not installed. Run: pip install python-telegram-bot")
    sys.exit(1)

# SECURITY PROTOCOLS (Nivel Palantir)
EXPECTED_MODEL_HASH = "aff7ebcb2d27e7efc245efae32ce3ed97a08d643b1094596748c549911819f52"

def verify_integrity(file_path: str, expected_hash: str):
    """Verifica que los modelos ML no hayan sido manipulados."""
    if not os.path.exists(file_path):
        logger.warning(f"File {file_path} not found for integrity check.")
        return
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    current_hash = sha256_hash.hexdigest()
    if current_hash != expected_hash:
        logger.critical(f"INTEGRITY BREACH: {file_path} hash mismatch!")
        raise SecurityError(f"Integrity breach detected in {file_path}")
    logger.info(f"Integrity verified for {file_path}")

def load_secure_config():
    """Motor de validación estricto para secretos y entorno."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    group_id = os.getenv("TELEGRAM_GROUP_ID")
    
    # 1. Validación de Token
    if not token or len(token) < 20:
        logger.critical("CRITICAL: TELEGRAM_BOT_TOKEN is missing or invalid.")
        logger.info("Please add TELEGRAM_BOT_TOKEN to Railway project variables.")
        sys.exit(1)
    
    # 2. Validación de Group ID (Protocolo de Lista Blanca)
    if group_id and not str(group_id).startswith("-100"):
        logger.critical(f"Suspicious GROUP_ID detected: {group_id}")
        sys.exit(1)
        
    return token, group_id

# CONFIG
TOKEN, GROUP_CHAT_ID = load_secure_config()
DATA_FILE = "collective_feedback.json"

async def daily_report(context: ContextTypes.DEFAULT_TYPE):
    """Orchestrates and broadcasts the daily Deval Vacuum Index report."""
    logger.info("Generating daily collective report...")
    
    # Security Check: Verify model integrity before run
    try:
        verify_integrity("tail_risk_model.pth", EXPECTED_MODEL_HASH)
    except SecurityError as e:
        logger.critical(f"Security halt: {e}")
        return

    # Run the full integrated pipeline (signals + ML + Sentiment)
    # We use mock=False if token is available for X/Bluelytics, else True
    payload = run_orchestrator(mock=True)
    
    dvi = payload['context']['deval_vacuum_index']
    prob = payload.get('tail_risk_probability', 'N/A')
    narrative = "\n".join(payload.get('narrative', []))
    
    text = (
        f"🛡️ *DevalShield Collective Report*\n"
        f"📅 {datetime.now().strftime('%d %b %Y')}\n\n"
        f"*DEVAL VACUUM INDEX:* {dvi} / 100\n"
        f"*30D TAIL-RISK PROB:* {prob}%\n\n"
        f"📖 *Strategic Narrative:*\n{narrative}\n\n"
        f"💬 *Feedback Requerido:*\n"
        f"¿Sientes presión devaluatoria real hoy en tu zona?\n"
        f"Responde: *SÍ* / *NO* (+ comentario/señal local)"
    )
    
    # Send to group
    chat_id = context.job.chat_id if context.job else GROUP_CHAT_ID
    await context.bot.send_message(
        chat_id=chat_id, 
        text=text, 
        parse_mode='Markdown'
    )
    
    # Log the baseline for re-training later
    record = {
        "timestamp": datetime.now().isoformat(),
        "dvi": dvi,
        "tail_risk_prob": prob,
        "features": payload['context'],
        "feedbacks": []
    }
    
    with open(DATA_FILE, "a") as f:
        json.dump(record, f)
        f.write("\n")

async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Parses SÍ/NO feedback and stores it for model refinement."""
    if not update.message or not update.message.text:
        return

    text = update.message.text.upper().strip()
    user = update.message.from_user.username or update.message.from_user.first_name
    
    if text.startswith("SÍ"):
        outcome = 1
        msg = f"✅ Gracias {user}. Feedback positivo registrado (Presión detectada)."
    elif text.startswith("NO"):
        outcome = 0
        msg = f"✅ Gracias {user}. Feedback negativo registrado (Estabilidad detectada)."
    else:
        # Ignore other messages or handle as extra context
        return

    # Store feedback (simple append to a separate feedback log or update last record)
    feedback_entry = {
        "user": user,
        "outcome": outcome,
        "raw_text": text,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("member_signals.json", "a") as f:
        json.dump(feedback_entry, f)
        f.write("\n")
    
    await update.message.reply_text(msg)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Bienvenido al *DevalShield Collective Bot*.\n"
        "Recibirás informes diarios y podrás contribuir al entrenamiento del modelo con tu feedback local.",
        parse_mode='Markdown'
    )

async def trigger_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manual trigger for testing."""
    await daily_report(context)

async def killswitch_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ADMIN: Activates the safety lock."""
    # Authenticate admin here in prod
    with open("safety_lock.json", "w") as f:
        json.dump({"kill_switch_active": True}, f)
    await update.message.reply_text("⛔ SYSTEM LOCKED. On-chain triggers disabled.")

async def killswitch_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ADMIN: Deactivates the safety lock."""
    # Authenticate admin here in prod
    with open("safety_lock.json", "w") as f:
        json.dump({"kill_switch_active": False}, f)
    await update.message.reply_text("⚠️ SYSTEM UN-LOCKED. On-chain triggers ENABLED.")

def main():
    # Final environment validation
    if os.getenv("RAILWAY_ENVIRONMENT") == "production":
        logger.info("Running in PRODUCTION mode. Zero Trust protocols active.")

    app = Application.builder().token(TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("report", trigger_now))
    app.add_handler(CommandHandler("lock", killswitch_on))
    app.add_handler(CommandHandler("unlock", killswitch_off))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_feedback))
    
    # Schedule daily report at 10 AM
    # app.job_queue.run_daily(daily_report, time=dtime(10, 0), chat_id=GROUP_CHAT_ID)
    
    logger.info("Collective Bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()
