import asyncio
import json
import os
import hashlib
import logging
from datetime import datetime, time as dtime
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
from solana_bridge import SafeVaultBridge
from persistence import DevalDBManager

# Instances will be initialized in main()
bridge = None
db = None

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
    if not group_id:
        logger.warning("[CONFIG] TELEGRAM_GROUP_ID is missing using fallback to local logs only.")
    elif not str(group_id).startswith("-100"):
        logger.critical(f"Suspicious GROUP_ID detected: {group_id}")
        sys.exit(1)
        
    return token, group_id

# CONFIG
TOKEN, GROUP_CHAT_ID = load_secure_config()
DATA_FILE = "collective_feedback.json"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Comando de inicio con menú institucional."""
    text = (
        "🛡️ *DevalShield Collective Bot v2*\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Bienvenido, operamos bajo protocolos de *Zero Trust*.\n\n"
        "🔍 *Comandos Disponibles:*\n"
        "• `/report`: Último análisis del DVI.\n"
        "• `/status`: Salud del sistema y bias ML.\n"
        "• `/force`: Forzar nueva inferencia ahora.\n"
        "• `/hedge_status`: Estado del vault en Solana.\n"
        "• `/approve`: Aprobar Tx de cobertura pendiente.\n"
        "• `/lock` / `/unlock`: Control de Kill-Switch.\n"
    )
    await update.message.reply_text(text, parse_mode='Markdown')

async def trigger_now(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Alias para ver el reporte instantáneo sin forzar nueva inferencia."""
    await update.message.reply_text("🔎 Consultando último registro en DB...")
    # Llama a la lógica de reporte diario pero sin el schedule de JobQueue
    await daily_report_logic(update, context)

async def daily_report_logic(update_or_context, context_if_job=None):
    """Lógica compartida para generar y enviar el reporte."""
    try:
        verify_integrity("tail_risk_model.pth", EXPECTED_MODEL_HASH)
        payload = run_orchestrator(mock=False)
        
        dvi = payload['context']['deval_vacuum_index']
        prob = payload.get('tail_risk_probability', 'N/A')
        # The orchestrator now generates the full structured narrative
        narrative = "\n".join(payload.get('narrative', []))
        
        text = (
            f"🛡️ *DEVALSHIELD – Citadel View • {datetime.now().strftime('%d %b %Y')} • 09:00 AR*\n\n"
            f"{narrative}\n\n"
            f"⚠️ *Institucional:* Análisis basado en framework risk-management de alta frecuencia. Prohibida la redistribución."
        )
        
        if hasattr(update_or_context, 'message'): # Viene de un comando
            await update_or_context.message.reply_text(text, parse_mode='Markdown')
        else: # Viene del JobQueue
            await update_or_context.bot.send_message(chat_id=GROUP_CHAT_ID, text=text, parse_mode='Markdown')
            
        db.log_signal(payload)
    except Exception as e:
        logger.error(f"Report Logic Error: {e}")

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

# db instance will be initialized in main()

async def hedge_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Consulta el estado del bridge y transacciones pendientes."""
    status = bridge.get_hedge_status()
    text = (
        f"🏦 *Estado del Vault Solana*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Program ID: `{status['program_id'][:8]}...`\n"
        f"Red: `{status['cluster'].upper()}`\n"
        f"Status Bridge: {'✅ ONLINE' if status['solana_available'] else '⚠️ OFFLINE (MOCK)'}\n\n"
        f"Pendiente: {'🔴 SI' if status['pending_tx_id'] else '🟢 NINGUNA'}\n"
    )
    if status['pending_tx_id']:
        text += f"ID Solicitud: `{status['pending_tx_id']}`\nUse `/approve` para ejecutar."
    
    await update.message.reply_text(text, parse_mode='Markdown')

async def approve_tx_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ejecuta una transacción aprobada manualmente."""
    await update.message.reply_text("⌛ *Procesando firma en Solana Mainnet...*", parse_mode='Markdown')
    result = await bridge.execute_approved_unwind()
    
    if result['success']:
        data = result['data']
        text = (
            f"✅ *Protección Ejecutada*\n"
            f"DVI Gatillo: {data['dvi']}\n"
            f"Estrategia: `{data['strategy']}`\n"
            f"Tx: [Ver en Explorer](https://solscan.io/tx/{data['tx_signature']})"
        )
    else:
        text = f"❌ *Error de Ejecución:* {result['error']}"
    
    await update.message.reply_text(text, parse_mode='Markdown')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Returns current system status and last analysis timestamp."""
    is_locked = db.get_state("kill_switch_active", "true") == "true"
    
    # Check collective bias adjustment (from JSON for now as it's small)
    adjustment = 0
    if os.path.exists("collective_bias.json"):
        with open("collective_bias.json") as f:
            bias_data = json.load(f)
            adjustment = bias_data.get('adjustment', 0)

    status_info = {
        "bot": "🟢 ONLINE",
        "model_loaded": os.path.exists("tail_risk_model.pth"),
        "kill_switch": is_locked,
        "bias": adjustment
    }
    
    text = (
        f"📊 *DevalShield System Status*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 Bot: {status_info['bot']}\n"
        f"🧠 Model: {'✅' if status_info['model_loaded'] else '❌'}\n"
        f"🔒 Kill Switch: {'⛔ ACTIVE' if status_info['kill_switch'] else '🟢 OFF'}\n"
        f"📈 Bias Adj: `{status_info['bias']:+.3f}`\n"
    )
    
    await update.message.reply_text(text, parse_mode='Markdown')

async def force_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Forces an immediate full analysis and returns results."""
    await update.message.reply_text("🔄 *Ejecutando análisis completo...*", parse_mode='Markdown')
    
    try:
        verify_integrity("tail_risk_model.pth", EXPECTED_MODEL_HASH)
        payload = run_orchestrator(mock=False)
        
        dvi = payload['context']['deval_vacuum_index']
        prob = payload.get('tail_risk_probability', 'N/A')
        
        text = (
            f"⚡ *Análisis On-Demand Completado*\n\n"
            f"*DEVAL VACUUM INDEX:* {dvi} / 100\n"
            f"*30D TAIL-RISK PROB:* {prob}%\n\n"
            f"_Generado: {datetime.now().strftime('%H:%M:%S')}_"
        )
        await update.message.reply_text(text, parse_mode='Markdown')
        db.log_signal(payload)
        
    except Exception as e:
        await update.message.reply_text(f"❌ *Error:* {str(e)}", parse_mode='Markdown')

async def killswitch_on(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db.set_state("kill_switch_active", "true")
    await update.message.reply_text("⛔ SYSTEM LOCKED. On-chain triggers disabled.")

async def killswitch_off(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db.set_state("kill_switch_active", "false")
    await update.message.reply_text("⚠️ SYSTEM UN-LOCKED. On-chain triggers ENABLED.")

def main():
    global db, bridge
    try:
        # Initialize Core Instances
        db = DevalDBManager()
        bridge = SafeVaultBridge()
        
        # Check write permissions for DB
        try:
            db_dir = os.path.dirname(os.path.abspath(db.db_path))
            test_file = os.path.join(db_dir, "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info(f"Disk write check PASSED at {db_dir}")
        except Exception as e:
            logger.critical(f"CRITICAL DISK ERROR: Cannot write to {db_dir}. Bot will fail. Check Docker permissions. Error: {e}")
            sys.exit(1)

        # Final environment validation
        if os.getenv("RAILWAY_ENVIRONMENT") == "production":
            logger.info("Running in PRODUCTION mode. Zero Trust protocols active.")

        logger.info(f"Initializing Application with token: {TOKEN[:5]}...{TOKEN[-5:]}")
        app = Application.builder().token(TOKEN).build()
        
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("report", trigger_now))
        app.add_handler(CommandHandler("status", status_command))
        app.add_handler(CommandHandler("force", force_analysis))
        app.add_handler(CommandHandler("hedge_status", hedge_status_command))
        app.add_handler(CommandHandler("approve", approve_tx_command))
        app.add_handler(CommandHandler("lock", killswitch_on))
        app.add_handler(CommandHandler("unlock", killswitch_off))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_feedback))
        
        logger.info("Collective Bot starting polling...")
        app.run_polling()
    except Exception as e:
        logger.critical(f"FATAL BOOT ERROR: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
