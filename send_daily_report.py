import asyncio
import os
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from telegram import Bot
from integrated_orchestrator import run_orchestrator

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

async def send_daily_report():
    """Script designed for Railway Cron execution."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_GROUP_ID")

    if not token or not chat_id:
        logger.error("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_GROUP_ID environment variables.")
        sys.exit(1)

    logger.info("Starting daily report orchestration...")
    
    try:
        # Run the full pipeline (Live data)
        payload = run_orchestrator(mock=False)
        
        # The orchestrator now generates the full narrative list
        narrative_str = "\n".join(payload.get('narrative', []))
        
        text = (
            f"🛡️ *DEVALSHIELD – Citadel View • {datetime.now().strftime('%d %b %Y')} • 09:00 AR*\n\n"
            f"{narrative_str}\n\n"
            f"🚀 *Modo operativo*\n"
            f"Simulation ON • Kill-Switch OFF • Vault Ready (Devnet)"
        )

        bot = Bot(token=token)
        await bot.send_message(chat_id=chat_id, text=text, parse_mode='Markdown')
        logger.info("Daily report sent successfully.")

    except Exception as e:
        logger.error(f"Failed to generate or send daily report: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(send_daily_report())
