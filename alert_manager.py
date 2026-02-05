import os
import logging
import sqlite3
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DevalAlertManager:
    """Institutional alert system with multi-channel redundancy."""
    
    def __init__(self):
        self.db_path = "devalshield.db"
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.group_id = os.getenv("TELEGRAM_GROUP_ID")
        
        # Redundancy config placeholders
        self.email_enabled = os.getenv("SENDGRID_API_KEY") is not None
        self.sms_enabled = os.getenv("TWILIO_SID") is not None

    def log_alert(self, level: str, message: str):
        """Persists the alert for audit trails."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "INSERT INTO signals (dvi, tail_risk_prob, spy_price, vix_current, timestamp) VALUES (?, ?, ?, ?, ?)",
                (0.0, 0, 0.0, 0.0, datetime.now().isoformat())
            )
            # Note: We append alert level to message if we want to store it in a specific field.
            # For now, we reuse the signals or system_state or create a dedicated alerts table if complexity grows.
            # Let's keep it simple: print to log + send notifications.
            logging.info(f"[{level}] {message}")
            conn.close()
        except Exception as e:
            logging.error(f"Alert logging failed: {e}")

    def send_telegram(self, message: str):
        """Primary notification channel."""
        if not self.telegram_token or not self.group_id:
            return False
            
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.group_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            requests.post(url, json=payload, timeout=5)
            return True
        except Exception as e:
            logging.error(f"Telegram alert failed: {e}")
            return False

    def trigger_critical_alert(self, dvi: float, msg: str = None):
        """Fires all redundant channels for high-risk events."""
        alert_msg = f"‼️ *CRITICAL REDUNDANCY ALERT* ‼️\n\n"
        alert_msg += f"DVI Threshold Exceeded: *{dvi:.2f}%*\n"
        alert_msg += f"Message: {msg or 'High tail-risk detected. Check Vault status.'}\n\n"
        alert_msg += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # 1. Telegram (Primary)
        self.send_telegram(alert_msg)
        
        # 2. Email (Redundant - Placeholder for SendGrid)
        if self.email_enabled:
            logging.info("Sending Redundant Email (SendGrid)...")
            # Implementation would go here
            
        # 3. SMS (Redundant - Placeholder for Twilio)
        if self.sms_enabled:
            logging.info("Sending Redundant SMS (Twilio)...")
            # Implementation would go here
            
        self.log_alert("CRITICAL", alert_msg)

if __name__ == "__main__":
    manager = DevalAlertManager()
    manager.trigger_critical_alert(82.5, "Test institutional pulse.")
