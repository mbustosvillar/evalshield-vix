import os
import sys
import logging
from solana_bridge import SafeVaultBridge
from alert_manager import DevalAlertManager

# Ensure PYTHONPATH
sys.path.append(os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ForceRisk")

def force_risk_event():
    """
    FORCES a critical risk condition for MVP testing.
    This bypasses the market data scanning and directly triggers 
    the Solana Bridge pending request and the Alert Manager.
    """
    print("--- 🛡️ FORCING CRITICAL RISK EVENT (MVP TEST) ---")
    
    # 1. Initialize Components
    bridge = SafeVaultBridge()
    alerts = DevalAlertManager()
    
    dvi_simulated = 85.5
    strategy = "FORCED_PROTECTION_MVP_TEST"

    print(f"[STEP 1] Creating pending request in DB for DVI={dvi_simulated}...")
    res = bridge.create_pending_request(dvi_simulated, strategy)
    print(f"  Result: {res}")

    print(f"\n[STEP 2] Triggering critical alerts to Telegram...")
    try:
        alerts.trigger_critical_alert(
            dvi_simulated, 
            f"🔥 EVENTO DE RIESGO FORZADO (TEST): Inestabilidad detectada. GATILLO: {strategy}"
        )
        print("  Alert sent successfully (Check Telegram!)")
    except Exception as e:
        print(f"  Error sending alert: {e}")

    print("\n--- ✅ FORCE EVENT COMPLETED ---")
    print("Próximos pasos para el usuario:")
    print("1. El bot 'collective_bot.py' debe estar corriendo.")
    print("2. En Telegram, usa /hedge_status para ver la solicitud pendiente.")
    print("3. Usa /approve para ejecutar la protección simula.")

if __name__ == "__main__":
    force_risk_event()
