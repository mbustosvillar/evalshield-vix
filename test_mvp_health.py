import os
import sys
import logging

# Ensure absolute paths for imports if necessary, though PYTHONPATH=. is set
sys.path.append(os.getcwd())

from data_provider import get_provider
from integrated_orchestrator import run_orchestrator
from solana_bridge import SafeVaultBridge

logging.basicConfig(level=logging.INFO)

def test_health():
    print("--- 🛡️ DevalShield MVP Health Check ---")
    
    # 1. Test Data Provider Fallback
    print("\n[1/3] Testing Data Provider & Fallbacks...")
    provider = get_provider(mode="live")
    context = provider.get_latest_context()
    print(f"Data source used: {context.get('provider', 'mock')}")
    print(f"Context Sample (SPY): {context.get('spy_price')}")

    # 2. Test Orchestrator (Dry Run)
    print("\n[2/3] Testing Orchestrator Engine...")
    try:
        # We run with mock=False to test the live-to-mock fallback logic
        payload = run_orchestrator(mock=False)
        print(f"Orchestration Success. DVI: {payload['context']['deval_vacuum_index']}")
    except Exception as e:
        print(f"Orchestrator Error: {e}")

    # 3. Test Solana Bridge Simulation
    print("\n[3/3] Testing Solana Bridge Simulation Mode...")
    bridge = SafeVaultBridge()
    # Mock a pending request manually in DB if empty to test /approve
    bridge.create_pending_request(85.0, "TEST_MVP_SIMULATION")
    status = bridge.get_hedge_status()
    print(f"Bridge Cluster: {status['cluster']}")
    print(f"Pending Tx ID: {status['pending_tx_id']}")

    print("\n--- ✅ Health Check Finished ---")

if __name__ == "__main__":
    test_health()
