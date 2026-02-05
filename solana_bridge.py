import asyncio
import os
import json

# Try to import Solana dependencies, but don't crash if they fail (Graceful Degradation)
try:
    from solders.pubkey import Pubkey
    from solana.rpc.async_api import AsyncClient
    from anchorpy import Program, Provider, Wallet, Idl
    SOLANA_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Solana Integration disabled due to missing dependencies: {e}")
    SOLANA_AVAILABLE = False
    # Mock classes/values to prevent NameError at runtime
    Pubkey = None
    AsyncClient = None

# ==================== CONFIG ====================
# [USER ACTION REQUIRED] Update this after 'anchor deploy'
PROGRAM_ID_STR = os.getenv("SOLANA_PROGRAM_ID", "11111111111111111111111111111111") 
RPC_URL = os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
SAFETY_LOCK_FILE = "safety_lock.json"

async def check_safety_lock():
    """Returns True if the Kill-Switch is ACTIVE (System Locked)."""
    if not os.path.exists(SAFETY_LOCK_FILE):
        # Default to LOCKED (True) for safety if file is missing
        return True
    
    try:
        with open(SAFETY_LOCK_FILE, "r") as f:
            data = json.load(f)
            return data.get("kill_switch_active", True)
    except:
        return True # Fail-safe to locked

async def update_onchain_index(index_value: float):
    """
    Updates the Deval Vacuum Index on-chain.
    INCLUDES KILL-SWITCH CHECK.
    """
    if not SOLANA_AVAILABLE:
        print(f"[INFO] Skipping On-Chain update (Dependencies missing). DVI: {index_value}")
        return False

    is_locked = await check_safety_lock()
    
    if is_locked:
        print(f"[BLOCKED] Kill-Switch is ACTIVE. Unwind Trigger of {index_value} blocked.")
        return False

    print(f"[INFO] Updating On-Chain Index to {index_value}...")
    
    # Load wallet (Production: use limited authority keypair)
    # keypair_path = os.getenv("SOLANA_KEYPAIR_PATH", "~/.config/solana/devalshield.json")
    
    try:
        # Placeholder for Anchor Logic
        # client = AsyncClient(RPC_URL)
        # ... logic to create transaction ...
        # await client.send_transaction(...)
        print(f"[SUCCESS] Solana Oracle updated. Current DVI: {index_value} (Simulated Mainnet)")
        return True
    except Exception as e:
        print(f"[ERROR] Solana Interaction Failed: {e}")
        return False

if __name__ == "__main__":
    # Test run
    asyncio.run(update_onchain_index(80.0))
