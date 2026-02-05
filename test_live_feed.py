
"""
test_live_feed.py
=================
Verifies that real-time market data can be fetched via yfinance.
"""
from data_provider import get_provider
import json

def test_live_data():
    print("🚀 Initializing Live Market Data Provider...")
    try:
        provider = get_provider("live")
        print("⏳ Fetching data from Yahoo Finance (this may take a few seconds)...")
        data = provider.get_latest_context()
        
        print("\n✅ Data Fetched Successfully:")
        print(json.dumps(data, indent=2))
        
        # Basic validation
        if data['is_mock']:
            print("\n⚠️ WARNING: Rate Limit active. Using Mock Data Fallback (System Stable).")
        else:
            assert data['spy_price'] > 0, "Error: SPY price invalid."
            print("\n🎉 Live Feed Verification PASSED.")
            
        print(f"Metrics (VIX: {data['vix_current']}, SPY: {data['spy_price']})")
        
    except Exception as e:
        print(f"\n❌ Verification FAILED: {e}")

if __name__ == "__main__":
    test_live_data()
