import sqlite3
import json
from datetime import datetime
import os

class DevalDBManager:
    """Institutional-grade persistence manager using SQLite."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # En Railway usamos /app, localmente el directorio actual
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(base_dir, "devalshield.db")
        else:
            self.db_path = db_path
            
        print(f"[DB] Initializing database at: {self.db_path}")
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Signals Table (Market metrics & DVI)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dvi REAL,
                    tail_risk_prob REAL,
                    spy_price REAL,
                    vix_current REAL,
                    vix_term_structure REAL,
                    raw_context TEXT
                )
            ''')
            
            # 2. Transactions Table (Vault Operations)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    strategy TEXT,
                    dvi REAL,
                    status TEXT, -- PENDING, EXECUTED, FAILED
                    tx_signature TEXT,
                    executed_at TEXT
                )
            ''')
            
            # 3. System State (Persistence for configs/kill-switch)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Default state
            cursor.execute('''
                INSERT OR IGNORE INTO system_state (key, value, updated_at)
                VALUES ('kill_switch_active', 'true', ?)
            ''', (datetime.now().isoformat(),))
            
            conn.commit()

    # --- System State Logic ---
    def set_state(self, key: str, value: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO system_state (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, str(value).lower(), datetime.now().isoformat()))
            conn.commit()

    def get_state(self, key: str, default: str = None):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT value FROM system_state WHERE key = ?', (key,))
            row = cursor.fetchone()
            return row[0] if row else default

    # --- Signals Logic ---
    def log_signal(self, payload: dict):
        with sqlite3.connect(self.db_path) as conn:
            ctx = payload.get('context', {})
            conn.execute('''
                INSERT INTO signals (
                    timestamp, dvi, tail_risk_prob, spy_price, 
                    vix_current, vix_term_structure, raw_context
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                ctx.get('deval_vacuum_index'),
                payload.get('tail_risk_probability'),
                ctx.get('spy_price'),
                ctx.get('vix_current'),
                ctx.get('vix_term_structure'),
                json.dumps(payload)
            ))
            conn.commit()

    # --- Transactions Logic ---
    def create_transaction_request(self, dvi: float, strategy: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO transactions (timestamp, strategy, dvi, status)
                VALUES (?, ?, ?, 'PENDING')
            ''', (datetime.now().isoformat(), strategy, dvi))
            conn.commit()
            return cursor.lastrowid

    def update_transaction(self, tx_id: int, status: str, signature: str = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE transactions 
                SET status = ?, tx_signature = ?, executed_at = ?
                WHERE id = ?
            ''', (status, signature, datetime.now().isoformat(), tx_id))
            conn.commit()

    def get_pending_transaction(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT id, dvi, strategy FROM transactions WHERE status = "PENDING" ORDER BY id DESC LIMIT 1')
            return cursor.fetchone()

if __name__ == "__main__":
    db = DevalDBManager()
    db.set_state("test_key", "active")
    print(f"Test State: {db.get_state('test_key')}")
    print("Database Initialized Successfully.")
