#!/bin/bash
set -e

# Setup PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/share/solana/install/active_release/bin:$PATH"

PROJECT_DIR="devalshield-vault"
cd "$PROJECT_DIR"

echo "[1/5] Generating Deployment Keypair..."
if [ ! -f mainnet-deployer.json ]; then
    solana-keygen new -o mainnet-deployer.json --no-bip39-passphrase --force
    echo "Keypair generated: mainnet-deployer.json"
else
    echo "Using existing keypair: mainnet-deployer.json"
fi

echo "[2/5] Configuring Solana CLI..."
solana config set --keypair ./mainnet-deployer.json
solana config set --url mainnet-beta

echo "[3/5] Initial Build to Generate Program Keypair..."
# This might fail on ID mismatch, but it creates the keypair in target/deploy
anchor build || true 

KEYPAIR_PATH="target/deploy/devalshield_vault-keypair.json"
if [ ! -f "$KEYPAIR_PATH" ]; then
    echo "Error: Keypair was not generated at $KEYPAIR_PATH"
    # Fallback: Generate it manually if build failed too early
    mkdir -p target/deploy
    solana-keygen new -o "$KEYPAIR_PATH" --no-bip39-passphrase --force
fi

PROGRAM_ID=$(solana address -k "$KEYPAIR_PATH")
echo ">> Generated Program ID: $PROGRAM_ID"

echo "[4/5] Updating Code with Program ID..."
# Update Anchor.toml
# Sed is tricky with mac/linux differences. Using python for reliability.
python3 -c "
import toml
data = toml.load('Anchor.toml')
data['programs']['localnet']['devalshield_vault'] = '$PROGRAM_ID'
with open('Anchor.toml', 'w') as f:
    toml.dump(data, f)
"
# Update lib.rs
# Simple string replace
python3 -c "
import os
path = 'programs/devalshield-vault/src/lib.rs'
with open(path, 'r') as f:
    content = f.read()
import re
new_content = re.sub(r'declare_id!\(\"[^\"]+\"\);', 'declare_id!(\"$PROGRAM_ID\");', content)
with open(path, 'w') as f:
    f.write(new_content)
"

echo "Code updated."

echo "[5/5] Final Release Build..."
anchor build

echo "--------------------------------------------------------"
echo "READY FOR DEPLOYMENT!"
echo "1. Get funds: solana balance"
echo "2. Address: $(solana address)"
echo "3. To Deploy: anchor deploy --provider.cluster mainnet"
echo "--------------------------------------------------------"
