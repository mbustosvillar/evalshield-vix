#!/bin/bash
set -e

# Setup PATH for this session
export PATH="$HOME/.cargo/bin:$HOME/.local/share/solana/install/active_release/bin:$PATH"

echo "Using Rust: $(command -v rustc)"
echo "Using Solana: $(command -v solana)"

echo "[3/4] Installing Anchor (AVM)..."
# Force install avm
cargo install --git https://github.com/coral-xyz/anchor avm --locked --force

# Install latest anchor version
avm install latest
avm use latest

echo "---------------------------------------------------"
echo "Verifying Anchor:"
anchor --version
