#!/bin/bash
set -e

# Export paths locally for this script execution
export PATH="$HOME/.cargo/bin:$HOME/.local/share/solana/install/active_release/bin:$PATH"

echo "[1/4] Installing Rust (No Path Mod)..."
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
else
    echo "Rust already installed."
fi

echo "[2/4] Installing Solana CLI (Stable)..."
if ! command -v solana &> /dev/null; then
    sh -c "$(curl -sSfL https://release.solana.com/v1.18.4/install)"
else
    echo "Solana CLI already installed."
fi

# Need to source specifically to make sure we catch the new binaries
export PATH="$HOME/.cargo/bin:$HOME/.local/share/solana/install/active_release/bin:$PATH"

echo "[3/4] Installing Anchor (AVM)..."
if ! command -v avm &> /dev/null; then
    # OpenSSL might be needed on Mac, sometimes it's flaky without it. 
    # Attempting standard cargo install.
    cargo install --git https://github.com/coral-xyz/anchor avm --locked --force
    avm install latest
    avm use latest
else
    echo "Anchor (AVM) already installed."
fi

echo "[4/4] Verifying Versions..."
rustc --version
solana --version
anchor --version

echo "---------------------------------------------------"
echo "INSTALLATION COMPLETE"
echo "Please run this command to update your current shell:"
echo 'export PATH="$HOME/.cargo/bin:$HOME/.local/share/solana/install/active_release/bin:$PATH"'
