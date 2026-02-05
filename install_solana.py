import os
import io
import tarfile
import requests
import shutil
import sys

# CONFIG
SOLANA_VERSION = "v1.18.4"
ARCH = "x86_64-apple-darwin" # Will update if uname -m says arm64
INSTALL_DIR = os.path.expanduser("~/.local/share/solana/install")
ACTIVE_RELEASE_DIR = os.path.join(INSTALL_DIR, "active_release")

def install_solana():
    # Detect Arch
    is_arm = os.uname().machine == 'arm64'
    target = "aarch64-apple-darwin" if is_arm else "x86_64-apple-darwin"
    
    url = f"https://github.com/solana-labs/solana/releases/download/{SOLANA_VERSION}/solana-release-{target}.tar.bz2"
    print(f"[INFO] Downloading Solana {SOLANA_VERSION} for {target} from {url}...")
    
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        
        print("[INFO] Extracting...")
        with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:bz2") as tar:
            tar.extractall(path=INSTALL_DIR)
            
        # The tar contains a folder named 'solana-release'. Rename/Move it.
        extracted_folder = os.path.join(INSTALL_DIR, "solana-release")
        target_folder = os.path.join(INSTALL_DIR, f"solana-release-{SOLANA_VERSION}")
        
        if os.path.exists(target_folder):
            shutil.rmtree(target_folder)
        os.rename(extracted_folder, target_folder)
        
        # Symlink active_release
        if os.path.exists(ACTIVE_RELEASE_DIR):
            os.unlink(ACTIVE_RELEASE_DIR)
        os.symlink(target_folder, ACTIVE_RELEASE_DIR)
        
        print(f"[SUCCESS] Solana installed to {ACTIVE_RELEASE_DIR}")
        print("Add this to your PATH: export PATH=\"$HOME/.local/share/solana/install/active_release/bin:$PATH\"")
        
    except Exception as e:
        print(f"[ERROR] Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if not os.path.exists(INSTALL_DIR):
        os.makedirs(INSTALL_DIR)
    install_solana()
