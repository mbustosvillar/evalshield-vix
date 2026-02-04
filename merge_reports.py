import json
import sys

def merge(signals_path, macro_path, output_path):
    with open(signals_path, 'r') as f:
        signals = json.load(f)
    
    with open(macro_path, 'r') as f:
        macro = json.load(f)
    
    # Inject macro data into signals
    signals['macro_report'] = macro
    
    with open(output_path, 'w') as f:
        json.dump(signals, f, indent=2)

if __name__ == "__main__":
    merge(sys.argv[1], sys.argv[2], sys.argv[3])
