
import sys
import os
from pathlib import Path
import json

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("⚠️ Pandas not found. Falling back to standard `json` library.")

def validate_jsonl(file_path_str):
    print(f"🔍 Validating JSONL: {file_path_str}")
    
    if not os.path.exists(file_path_str):
        # Try relative to script location if absolute path fails
        p = Path(file_path_str.strip(os.sep)) 
        if not p.exists():
             print(f"❌ File not found: {file_path_str}")
             return
        file_path_str = str(p)

    if HAS_PANDAS:
        try:
            df = pd.read_json(file_path_str, lines=True)
            print(f"✅ Successfully loaded into DataFrame!")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {df.columns.tolist()}")
            
            print("\n--- First 2 Rows ---")
            print(df.head(2))
            
            if 'query_embedding' in df.columns:
                first_emb = df.iloc[0]['query_embedding']
                print(f"\n--- Embedding Check (Row 0) ---")
                print(f"   Type: {type(first_emb)}")
                if isinstance(first_emb, list):
                     print(f"   Length: {len(first_emb)}")
                     print(f"   First 5 values: {first_emb[:5]}")
        except Exception as e:
            print(f"❌ Error reading JSONL with pandas: {e}")
            return
    else:
        print("ℹ️ verifying using standard json library...")
        try:
            with open(file_path_str, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"✅ Successfully read {len(lines)} lines.")
            if len(lines) > 0:
                first_obj = json.loads(lines[0])
                print(f"   First record keys: {list(first_obj.keys())}")
                if 'query_embedding' in first_obj:
                    emb = first_obj['query_embedding']
                    print(f"   Embedding length: {len(emb)}")
                    print(f"   First 5 values: {emb[:5]}")
        except Exception as e:
            print(f"❌ Error reading JSONL with json: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Default fallback for testing if no arg provided, matches the user's example context
        path = r"runs\run_20260215_102924_woM_INS_S_HEAD50_nCH2000\outputs\queries_embeddings_20260215_102924.jsonl"
    
    validate_jsonl(path)
