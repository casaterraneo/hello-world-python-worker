"""
Genera i file .bin (float32 raw little-endian) dai moduli Python con i pesi hardcoded.
Usato per caricare i pesi su Cloudflare KV tramite requests.http o wrangler.

Uso:
    python export_weights_bin.py

Output:
    tris_weights.bin     — pesi tris  (float32 LE raw)
    briscola_weights.bin — pesi briscola (float32 LE raw)

Poi usa le voci in requests.http per fare upload su KV locale o prod.
"""
import base64
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tris_weights import WEIGHTS_B64 as TRIS_B64
from briscola_weights import WEIGHTS_B64 as BRISCOLA_B64

def export(b64_data: str, output_path: str) -> int:
    raw = base64.b64decode(b64_data)
    with open(output_path, "wb") as f:
        f.write(raw)
    n_floats = len(raw) // 4
    print(f"  {output_path}: {len(raw)} bytes ({n_floats} float32)")
    return len(raw)

if __name__ == "__main__":
    print("Esportazione pesi in formato binario raw...")
    export(TRIS_B64,     "tris_weights.bin")
    export(BRISCOLA_B64, "briscola_weights.bin")
    print("Done. Aggiorna X-Model-Hash in requests.http prima di fare upload.")
