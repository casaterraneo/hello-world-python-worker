"""
Esporta i pesi del modello briscola in formato binario raw (float32 LE).
Output: weights/briscola_weights.bin  +  weights/briscola_hash.txt

Lancia con:
    & "C:\AS\autoresearch-my\.venv\Scripts\python.exe" weights\export_briscola.py
"""
import hashlib
import os
import struct
import sys

import torch

MODEL_PATH = r"C:\AS\autoresearch-my\briscola2\best_models\model_0.8640_0f4a93c.pt"

# Architettura: Linear(31,256)->ReLU->Linear(256,256)->ReLU->Linear(256,3)
EXPECTED = [
    ("w0", (256, 31)),
    ("b0", (256,)),
    ("w2", (256, 256)),
    ("b2", (256,)),
    ("w4", (3, 256)),
    ("b4", (3,)),
]

HERE = os.path.dirname(os.path.abspath(__file__))
BIN_PATH  = os.path.join(HERE, "briscola_weights.bin")
HASH_PATH = os.path.join(HERE, "briscola_hash.txt")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERRORE: modello non trovato: {MODEL_PATH}")
        sys.exit(1)

    # Calcola hash dal nome del file (già nel filename: model_0.8640_0f4a93c.pt)
    model_name = os.path.basename(MODEL_PATH)
    parts = model_name.replace(".pt", "").split("_")
    model_hash = parts[-1] if len(parts) >= 3 else hashlib.sha1(open(MODEL_PATH, "rb").read()).hexdigest()[:7]
    print(f"Model hash: {model_hash}")

    # Carica pesi
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif hasattr(ckpt, "state_dict"):
        sd = ckpt.state_dict()
    else:
        sd = ckpt

    tensors = list(sd.values())
    print(f"Tensori trovati: {len(tensors)}")

    # Verifica shapes
    if len(tensors) != len(EXPECTED):
        print(f"ERRORE: attesi {len(EXPECTED)} tensori, trovati {len(tensors)}")
        sys.exit(1)
    for i, (name, shape) in enumerate(EXPECTED):
        actual = tuple(tensors[i].shape)
        status = "OK" if actual == shape else "MISMATCH"
        print(f"  {status} {name}: {actual}")
        if status == "MISMATCH":
            sys.exit(1)

    # Serializza come float32 raw LE
    all_floats = []
    for t in tensors:
        all_floats.extend(t.detach().cpu().float().flatten().tolist())
    raw = struct.pack(f"{len(all_floats)}f", *all_floats)

    with open(BIN_PATH, "wb") as f:
        f.write(raw)
    with open(HASH_PATH, "w") as f:
        f.write(model_hash)

    print(f"\nScritto: {BIN_PATH} ({len(raw)} bytes, {len(all_floats)} float32)")
    print(f"Scritto: {HASH_PATH} ({model_hash})")


if __name__ == "__main__":
    main()
