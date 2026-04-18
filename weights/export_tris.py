"""
Esporta i pesi del modello tris in formato binario raw (float32 LE).
Output: weights/tris_weights.bin  +  weights/tris_hash.txt

Lancia con:
    & "C:\AS\autoresearch-my\.venv\Scripts\python.exe" weights\export_tris.py
"""
import hashlib
import os
import struct
import sys

import torch

MODEL_PATH = r"C:\AS\autoresearch-my\tris\model.pt"

# Architettura: Linear(9,256)->ReLU->Linear(256,256)->ReLU->Linear(256,256)->ReLU->Linear(256,9)
EXPECTED = [
    ("w0", (256, 9)),
    ("b0", (256,)),
    ("w2", (256, 256)),
    ("b2", (256,)),
    ("w4", (256, 256)),
    ("b4", (256,)),
    ("w6", (9, 256)),
    ("b6", (9,)),
]

HERE = os.path.dirname(os.path.abspath(__file__))
BIN_PATH  = os.path.join(HERE, "tris_weights.bin")
HASH_PATH = os.path.join(HERE, "tris_hash.txt")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERRORE: modello non trovato: {MODEL_PATH}")
        sys.exit(1)

    # Calcola hash del file .pt
    sha1 = hashlib.sha1(open(MODEL_PATH, "rb").read()).hexdigest()[:7]
    print(f"Model hash: {sha1}")

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
        f.write(sha1)

    print(f"\nScritto: {BIN_PATH} ({len(raw)} bytes, {len(all_floats)} float32)")
    print(f"Scritto: {HASH_PATH} ({sha1})")


if __name__ == "__main__":
    main()
