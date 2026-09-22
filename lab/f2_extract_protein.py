"""F2 — Extrae y GUARDA una proteína como artefacto ("oro").

Extrae el function vector de una capacidad, mide su eficacia, y lo guarda:
  proteins/<nombre>.f32   → el vector crudo (float32 little-endian) — cargable en Python/Rust/Dart
  proteins/<nombre>.json  → metadatos + sha256 (procedencia e integridad)

Uso:
  python f2_extract_protein.py                               # traducción EN→ES, Gemma 4 E2B
  python f2_extract_protein.py --task capital --layer 20 --scale 4.0
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

import protein_core as pc
from f1_protein_probes import DATA, TPL


def split(task, n_test, seed):
    d = list(DATA[task])
    random.Random(seed).shuffle(d)
    return d[n_test:], d[:n_test]


def accuracy(model, layers, L, task, use_chat, tok, n_test, seed, vec=None, scale=0.0):
    _, test = split(task, n_test, seed)
    p, s = TPL[task]["prefix"], TPL[task]["sep"]
    hits = 0
    for x, y in test:
        ids = pc.encode(tok, [], x, p, s, use_chat)
        if pc.predict_first(model, layers, L, ids, vec, scale) == pc.first_token(tok, y, use_chat):
            hits += 1
    return hits / float(len(test))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--task", default="translate", choices=list(DATA))
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--n-extract", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="proteins")
    a = ap.parse_args()

    tok, model, use_chat = pc.load_model(a.model)
    layers = pc.get_decoder_layers(model)
    L = a.layer if a.layer >= 0 else pc.default_layer(len(layers))
    demos, _ = split(a.task, a.n_test, a.seed)
    p, s = TPL[a.task]["prefix"], TPL[a.task]["sep"]

    print(f"Extrayendo proteína '{a.task}' de {a.model} (capa {L}, escala {a.scale}) …", flush=True)
    vec = pc.extract_vector(model, layers, L, demos, a.n_extract, a.k_shot, p, s, use_chat, tok,
                            random.Random(a.seed))

    base = accuracy(model, layers, L, a.task, use_chat, tok, a.n_test, a.seed)
    prot = accuracy(model, layers, L, a.task, use_chat, tok, a.n_test, a.seed, vec, a.scale)
    print(f"  eficacia: base {base:.2f} → +proteína {prot:.2f}  (ganancia {prot-base:+.2f})")

    # --- guardar como oro: vector crudo f32 LE + metadatos + sha256 ---
    arr = vec.to(torch.float32).cpu().numpy().astype("<f4")  # little-endian float32
    raw = arr.tobytes()
    sha = hashlib.sha256(raw).hexdigest()

    short = a.model.split("/")[-1]
    name = f"{a.task}.{short}.L{L}.s{a.scale}"
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    (out / f"{name}.f32").write_bytes(raw)
    meta = {
        "name": name,
        "capability": a.task,
        "model": a.model,
        "layer": L,
        "scale": a.scale,
        "dims": int(arr.shape[0]),
        "dtype": "float32",
        "byte_order": "little-endian",
        "format": "raw .f32 (dims * 4 bytes); cargar como float32[dims]",
        "extraction": {"method": "mean-activation v0 (Hendel 2023)",
                       "n_extract": a.n_extract, "k_shot": a.k_shot,
                       "use_chat": use_chat, "seed": a.seed},
        "efficacy": {"accuracy_base": round(base, 4), "accuracy_protein": round(prot, 4),
                     "metric": "first-token zero-shot", "n_test": a.n_test},
        "sha256": sha,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "use": "Inyectar: hidden[capa L, último token] += scale * vector  (ver protein_core.inject).",
    }
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    size = len(raw)
    print(f"\n  GUARDADO ✓")
    print(f"   {out / (name + '.f32')}   ({size} bytes ≈ {size/1024:.1f} KB)")
    print(f"   {out / (name + '.json')}  (metadatos + sha256)")
    print(f"   sha256: {sha[:16]}…")
    print("\n  Es la primera pieza de la biblioteca de proteínas de Launer. Guárdala como oro.")


if __name__ == "__main__":
    main()
