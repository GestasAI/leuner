"""F2 — Guardar la primera proteína de Launer como artefacto real.

Extrae el vector-tarea de traducción EN→ES con los parámetros verificados en F1 y lo
serializa en un archivo .protein (numpy + metadatos JSON). El artefacto resultante es
la primera proteína empaquetada de Launer: un saber-hacer de 3 KB listo para el runtime.

Formato del artefacto:
  {nombre}.protein  →  npz con:
    - vector    : float32, shape (hidden_size,)
    - metadata  : JSON con modelo, capa, escala, tarea, n_extract, k_shot, seed, fecha,
                  acc_base, acc_protein, hash_sha256

Uso:
  python f2_save_protein.py --model google/gemma-4-E2B-it
  python f2_save_protein.py --model google/gemma-4-E2B-it --out proteins/translate_gemma4.protein
"""
import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import protein_core as pc
from f1_protein_probes import DATA, TPL

TASK = "translate"
DEFAULT_LAYER = 20
DEFAULT_SCALE = 4.0


def acc(model, layers, L, task, use_chat, tok, test_pairs, vec=None, scale=0.0):
    p, s = TPL[task]["prefix"], TPL[task]["sep"]
    hits = 0
    for x, y in test_pairs:
        ids = pc.encode(tok, [], x, p, s, use_chat)
        if pc.predict_first(model, layers, L, ids, vec, scale) == pc.first_token(tok, y, use_chat):
            hits += 1
    return hits / float(len(test_pairs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--n-extract", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tok, model, use_chat = pc.load_model(a.model)
    layers = pc.get_decoder_layers(model)
    L = a.layer

    print(f"\nModelo: {a.model} | capa {L} | scale {a.scale} | chat={'sí' if use_chat else 'no'}")
    print(f"Tarea: {TASK} | n_extract={a.n_extract} k_shot={a.k_shot} n_test={a.n_test} seed={a.seed}\n")

    rng = random.Random(a.seed)
    all_pairs = list(DATA[TASK])
    rng.shuffle(all_pairs)
    demos, test_pairs = all_pairs[a.n_test:], all_pairs[:a.n_test]

    p, s = TPL[TASK]["prefix"], TPL[TASK]["sep"]
    print("Extrayendo vector… (puede tardar)", flush=True)
    vec = pc.extract_vector(model, layers, L, demos, a.n_extract, a.k_shot, p, s, use_chat, tok, random.Random(a.seed))

    print("Midiendo exactitud…", flush=True)
    base = acc(model, layers, L, TASK, use_chat, tok, test_pairs)
    with_vec = acc(model, layers, L, TASK, use_chat, tok, test_pairs, vec, a.scale)
    print(f"  base: {base:.2f} → +proteína: {with_vec:.2f} (ganancia {with_vec-base:+.2f})")

    vec_f32 = vec.float().numpy()
    vec_bytes = vec_f32.tobytes()
    sha256 = hashlib.sha256(vec_bytes).hexdigest()
    size_kb = len(vec_bytes) / 1024

    metadata = {
        "model": a.model,
        "task": TASK,
        "layer": L,
        "scale": a.scale,
        "n_layers": len(layers),
        "hidden_size": int(vec_f32.shape[0]),
        "n_extract": a.n_extract,
        "k_shot": a.k_shot,
        "n_test": a.n_test,
        "seed": a.seed,
        "acc_base": round(base, 4),
        "acc_protein": round(with_vec, 4),
        "gain": round(with_vec - base, 4),
        "size_kb": round(size_kb, 2),
        "sha256": sha256,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    tag = a.model.split("/")[-1].replace("-", "_").lower()
    out_path = Path(a.out) if a.out else Path("proteins") / f"{TASK}_{tag}.protein"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), vector=vec_f32, metadata=np.array(json.dumps(metadata)))

    print(f"\nProteína guardada: {out_path}")
    print(f"  Peso: {size_kb:.1f} KB | hidden={vec_f32.shape[0]} | sha256={sha256[:16]}…")
    print(f"  Metadatos: modelo={a.model} | capa={L} | scale={a.scale} | tarea={TASK}")
    print(f"  Exactitud: base={base:.0%} → +proteína={with_vec:.0%} (+{with_vec-base:.0%})")
    print("\nPrimera proteína de Launer empaquetada.")


if __name__ == "__main__":
    main()
