"""F1b — Barrido de escala + composición de DOS proteínas reales.

Aísla la escala en UN modelo y mide la composición de dos tareas que AMBAS funcionan
(por defecto capital + traducción; antónimo no tiene proteína en Gemma 4 chat).
Para cada escala, en el mismo modelo y capa:
  - A sola, B sola        ← ¿cada proteína funciona a esa escala?
  - comp_A, comp_B        ← A+B sumadas, ¿sobreviven las dos?  (= composición real)
  - fuga A→B              ← la proteína de A sobre la tarea B (especificidad)

Uso:
  python f1b_scale_sweep.py --model google/gemma-4-E2B-it
  python f1b_scale_sweep.py --model google/gemma-4-E2B-it --task-a capital --task-b translate
"""
import argparse
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import protein_core as pc
from f1_protein_probes import DATA, TPL


def split(task, n_test, seed):
    d = list(DATA[task])
    random.Random(seed).shuffle(d)
    return d[n_test:], d[:n_test]


def vec_for(model, layers, L, task, k, n_ext, use_chat, tok, n_test, seed):
    demos, _ = split(task, n_test, seed)
    p, s = TPL[task]["prefix"], TPL[task]["sep"]
    return pc.extract_vector(model, layers, L, demos, n_ext, k, p, s, use_chat, tok, random.Random(seed))


def acc(model, layers, L, task, use_chat, tok, n_test, seed, vec=None, scale=0.0):
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
    ap.add_argument("--task-a", default="capital", choices=list(DATA))
    ap.add_argument("--task-b", default="translate", choices=list(DATA))
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--scales", default="0.5,1,2,3,4,6")
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--n-extract", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    nt, sd, A, B = a.n_test, a.seed, a.task_a, a.task_b
    scales = [float(x) for x in a.scales.split(",")]

    tok, model, use_chat = pc.load_model(a.model)
    layers = pc.get_decoder_layers(model)
    L = a.layer if a.layer >= 0 else pc.default_layer(len(layers))
    print(f"\nModelo: {a.model} | capa L={L} | chat={'sí' if use_chat else 'no'} | A={A} B={B} | seed={sd}\n")

    v_a = vec_for(model, layers, L, A, a.k_shot, a.n_extract, use_chat, tok, nt, sd)
    v_b = vec_for(model, layers, L, B, a.k_shot, a.n_extract, use_chat, tok, nt, sd)

    hdr = f"  {'escala':>6} | {'A sola':>6} | {'B sola':>6} | {'comp_A':>6} | {'comp_B':>6} | {'fuga A→B':>8}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    best = None
    for s in scales:
        comb = v_a + v_b
        a_s = acc(model, layers, L, A, use_chat, tok, nt, sd, v_a, s)
        b_s = acc(model, layers, L, B, use_chat, tok, nt, sd, v_b, s)
        c_a = acc(model, layers, L, A, use_chat, tok, nt, sd, comb, s)
        c_b = acc(model, layers, L, B, use_chat, tok, nt, sd, comb, s)
        lk = acc(model, layers, L, B, use_chat, tok, nt, sd, v_a, s)
        print(f"  {s:>6.1f} | {a_s:>6.2f} | {b_s:>6.2f} | {c_a:>6.2f} | {c_b:>6.2f} | {lk:>8.2f}")
        if a_s >= 0.5 and b_s >= 0.5 and c_a >= 0.7 * a_s and c_b >= 0.7 * b_s and best is None:
            best = s

    print("\n  Lectura: 'A/B sola' deben funcionar; 'comp_*' altas = COMPONEN; ≈0 = interfieren.")
    if best is not None:
        print(f"  → Componen las dos manteniéndose a escala ~{best}. La composición v0 ES posible en Gemma 4.")
    else:
        print("  → Ninguna escala mantuvo ambas. En Gemma 4: una proteína a la vez (composición → ortogonalizar).")


if __name__ == "__main__":
    main()
