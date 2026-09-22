"""F4 — Robustez de la proteina translate.en_es sobre 5 semillas.

Por cada semilla (0..4): extrae la proteina, mide accuracy en el test set,
registra el valor. Calcula media, desviacion estandar e intervalo de confianza
95% por bootstrap.

Gate:
  media >= 0.75 -> RESULTADO ROBUSTO
  media >= 0.50 -> SENAL PROMETEDORA
  media  < 0.50 -> REVISAR

Salida: proteins/robustness_report.json
"""
import json
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import protein_core as pc
from f1_protein_probes import DATA, TPL

MODEL_ID = "google/gemma-4-E2B-it"
TASK = "translate"
LAYER = 20
SCALE = 4.0
N_EXTRACT = 10
N_TEST = 20
K_SHOT = 5
SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP_REPS = 2000


def split(n_test, seed):
    d = list(DATA[TASK])
    random.Random(seed).shuffle(d)
    return d[n_test:], d[:n_test]


def run_seed(model, tok, use_chat, seed):
    """Extrae proteina con `seed` y mide accuracy. Devuelve (base, prot)."""
    layers = pc.get_decoder_layers(model)
    demos, test = split(N_TEST, seed)
    p, s = TPL[TASK]["prefix"], TPL[TASK]["sep"]

    vec = pc.extract_vector(
        model, layers, LAYER, demos, N_EXTRACT, K_SHOT,
        p, s, use_chat, tok, random.Random(seed)
    )

    hits_base = 0
    hits_prot = 0
    for x, y in test:
        ids = pc.encode(tok, [], x, p, s, use_chat)
        expected = pc.first_token(tok, y, use_chat)
        if pc.predict_first(model, layers, LAYER, ids, None, 0.0) == expected:
            hits_base += 1
        if pc.predict_first(model, layers, LAYER, ids, vec, SCALE) == expected:
            hits_prot += 1

    n = float(len(test))
    return hits_base / n, hits_prot / n


def bootstrap_ci(values, reps, alpha=0.05, seed=42):
    """IC percentilico 95% por bootstrap."""
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(reps)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def gate_label(mean):
    if mean >= 0.75:
        return "RESULTADO ROBUSTO"
    if mean >= 0.50:
        return "SENAL PROMETEDORA"
    return "REVISAR"


def main():
    print(f"\n=== F4 — Robustez sobre {len(SEEDS)} semillas ===", flush=True)
    print(f"Modelo: {MODEL_ID} | Tarea: {TASK} | Capa: {LAYER} | Scale: {SCALE}", flush=True)
    t0 = time.time()

    # Cargar modelo una sola vez
    print("\nCargando modelo ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_chat = getattr(tok, "chat_template", None) is not None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    print(f"  Modelo cargado. chat={use_chat}\n", flush=True)

    results = []
    for seed in SEEDS:
        print(f"  Semilla {seed} ...", end=" ", flush=True)
        base, prot = run_seed(model, tok, use_chat, seed)
        results.append({"seed": seed, "accuracy_base": round(base, 4),
                        "accuracy_protein": round(prot, 4),
                        "gain": round(prot - base, 4)})
        print(f"base={base:.2f} | +proteina={prot:.2f} (+{prot - base:.2f})", flush=True)

    prot_vals = [r["accuracy_protein"] for r in results]
    base_vals = [r["accuracy_base"] for r in results]

    mean_p = float(np.mean(prot_vals))
    std_p = float(np.std(prot_vals, ddof=1))
    lo, hi = bootstrap_ci(prot_vals, BOOTSTRAP_REPS)
    mean_b = float(np.mean(base_vals))
    label = gate_label(mean_p)

    print(f"\n{'Semilla':>8} | {'Base':>6} | {'Accuracy':>8}")
    print("-" * 32)
    for r in results:
        print(f"{r['seed']:>8} | {r['accuracy_base']:>6.2f} | {r['accuracy_protein']:>8.2f}")
    print("-" * 32)
    print(f"  Media proteina: {mean_p:.4f}")
    print(f"  Std (n-1):      {std_p:.4f}")
    print(f"  IC 95%:         [{lo:.4f}, {hi:.4f}]")
    print(f"  Media base:     {mean_b:.4f}")
    print(f"\n  -> {label}")

    elapsed = time.time() - t0

    report = {
        "task": TASK,
        "model": MODEL_ID,
        "layer": LAYER,
        "scale": SCALE,
        "n_extract": N_EXTRACT,
        "n_test": N_TEST,
        "k_shot": K_SHOT,
        "seeds": results,
        "summary": {
            "mean_accuracy_protein": round(mean_p, 4),
            "std_accuracy_protein": round(std_p, 4),
            "ci_95_bootstrap": [round(lo, 4), round(hi, 4)],
            "mean_accuracy_base": round(mean_b, 4),
            "bootstrap_reps": BOOTSTRAP_REPS,
            "gate": label,
        },
        "elapsed_s": round(elapsed, 1),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out = Path("proteins")
    out.mkdir(exist_ok=True)
    out_path = out / "robustness_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Guardado: {out_path}", flush=True)
    print(f"  Tiempo total: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
