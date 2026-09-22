"""F3 — Supervivencia de la proteina translate.en_es a cuantizacion int4/int8/bfloat16.

Estrategia de degradacion elegante:
  1. bitsandbytes load_in_4bit=True  (requiere CUDA)
  2. bitsandbytes load_in_8bit=True  (requiere CUDA)
  3. torch.bfloat16                  (CPU — proxy de cuantizacion documentado)

Gate de supervivencia: accuracy_quant >= 0.70 * accuracy_bf16 -> SUPERVIVE

Salida: proteins/int4_survival_report.json
"""
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
SEED = 0
SURVIVAL_THRESHOLD = 0.70


def split(n_test, seed):
    d = list(DATA[TASK])
    random.Random(seed).shuffle(d)
    return d[n_test:], d[:n_test]


def measure_accuracy(model, tok, use_chat, vec=None):
    """Mide first-token accuracy en el test set fijo (seed=SEED)."""
    layers = pc.get_decoder_layers(model)
    _, test = split(N_TEST, SEED)
    p, s = TPL[TASK]["prefix"], TPL[TASK]["sep"]
    hits = 0
    for x, y in test:
        ids = pc.encode(tok, [], x, p, s, use_chat)
        expected = pc.first_token(tok, y, use_chat)
        pred = pc.predict_first(model, layers, LAYER, ids, vec, SCALE if vec is not None else 0.0)
        if pred == expected:
            hits += 1
    return hits / float(len(test))


def extract_protein(model, tok, use_chat):
    """Extrae el vector de tarea con los mismos parametros que F2."""
    layers = pc.get_decoder_layers(model)
    demos, _ = split(N_TEST, SEED)
    p, s = TPL[TASK]["prefix"], TPL[TASK]["sep"]
    vec = pc.extract_vector(
        model, layers, LAYER, demos, N_EXTRACT, K_SHOT,
        p, s, use_chat, tok, random.Random(SEED)
    )
    return vec


def try_load_quantized():
    """Intenta cargar el modelo con distintos niveles de cuantizacion.
    Devuelve (model, tok, use_chat, quant_label, quant_note).
    """
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_chat = getattr(tok, "chat_template", None) is not None

    # Intento 1: bitsandbytes int4 (requiere CUDA)
    if torch.cuda.is_available():
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            print("  CUDA disponible. Intentando load_in_4bit=True ...", flush=True)
            cfg = BitsAndBytesConfig(load_in_4bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=cfg, device_map="auto", low_cpu_mem_usage=True
            )
            model.eval()
            print("  Cargado en int4 (bitsandbytes, CUDA).", flush=True)
            return model, tok, use_chat, "int4", "bitsandbytes NF4, CUDA"
        except Exception as e:
            print(f"  load_in_4bit fallo: {e}", flush=True)

        # Intento 2: bitsandbytes int8
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            print("  Intentando load_in_8bit=True ...", flush=True)
            cfg = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, quantization_config=cfg, device_map="auto", low_cpu_mem_usage=True
            )
            model.eval()
            print("  Cargado en int8 (bitsandbytes, CUDA).", flush=True)
            return model, tok, use_chat, "int8", "bitsandbytes int8, CUDA"
        except Exception as e:
            print(f"  load_in_8bit fallo: {e}", flush=True)
    else:
        print("  Sin CUDA — bitsandbytes requiere GPU. Saltando int4/int8.", flush=True)

    # Intento 3: bfloat16 en CPU (proxy documentado)
    print("  Proxy: cargando en bfloat16 CPU (precision reducida vs float32).", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    note = ("PROXY bfloat16 CPU: sin CUDA no es posible int4/int8 real con bitsandbytes. "
            "bfloat16 tiene 3 bits de mantisa vs float32 (23 bits) — captura efectos de "
            "precision reducida pero NO es cuantizacion post-entrenamiento (PTQ).")
    return model, tok, use_chat, "bfloat16_proxy", note


def main():
    print(f"\n=== F3 — Supervivencia a cuantizacion ===", flush=True)
    print(f"Modelo: {MODEL_ID} | Tarea: {TASK} | Capa: {LAYER} | Scale: {SCALE}", flush=True)
    t0 = time.time()

    # --- Cargar modelo referencia (bfloat16, mismo que en F2) ---
    print("\n[1/4] Cargando modelo referencia bfloat16 ...", flush=True)
    tok_ref = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok_ref.pad_token is None:
        tok_ref.pad_token = tok_ref.eos_token
    use_chat_ref = getattr(tok_ref, "chat_template", None) is not None
    model_ref = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model_ref.eval()

    print("\n[2/4] Extrayendo proteina y midiendo accuracy en referencia bfloat16 ...", flush=True)
    vec_ref = extract_protein(model_ref, tok_ref, use_chat_ref)
    base_ref = measure_accuracy(model_ref, tok_ref, use_chat_ref, vec=None)
    prot_ref = measure_accuracy(model_ref, tok_ref, use_chat_ref, vec=vec_ref)
    print(f"  Referencia bfloat16: base={base_ref:.2f} | +proteina={prot_ref:.2f} "
          f"(ganancia {prot_ref - base_ref:+.2f})", flush=True)

    # Liberar modelo referencia antes de cargar el cuantizado
    del model_ref
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Cargar modelo cuantizado (o proxy) ---
    print("\n[3/4] Cargando modelo cuantizado (o proxy) ...", flush=True)
    model_q, tok_q, use_chat_q, quant_label, quant_note = try_load_quantized()

    print("\n[4/4] Extrayendo proteina en modelo cuantizado y midiendo accuracy ...", flush=True)
    vec_q = extract_protein(model_q, tok_q, use_chat_q)
    base_q = measure_accuracy(model_q, tok_q, use_chat_q, vec=None)
    prot_q = measure_accuracy(model_q, tok_q, use_chat_q, vec=vec_q)
    print(f"  Cuantizado ({quant_label}): base={base_q:.2f} | +proteina={prot_q:.2f} "
          f"(ganancia {prot_q - base_q:+.2f})", flush=True)

    # --- Gate de supervivencia ---
    threshold = SURVIVAL_THRESHOLD * prot_ref
    survives = prot_q >= threshold
    delta = prot_q - prot_ref
    status = "SUPERVIVE" if survives else "NO_SUPERVIVE"

    print(f"\n  Gate supervivencia: {prot_q:.2f} >= {threshold:.2f} "
          f"(70% de {prot_ref:.2f}) -> {status}", flush=True)
    print(f"  Delta cuantizado vs referencia: {delta:+.2f}", flush=True)

    elapsed = time.time() - t0

    # --- Guardar report ---
    report = {
        "task": TASK,
        "model": MODEL_ID,
        "layer": LAYER,
        "scale": SCALE,
        "seed": SEED,
        "n_extract": N_EXTRACT,
        "n_test": N_TEST,
        "reference": {
            "dtype": "bfloat16",
            "device": "cpu",
            "accuracy_base": round(base_ref, 4),
            "accuracy_protein": round(prot_ref, 4),
            "gain": round(prot_ref - base_ref, 4),
        },
        "quantized": {
            "quant_label": quant_label,
            "quant_note": quant_note,
            "accuracy_base": round(base_q, 4),
            "accuracy_protein": round(prot_q, 4),
            "gain": round(prot_q - base_q, 4),
            "delta_vs_reference": round(delta, 4),
        },
        "gate": {
            "threshold_factor": SURVIVAL_THRESHOLD,
            "threshold_absolute": round(threshold, 4),
            "status": status,
        },
        "elapsed_s": round(elapsed, 1),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limitation": (
            "Sin CUDA, la cuantizacion real (bitsandbytes NF4/int8) no esta disponible. "
            "El resultado bfloat16 mide precision reducida, no PTQ. "
            "Para el test real de int4 se requiere GPU con soporte CUDA."
        ) if quant_label == "bfloat16_proxy" else None,
    }

    out = Path("proteins")
    out.mkdir(exist_ok=True)
    out_path = out / "int4_survival_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Guardado: {out_path}", flush=True)
    print(f"  Tiempo total: {elapsed:.1f}s", flush=True)
    print(f"\n  RESULTADO: {status} ({quant_label})", flush=True)
    if quant_label == "bfloat16_proxy":
        print("  AVISO: resultado es proxy bfloat16 (sin GPU CUDA). "
              "Test real de int4 pendiente.", flush=True)


if __name__ == "__main__":
    main()
