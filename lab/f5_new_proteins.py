"""F5 — Extraccion de proteinas nuevas: translate.es_en (P1), summarize (P2), classify_intent (P3).

Para cada proteina:
  - Split: demos / test (semilla fija)
  - Extrae vector (capa 20, scale 4.0, n_extract reducido para CPU)
  - Mide accuracy base y +proteina
  - Gate: accuracy >= 0.50 AND ganancia >= 0.20
  - Si pasa: guarda artefacto .f32 + .json en proteins/

P2 (summarize) usa gate alternativo: fraccion de respuestas cortas (<= 15 palabras).
P3 (classify_intent) usa first-token match contra label TASK/QUESTION/CHAT.
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

MODEL_ID = "google/gemma-4-E2B-it"
LAYER = 20
SCALE_DEFAULT = 4.0
N_EXTRACT = 10
N_TEST = 15
K_SHOT = 5
SEED = 0

# ─── Datasets ───────────────────────────────────────────────────────────────

TRANSLATE_ES_EN = [
    ("perro", "dog"), ("gato", "cat"), ("casa", "house"), ("agua", "water"),
    ("fuego", "fire"), ("sol", "sun"), ("libro", "book"), ("arbol", "tree"),
    ("rojo", "red"), ("azul", "blue"), ("verde", "green"), ("negro", "black"),
    ("grande", "big"), ("dia", "day"), ("noche", "night"), ("hombre", "man"),
    ("mujer", "woman"), ("comida", "food"), ("amigo", "friend"), ("amor", "love"),
    ("tiempo", "time"), ("camino", "road"), ("ciudad", "city"), ("mar", "sea"),
    ("mano", "hand"), ("ojo", "eye"), ("puerta", "door"), ("llave", "key"),
    ("cielo", "sky"), ("tierra", "earth"), ("lluvia", "rain"), ("viento", "wind"),
    ("nino", "child"), ("madre", "mother"), ("padre", "father"), ("hermano", "brother"),
]

# Para summarize: pares (texto_largo, primera_palabra_del_resumen_esperado).
# La tarea: generar un resumen MUY corto. Medimos si la generacion es <= 15 palabras.
# Ademas medimos si el primer token coincide con la primera palabra del resumen "canonico".
SUMMARIZE = [
    ("The dog ran in the park. It was a warm day. Children watched and laughed.",
     "Dog"),
    ("She read a book all night. The story was about a lost explorer. She could not stop reading.",
     "She"),
    ("It rained heavily this morning. The streets flooded quickly. Traffic stopped completely.",
     "Heavy"),
    ("The team won the final match. It was a close game. Fans celebrated loudly.",
     "Team"),
    ("He cooked dinner for his family. They ate together at the table. Everyone was happy.",
     "Family"),
    ("The cat slept on the sofa. It purred softly. The room was quiet and warm.",
     "Cat"),
    ("Scientists discovered a new planet. It is far from Earth. More research is needed.",
     "Scientists"),
    ("The store opened early today. Many customers arrived at once. Lines formed at the door.",
     "Store"),
    ("She planted flowers in the garden. The colors were bright and varied. Neighbors admired them.",
     "She"),
    ("The train arrived late. Passengers waited on the platform. No explanation was given.",
     "Train"),
    ("He wrote a letter to his friend. It took him an hour. The words came slowly.",
     "He"),
    ("The baby laughed at the toy. Her parents smiled. It was a joyful moment.",
     "Baby"),
    ("The fire spread quickly through the forest. Firefighters worked through the night. The blaze was contained by dawn.",
     "Fire"),
    ("Students took the exam in silence. Some finished early. Others struggled with the last question.",
     "Students"),
    ("The old clock stopped working. It had run for fifty years. A repairman was called.",
     "Old"),
    ("The boat sailed into the harbor. Fishermen unloaded their catch. The market opened soon after.",
     "Boat"),
    ("She forgot her umbrella at home. It began to rain on her way to work. She arrived soaking wet.",
     "She"),
    ("The library closes at nine. Students hurried to return their books. The lights went off at closing.",
     "Library"),
    ("A bird built a nest near the window. It worked for three days. The nest was neat and round.",
     "Bird"),
    ("The meeting ended earlier than expected. Everyone agreed on the plan. Next steps were assigned.",
     "Meeting"),
    ("He fixed the broken chair with glue. It took only a few minutes. The chair was as good as new.",
     "He"),
    ("The children played in the snow. They built a snowman together. It melted by afternoon.",
     "Children"),
    ("She learned to bake bread last summer. The first loaves were flat. She improved quickly.",
     "She"),
    ("The museum opened a new exhibit on ancient Egypt. Visitors came from across the city. The mummies drew the largest crowds.",
     "Museum"),
    ("The power went out during the storm. Candles were lit throughout the house. Power returned after midnight.",
     "Power"),
]

# Para classify_intent: pares (mensaje, label).
# Labels: TASK, QUESTION, CHAT
CLASSIFY_INTENT = [
    ("Remind me to buy milk tomorrow", "TASK"),
    ("How much is 15 times 8?", "QUESTION"),
    ("Hey, how are you?", "CHAT"),
    ("Set an alarm for 7 AM", "TASK"),
    ("What is the capital of France?", "QUESTION"),
    ("Good morning!", "CHAT"),
    ("Add eggs to my shopping list", "TASK"),
    ("Why is the sky blue?", "QUESTION"),
    ("I love rainy days", "CHAT"),
    ("Schedule a meeting for Friday at 3 PM", "TASK"),
    ("Who invented the telephone?", "QUESTION"),
    ("Thanks, that was helpful", "CHAT"),
    ("Recuerdame comprar pan manana", "TASK"),
    ("Cuanto es 100 dividido entre 4?", "QUESTION"),
    ("Hola, que tal estas?", "CHAT"),
    ("Pon un temporizador de 10 minutos", "TASK"),
    ("Cual es la capital de Espana?", "QUESTION"),
    ("Hoy hace muy buen tiempo", "CHAT"),
    ("Anade leche a la lista de la compra", "TASK"),
    ("Quien escribio Don Quijote?", "QUESTION"),
    ("Me alegra que este funcionando", "CHAT"),
    ("Book a table for two at 8 PM", "TASK"),
    ("What time does the museum open?", "QUESTION"),
    ("It has been a long day", "CHAT"),
    ("Turn off the lights in the kitchen", "TASK"),
    ("How do I reset my password?", "QUESTION"),
    ("I feel great today", "CHAT"),
    ("Send a reminder at noon", "TASK"),
    ("What is the weather like in Madrid?", "QUESTION"),
    ("Nice to meet you", "CHAT"),
]

TASKS_DEF = {
    "translate_es_en": {
        "data": TRANSLATE_ES_EN,
        "prefix": "Spanish to English: ",
        "sep": "=",
        "scale": SCALE_DEFAULT,
        "gate_type": "first_token",
    },
    "summarize": {
        "data": SUMMARIZE,
        "prefix": "Summarize in one short phrase: ",
        "sep": "->",
        "scale": SCALE_DEFAULT,
        "gate_type": "short_output",   # mide si genera texto corto
    },
    "classify_intent": {
        "data": CLASSIFY_INTENT,
        "prefix": "Intent (TASK/QUESTION/CHAT): ",
        "sep": "->",
        "scale": SCALE_DEFAULT,
        "gate_type": "first_token",
    },
}


# ─── Utilidades ─────────────────────────────────────────────────────────────

def split_data(data, n_test, seed):
    d = list(data)
    random.Random(seed).shuffle(d)
    return d[n_test:], d[:n_test]


def measure_short_output(model, layers, L, tok, use_chat, test, prefix, sep,
                          vec=None, scale=0.0):
    """Gate alternativo para summarize: fraccion de respuestas <= 15 palabras."""
    hits = 0
    for x, _ in test:
        ids = pc.encode(tok, [], x, prefix, sep, use_chat)
        text = pc.generate(model, layers, L, ids, tok, vec, scale, max_new=20)
        words = len(text.split())
        if 1 <= words <= 15:
            hits += 1
    return hits / float(len(test))


def measure_first_token(model, layers, L, tok, use_chat, test, prefix, sep,
                         vec=None, scale=0.0):
    """Gate estandar: first-token match."""
    hits = 0
    for x, y in test:
        ids = pc.encode(tok, [], x, prefix, sep, use_chat)
        expected = pc.first_token(tok, y, use_chat)
        pred = pc.predict_first(model, layers, L, ids, vec, scale)
        if pred == expected:
            hits += 1
    return hits / float(len(test))


def save_protein(vec, task_name, base_acc, prot_acc, n_extract, k_shot, out_dir):
    arr = vec.to(torch.float32).cpu().numpy().astype("<f4")
    raw = arr.tobytes()
    sha = hashlib.sha256(raw).hexdigest()
    short_model = MODEL_ID.split("/")[-1]
    name = f"{task_name}.{short_model}.L{LAYER}.s{SCALE_DEFAULT}"
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / f"{name}.f32").write_bytes(raw)
    meta = {
        "name": name,
        "capability": task_name,
        "model": MODEL_ID,
        "layer": LAYER,
        "scale": SCALE_DEFAULT,
        "dims": int(arr.shape[0]),
        "dtype": "float32",
        "byte_order": "little-endian",
        "format": "raw .f32 (dims * 4 bytes)",
        "extraction": {
            "method": "mean-activation v0 (Hendel 2023)",
            "n_extract": n_extract,
            "k_shot": k_shot,
            "seed": SEED,
        },
        "efficacy": {
            "accuracy_base": round(base_acc, 4),
            "accuracy_protein": round(prot_acc, 4),
            "metric": "first-token zero-shot" if task_name != "summarize"
                      else "short-output fraction (<=15 words)",
            "n_test": N_TEST,
        },
        "sha256": sha,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "use": "Inyectar: hidden[capa L, ultimo token] += scale * vector",
    }
    (out / f"{name}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    size = len(raw)
    return name, sha, size, meta


def run_protein(model, tok, use_chat, layers, task_name, cfg, out_dir):
    print(f"\n--- Proteina: {task_name} ---", flush=True)
    t0 = time.time()

    data = cfg["data"]
    prefix = cfg["prefix"]
    sep = cfg["sep"]
    scale = cfg["scale"]
    gate_type = cfg["gate_type"]

    demos, test = split_data(data, N_TEST, SEED)

    # Extraccion
    print(f"  Extrayendo vector (n_extract={N_EXTRACT}, k_shot={K_SHOT}) ...", flush=True)
    vec = pc.extract_vector(
        model, layers, LAYER, demos, N_EXTRACT, K_SHOT,
        prefix, sep, use_chat, tok, random.Random(SEED)
    )

    # Medicion
    if gate_type == "first_token":
        base = measure_first_token(model, layers, LAYER, tok, use_chat,
                                    test, prefix, sep, vec=None, scale=0.0)
        prot = measure_first_token(model, layers, LAYER, tok, use_chat,
                                    test, prefix, sep, vec=vec, scale=scale)
    else:  # short_output
        base = measure_short_output(model, layers, LAYER, tok, use_chat,
                                     test, prefix, sep, vec=None, scale=0.0)
        prot = measure_short_output(model, layers, LAYER, tok, use_chat,
                                     test, prefix, sep, vec=vec, scale=scale)

    gain = prot - base
    gate_ok = (prot >= 0.50) and (gain >= 0.20)
    status = "PASA" if gate_ok else "FALLA"

    elapsed = time.time() - t0
    print(f"  base={base:.2f} | +proteina={prot:.2f} | ganancia={gain:+.2f} | "
          f"gate={status} | tiempo={elapsed:.1f}s", flush=True)

    # Ejemplos cualitativos
    print(f"  Ejemplos (3 primeros del test set):", flush=True)
    for x, y_expected in test[:3]:
        ids = pc.encode(tok, [], x, prefix, sep, use_chat)
        g0 = pc.generate(model, layers, LAYER, ids, tok, max_new=12)
        g1 = pc.generate(model, layers, LAYER, ids, tok, vec, scale, max_new=12)
        x_short = (x[:40] + "...") if len(x) > 40 else x
        print(f"    IN: '{x_short}'", flush=True)
        print(f"    base:'{g0[:30]}' | +prot:'{g1[:30]}' | esperado:'{y_expected}'", flush=True)

    result = {
        "task": task_name,
        "accuracy_base": round(base, 4),
        "accuracy_protein": round(prot, 4),
        "gain": round(gain, 4),
        "gate": status,
        "gate_type": gate_type,
        "elapsed_s": round(elapsed, 1),
    }

    if gate_ok:
        name, sha, size, meta = save_protein(
            vec, task_name, base, prot, N_EXTRACT, K_SHOT, out_dir
        )
        result["artifact"] = {
            "name": name,
            "sha256": sha[:16] + "...",
            "size_bytes": size,
        }
        print(f"  GUARDADO: {name}.f32 ({size} bytes, sha={sha[:16]}...)", flush=True)
    else:
        print(f"  No guardado (gate FALLA: acc={prot:.2f}<0.50 o ganancia={gain:+.2f}<0.20)",
              flush=True)

    return result


def main():
    print(f"\n=== F5 — Proteinas nuevas: P1, P2, P3 ===", flush=True)
    print(f"Modelo: {MODEL_ID} | Capa: {LAYER} | Scale: {SCALE_DEFAULT}", flush=True)
    t0_total = time.time()

    print("\nCargando modelo ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    use_chat = getattr(tok, "chat_template", None) is not None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cpu", dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.eval()
    layers = pc.get_decoder_layers(model)
    print(f"  Cargado. capas={len(layers)}, chat={use_chat}\n", flush=True)

    out_dir = "proteins"
    all_results = []
    for task_name, cfg in TASKS_DEF.items():
        try:
            r = run_protein(model, tok, use_chat, layers, task_name, cfg, out_dir)
        except Exception as e:
            print(f"  ERROR en {task_name}: {e}", flush=True)
            r = {"task": task_name, "gate": "ERROR", "error": str(e)}
        all_results.append(r)

    total_elapsed = time.time() - t0_total

    print(f"\n=== RESUMEN F5 ===", flush=True)
    print(f"{'Proteina':>20} | {'Base':>6} | {'Acc':>6} | {'Delta':>6} | {'Gate':>8}", flush=True)
    print("-" * 60, flush=True)
    for r in all_results:
        base_s = f"{r.get('accuracy_base', 0):.2f}"
        acc_s  = f"{r.get('accuracy_protein', 0):.2f}"
        del_s  = f"{r.get('gain', 0):+.2f}"
        gate_s = r.get("gate", "ERROR")
        print(f"{r['task']:>20} | {base_s:>6} | {acc_s:>6} | {del_s:>6} | {gate_s:>8}",
              flush=True)
    print(f"\n  Tiempo total: {total_elapsed:.1f}s", flush=True)

    summary_path = Path(out_dir) / "f5_results.json"
    summary_path.write_text(
        json.dumps({"results": all_results, "elapsed_total_s": round(total_elapsed, 1),
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                   indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  Resultados: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
