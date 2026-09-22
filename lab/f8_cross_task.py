"""F8 — Prueba cruzada: ¿una proteína activa capacidades distintas de la suya?

Pregunta (hipótesis D4, "proteína polivalente"): el receptor (capa 20) es común a todas
las capacidades, pero cada proteína medida hasta hoy fue específica. ¿Qué pasa si a la
proteína de traducción ES→EN le pedimos capitales, y a la de capitales le pedimos
antónimos, continentes, notas musicales o el número siguiente? Y si extraemos una
proteína de demostraciones MEZCLADAS (capitales + traducción), ¿activa las dos?

Mide, para cada tarea, el acierto zero-shot (primer token) en cuatro condiciones:
  base · +proteína ES→EN · +proteína capitales · +proteína mezclada
y muestra generaciones cortas en las celdas cruzadas. No aprueba ni suspende: informa.

Uso (modelo local ya descomprimido):
  python f8_cross_task.py --model ../../modelo/gemma-4-e2b-it-hf
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
from f0_datasets import TASKS as BASE
from f5_new_proteins import TRANSLATE_ES_EN

HERE = Path(__file__).resolve().parent

# ─── Tareas nuevas (abstractas o de otro dominio) ──────────────────────────────
CONTINENT = [
    ("France", "Europe"), ("Spain", "Europe"), ("Italy", "Europe"), ("Germany", "Europe"),
    ("Portugal", "Europe"), ("Greece", "Europe"), ("Poland", "Europe"), ("Norway", "Europe"),
    ("Sweden", "Europe"), ("Ireland", "Europe"), ("Hungary", "Europe"), ("Belgium", "Europe"),
    ("Japan", "Asia"), ("China", "Asia"), ("India", "Asia"), ("Thailand", "Asia"),
    ("Vietnam", "Asia"), ("Iran", "Asia"), ("Iraq", "Asia"), ("Jordan", "Asia"),
    ("Lebanon", "Asia"), ("Nepal", "Asia"), ("Korea", "Asia"), ("Pakistan", "Asia"),
    ("Egypt", "Africa"), ("Kenya", "Africa"), ("Morocco", "Africa"), ("Nigeria", "Africa"),
    ("Ethiopia", "Africa"), ("Ghana", "Africa"), ("Senegal", "Africa"), ("Tanzania", "Africa"),
    ("Australia", "Oceania"), ("Fiji", "Oceania"), ("Samoa", "Oceania"), ("Tonga", "Oceania"),
    ("Peru", "South"), ("Chile", "South"), ("Brazil", "South"), ("Argentina", "South"),
    ("Colombia", "South"), ("Canada", "North"), ("Cuba", "North"), ("Guatemala", "North"),
]
NEXT_NOTE = [
    ("C", "D"), ("D", "E"), ("E", "F"), ("F", "G"), ("G", "A"), ("A", "B"), ("B", "C"),
    ("Do", "Re"), ("Re", "Mi"), ("Mi", "Fa"), ("Fa", "Sol"), ("Sol", "La"), ("La", "Si"), ("Si", "Do"),
]
NEXT_NUMBER = [
    ("one", "two"), ("two", "three"), ("three", "four"), ("four", "five"), ("five", "six"),
    ("six", "seven"), ("seven", "eight"), ("eight", "nine"), ("nine", "ten"), ("ten", "eleven"),
    ("eleven", "twelve"), ("twelve", "thirteen"), ("thirteen", "fourteen"), ("fourteen", "fifteen"),
    ("fifteen", "sixteen"), ("sixteen", "seventeen"), ("seventeen", "eighteen"), ("eighteen", "nineteen"),
    ("nineteen", "twenty"), ("twenty", "twenty"),
]

TASKS = {
    "capital":         {"data": BASE["capital"],    "prefix": "Capital of ",              "sep": ":"},
    "translate_es_en": {"data": TRANSLATE_ES_EN,    "prefix": "Spanish to English: ",     "sep": "="},
    "antonym":         {"data": BASE["antonym"],    "prefix": "opposite of ",             "sep": ":"},
    "continent":       {"data": CONTINENT,          "prefix": "Continent of ",            "sep": ":"},
    "next_note":       {"data": NEXT_NOTE,          "prefix": "Next musical note after ", "sep": ":"},
    "next_number":     {"data": NEXT_NUMBER,        "prefix": "Number after ",            "sep": ":"},
}


# ─── Utilidades ──────────────────────────────────────────────────────────────
def split(task, n_test, seed):
    d = list(TASKS[task]["data"])
    random.Random(seed).shuffle(d)
    n = min(n_test, len(d) // 2)
    return d[n:], d[:n]


def load_f32(path, dims=1536):
    arr = np.frombuffer(Path(path).read_bytes(), dtype="<f4")
    if arr.shape[0] != dims:
        raise ValueError(f"{path}: {arr.shape[0]} dims, esperaba {dims}")
    return torch.from_numpy(arr.copy())


def accuracy(model, layers, L, task, use_chat, tok, n_test, seed, vec=None, scale=0.0):
    _, test = split(task, n_test, seed)
    p, s = TASKS[task]["prefix"], TASKS[task]["sep"]
    hits = 0
    for x, y in test:
        ids = pc.encode(tok, [], x, p, s, use_chat)
        if pc.predict_first(model, layers, L, ids, vec, scale) == pc.first_token(tok, y, use_chat):
            hits += 1
    return hits / float(len(test))


def samples(model, layers, L, task, use_chat, tok, n_test, seed, vec, scale, n=3):
    _, test = split(task, n_test, seed)
    p, s = TASKS[task]["prefix"], TASKS[task]["sep"]
    out = []
    for x, y in test[:n]:
        ids = pc.encode(tok, [], x, p, s, use_chat)
        g0 = pc.generate(model, layers, L, ids, tok, max_new=8)
        g1 = pc.generate(model, layers, L, ids, tok, vec, scale, max_new=8)
        out.append({"input": x, "expected": y, "base": g0, "protein": g1})
    return out


def encode_mixed(tok, demos, query, use_chat):
    """Demos de varias tareas, cada una con su prefijo y separador. query = (prefix, x, sep)."""
    if use_chat:
        msgs = []
        for p, x, s, y in demos:
            msgs.append({"role": "user", "content": f"{p}{x}{s}"})
            msgs.append({"role": "assistant", "content": y})
        p, x, s = query
        msgs.append({"role": "user", "content": f"{p}{x}{s}"})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt").input_ids
    text = "".join(f"{p}{x}{s} {y}\n" for p, x, s, y in demos)
    p, x, s = query
    return tok(text + f"{p}{x}{s}", return_tensors="pt").input_ids


def extract_mixed(model, layers, L, tasks, n_prompts, k_shot, use_chat, tok, n_test, seed):
    """Proteína polivalente: media de activaciones sobre prompts con demos mezcladas."""
    pool = []
    for t in tasks:
        demos, _ = split(t, n_test, seed)
        p, s = TASKS[t]["prefix"], TASKS[t]["sep"]
        pool += [(p, x, s, y) for x, y in demos]
    rng = random.Random(seed)
    acc = None
    for _ in range(n_prompts):
        chosen = rng.sample(pool, k_shot + 1)
        demos, (qp, qx, qs, _) = chosen[:k_shot], chosen[k_shot]
        ids = encode_mixed(tok, demos, (qp, qx, qs), use_chat)
        with pc.capture(layers[L]) as box, torch.no_grad():
            model(ids)
        acc = box[0] if acc is None else acc + box[0]
    return acc / float(n_prompts)


def save_protein(vec, name, meta, out_dir):
    arr = vec.to(torch.float32).cpu().numpy().astype("<f4")
    raw = arr.tobytes()
    sha = hashlib.sha256(raw).hexdigest()
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    (out / f"{name}.f32").write_bytes(raw)
    meta = dict(meta, name=name, dims=int(arr.shape[0]), dtype="float32", byte_order="little-endian",
                format="raw .f32 (dims * 4 bytes)", sha256=sha,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                use="Inyectar: hidden[capa L, ultimo token] += scale * vector")
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return sha


# ─── Programa ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ruta local o id HF de Gemma 4 E2B-it")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--n-extract", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--es-en", default=str(HERE / "proteins" / "translate_es_en.gemma-4-E2B-it.L20.s4.0.f32"))
    ap.add_argument("--out-dir", default=str(HERE / "proteins"))
    ap.add_argument("--report", default=str(HERE / "cross_task_report.json"))
    a = ap.parse_args()

    t0 = time.time()
    tok, model, use_chat = pc.load_model(a.model)
    layers = pc.get_decoder_layers(model)
    L = a.layer
    print(f"Modelo cargado en {time.time()-t0:.0f}s | capas {len(layers)} | L={L} | scale={a.scale} | "
          f"chat={'sí' if use_chat else 'no'} | seed={a.seed}", flush=True)

    # Proteína ES→EN: artefacto ya guardado (11-jun-2026).
    v_es_en = load_f32(a.es_en)
    print(f"Proteína ES→EN cargada de {a.es_en}", flush=True)

    # Proteína de capitales: se extrae ahora (no estaba guardada) y se guarda.
    print("Extrayendo proteína de capitales …", flush=True)
    demos, _ = split("capital", a.n_test, a.seed)
    v_cap = pc.extract_vector(model, layers, L, demos, a.n_extract, a.k_shot,
                              TASKS["capital"]["prefix"], TASKS["capital"]["sep"], use_chat, tok,
                              random.Random(a.seed))

    # Proteína mezclada (D4): capitales + traducción ES→EN en las mismas demostraciones.
    print("Extrayendo proteína mezclada (capital + translate_es_en) …", flush=True)
    v_mix = extract_mixed(model, layers, L, ["capital", "translate_es_en"], a.n_extract, a.k_shot,
                          use_chat, tok, a.n_test, a.seed)

    proteins = {"es_en": v_es_en, "capital": v_cap, "mixed": v_mix}
    cos = {}
    for i in proteins:
        for j in proteins:
            if i < j:
                cos[f"{i}~{j}"] = round(float(torch.nn.functional.cosine_similarity(
                    proteins[i].float(), proteins[j].float(), dim=0)), 4)
    print("Similitud coseno entre proteínas:", cos, flush=True)

    results = {}
    print("\n== Acierto zero-shot (primer token) ==", flush=True)
    print(f"{'tarea':<16}{'base':>7}{'+es_en':>9}{'+capital':>10}{'+mixed':>9}   n", flush=True)
    for task in TASKS:
        _, test = split(task, a.n_test, a.seed)
        row = {"n": len(test), "base": accuracy(model, layers, L, task, use_chat, tok, a.n_test, a.seed)}
        for name, vec in proteins.items():
            row[name] = accuracy(model, layers, L, task, use_chat, tok, a.n_test, a.seed, vec, a.scale)
        results[task] = row
        print(f"{task:<16}{row['base']:>7.2f}{row['es_en']:>9.2f}{row['capital']:>10.2f}{row['mixed']:>9.2f}   {row['n']}",
              flush=True)

    print("\n== Generaciones en celdas cruzadas ==", flush=True)
    gens = {}
    for task, pname in [("capital", "es_en"), ("translate_es_en", "capital"), ("antonym", "capital"),
                        ("continent", "capital"), ("next_note", "capital"), ("next_number", "capital"),
                        ("capital", "mixed"), ("translate_es_en", "mixed")]:
        s = samples(model, layers, L, task, use_chat, tok, a.n_test, a.seed, proteins[pname], a.scale)
        gens[f"{task}+{pname}"] = s
        for r in s:
            print(f"  [{task} + {pname}] {r['input']:>10} → base:'{r['base'][:28]}' | prot:'{r['protein'][:28]}' | esperado:{r['expected']}",
                  flush=True)

    # Guardar artefactos nuevos.
    short = Path(a.model).name if Path(a.model).exists() else a.model.split("/")[-1]
    common = {"model": a.model, "layer": L, "scale": a.scale,
              "extraction": {"method": "mean-activation v0 (Hendel 2023)", "n_extract": a.n_extract,
                             "k_shot": a.k_shot, "seed": a.seed}}
    sha_cap = save_protein(v_cap, f"capital.{short}.L{L}.s{a.scale}",
                           dict(common, capability="capital",
                                efficacy={"accuracy_base": results["capital"]["base"],
                                          "accuracy_protein": results["capital"]["capital"],
                                          "metric": "first-token zero-shot", "n_test": results["capital"]["n"]}),
                           a.out_dir)
    sha_mix = save_protein(v_mix, f"mixed_capital_es_en.{short}.L{L}.s{a.scale}",
                           dict(common, capability="mixed: capital + translate_es_en",
                                efficacy={"capital": results["capital"]["mixed"],
                                          "translate_es_en": results["translate_es_en"]["mixed"],
                                          "metric": "first-token zero-shot"}),
                           a.out_dir)

    report = {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": a.model,
              "layer": L, "scale": a.scale, "k_shot": a.k_shot, "n_extract": a.n_extract, "seed": a.seed,
              "cosine": cos, "accuracy": results, "generations": gens,
              "artifacts": {"capital_sha256": sha_cap, "mixed_sha256": sha_mix},
              "elapsed_s": round(time.time() - t0)}
    Path(a.report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme: {a.report}  ({report['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
