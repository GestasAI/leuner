"""F8b — Remedición con métrica de generación (la de primer token ocultaba aciertos).

F8 mostró que la proteína mezclada escribe "Love", "Night", "Perro = dog" en traducción,
pero el primer token no coincidía ("Love" vs "love"). Aquí se mide sobre la respuesta
generada (8 tokens): acierto si la palabra esperada aparece, sin distinguir mayúsculas.
Se mide capital y translate_es_en con las tres proteínas ya guardadas.

Uso:
  python f8b_lenient_measure.py --model ../../modelo/gemma-4-e2b-it-hf
"""
import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import protein_core as pc
from f8_cross_task import TASKS, split, load_f32

HERE = Path(__file__).resolve().parent


def norm(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def lenient_accuracy(model, layers, L, task, use_chat, tok, n_test, seed, vec=None, scale=0.0, max_new=8):
    _, test = split(task, n_test, seed)
    p, s = TASKS[task]["prefix"], TASKS[task]["sep"]
    hits, rows = 0, []
    for x, y in test:
        ids = pc.encode(tok, [], x, p, s, use_chat)
        g = pc.generate(model, layers, L, ids, tok, vec, scale, max_new=max_new)
        ok = norm(y) in norm(g)
        hits += int(ok)
        rows.append({"input": x, "expected": y, "generated": g, "hit": ok})
    return hits / float(len(test)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=str(HERE / "cross_task_lenient_report.json"))
    a = ap.parse_args()

    t0 = time.time()
    tok, model, use_chat = pc.load_model(a.model)
    layers = pc.get_decoder_layers(model)
    L = a.layer
    short = Path(a.model).name
    pdir = HERE / "proteins"
    proteins = {
        "es_en": load_f32(pdir / "translate_es_en.gemma-4-E2B-it.L20.s4.0.f32"),
        "capital": load_f32(pdir / f"capital.{short}.L{L}.s{a.scale}.f32"),
        "mixed": load_f32(pdir / f"mixed_capital_es_en.{short}.L{L}.s{a.scale}.f32"),
    }
    results, details = {}, {}
    print(f"{'tarea':<16}{'base':>7}{'+es_en':>9}{'+capital':>10}{'+mixed':>9}   n  (métrica: palabra esperada en 8 tokens, sin mayúsculas)", flush=True)
    for task in ["capital", "translate_es_en"]:
        row = {}
        row["base"], details[f"{task}+base"] = lenient_accuracy(model, layers, L, task, use_chat, tok, a.n_test, a.seed)
        for name, vec in proteins.items():
            row[name], details[f"{task}+{name}"] = lenient_accuracy(model, layers, L, task, use_chat, tok, a.n_test, a.seed, vec, a.scale)
        row["n"] = len(details[f"{task}+base"])
        results[task] = row
        print(f"{task:<16}{row['base']:>7.2f}{row['es_en']:>9.2f}{row['capital']:>10.2f}{row['mixed']:>9.2f}   {row['n']}", flush=True)

    print("\n== Respuestas generadas: translate_es_en + mixed ==", flush=True)
    for r in details["translate_es_en+mixed"]:
        print(f"  {r['input']:>10} → '{r['generated'][:32]}' | esperado: {r['expected']} | {'OK' if r['hit'] else '--'}", flush=True)
    print("\n== Respuestas generadas: capital + mixed ==", flush=True)
    for r in details["capital+mixed"]:
        print(f"  {r['input']:>10} → '{r['generated'][:32]}' | esperado: {r['expected']} | {'OK' if r['hit'] else '--'}", flush=True)

    Path(a.report).write_text(json.dumps({"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          "model": a.model, "layer": L, "scale": a.scale, "seed": a.seed,
                                          "metric": "expected word in first 8 generated tokens, case/accent-insensitive",
                                          "accuracy": results, "details": details,
                                          "elapsed_s": round(time.time() - t0)}, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nInforme: {a.report}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
