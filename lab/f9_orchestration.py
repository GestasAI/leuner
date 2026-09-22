"""F9 — Orquestación de proteínas: primera dendrita con varias proteínas (borrador de fases 2 y 4).

Cadena: compositor → país → capital → capital en español. Tres proteínas (país del compositor,
capital, traducción EN→ES) y un ENRUTADOR que decide, en cada pregunta, qué proteína inyectar.
El enrutador es una decisión tipada: el modelo elige entre opciones A/B/C/D y devuelve una
probabilidad por opción (softmax de los logits de las letras). No está calibrado: se informa tal cual.

Mide:
  1. Proteína nueva composer_country: base vs +proteína (primer token y generación).
  2. Enrutador: acierto al elegir la proteína correcta para 3 tipos de pregunta.
  3. Cadena completa por compositor: orquestada (enrutador + proteínas) vs base (sin proteínas)
     vs proteína fija equivocada. Acierto por paso y de extremo a extremo.

Uso:
  python f9_orchestration.py --model ../../modelo/gemma-4-e2b-it-hf
"""
import argparse
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch

import protein_core as pc
from f0_datasets import CAPITAL
from f8_cross_task import load_f32, save_protein

HERE = Path(__file__).resolve().parent

COMPOSER_COUNTRY = [
    ("Mozart", "Austria"), ("Schubert", "Austria"), ("Haydn", "Austria"), ("Bruckner", "Austria"),
    ("Beethoven", "Germany"), ("Bach", "Germany"), ("Brahms", "Germany"), ("Wagner", "Germany"),
    ("Handel", "Germany"), ("Schumann", "Germany"), ("Vivaldi", "Italy"), ("Verdi", "Italy"),
    ("Puccini", "Italy"), ("Rossini", "Italy"), ("Monteverdi", "Italy"), ("Paganini", "Italy"),
    ("Debussy", "France"), ("Ravel", "France"), ("Berlioz", "France"), ("Bizet", "France"),
    ("Satie", "France"), ("Tchaikovsky", "Russia"), ("Rachmaninoff", "Russia"), ("Mussorgsky", "Russia"),
    ("Shostakovich", "Russia"), ("Chopin", "Poland"), ("Sibelius", "Finland"), ("Grieg", "Norway"),
    ("Liszt", "Hungary"), ("Bartok", "Hungary"), ("Falla", "Spain"), ("Albeniz", "Spain"),
    ("Granados", "Spain"), ("Nielsen", "Denmark"), ("Smetana", "Czechia"), ("Dvorak", "Czechia"),
]
CAPITAL_ES = {  # capital en inglés → en español (paso 3)
    "Vienna": "Viena", "Berlin": "Berlin", "Rome": "Roma", "Paris": "Paris", "Moscow": "Moscu",
    "Warsaw": "Varsovia", "Helsinki": "Helsinki", "Oslo": "Oslo", "Budapest": "Budapest",
    "Madrid": "Madrid", "Copenhagen": "Copenhague", "Prague": "Praga",
}
COUNTRY_CAPITAL = dict(CAPITAL)
COUNTRY_CAPITAL.update({"Czechia": "Prague"})

STEPS = {
    "composer_country": {"prefix": "Country of composer ", "sep": ":", "option": "A"},
    "capital":          {"prefix": "Capital of ",          "sep": ":", "option": "B"},
    "translate":        {"prefix": "English to Spanish: ", "sep": "=", "option": "C"},
}
ROUTER_PROMPT = ("Which capability answers this question? "
                 "A) country of a composer, B) capital of a country, C) translate a word to Spanish, D) none. "
                 "Question: \"{q}\". Answer with only the letter.")


def norm(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower()


def first_word(text):
    t = re.sub(r"[*_`\"'()]", " ", text).strip()
    t = t.split("\n")[0]
    m = re.match(r"[A-Za-zÀ-ÿ]+(?:\s[A-Z][A-Za-zÀ-ÿ]+)?", t)
    return m.group(0).strip() if m else t.split(" ")[0]


def split(data, n_test, seed):
    d = list(data)
    random.Random(seed).shuffle(d)
    n = min(n_test, len(d) // 2)
    return d[n:], d[:n]


def route(model, tok, question, use_chat):
    """Decisión tipada: probabilidades sobre A/B/C/D leídas de los logits del primer token."""
    msgs = [{"role": "user", "content": ROUTER_PROMPT.format(q=question)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) if use_chat \
        else ROUTER_PROMPT.format(q=question)
    ids = tok(text, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :].float()
    opt_ids = {o: tok.encode(o, add_special_tokens=False)[0] for o in "ABCD"}
    sub = torch.stack([logits[opt_ids[o]] for o in "ABCD"])
    probs = torch.softmax(sub, dim=0).tolist()
    choice = "ABCD"[int(np.argmax(probs))]
    return choice, {o: round(p, 3) for o, p in zip("ABCD", probs)}


def answer(model, layers, L, tok, use_chat, step, x, vec, scale, max_new=6):
    p, s = STEPS[step]["prefix"], STEPS[step]["sep"]
    ids = pc.encode(tok, [], x, p, s, use_chat)
    return pc.generate(model, layers, L, ids, tok, vec, scale, max_new=max_new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--scale", type=float, default=4.0)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--n-extract", type=int, default=10)
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=str(HERE / "orchestration_report.json"))
    a = ap.parse_args()

    t0 = time.time()
    tok, model, use_chat = pc.load_model(a.model)
    layers = pc.get_decoder_layers(model)
    L = a.layer
    short = Path(a.model).name
    pdir = HERE / "proteins"

    # Proteínas existentes.
    v_cap = load_f32(pdir / f"capital.{short}.L{L}.s{a.scale}.f32")
    npz = np.load(pdir / "translate_gemma_4_e2b_it.protein.npz", allow_pickle=True)
    v_tr = torch.from_numpy(np.asarray(npz["vector"], dtype=np.float32))
    print("Proteínas cargadas: capital, translate EN→ES", flush=True)

    # 1. Proteína nueva: país del compositor.
    demos, test = split(COMPOSER_COUNTRY, a.n_test, a.seed)
    print(f"Extrayendo proteína composer_country (demos {len(demos)}, test {len(test)}) …", flush=True)
    v_cc = pc.extract_vector(model, layers, L, demos, a.n_extract, a.k_shot,
                             STEPS["composer_country"]["prefix"], STEPS["composer_country"]["sep"],
                             use_chat, tok, random.Random(a.seed))
    cc = {"base": 0, "protein": 0, "rows": []}
    for x, y in test:
        g0 = answer(model, layers, L, tok, use_chat, "composer_country", x, None, 0.0)
        g1 = answer(model, layers, L, tok, use_chat, "composer_country", x, v_cc, a.scale)
        cc["base"] += int(norm(y) in norm(g0)); cc["protein"] += int(norm(y) in norm(g1))
        cc["rows"].append({"composer": x, "expected": y, "base": g0, "protein": g1})
    n = len(test)
    print(f"composer_country: base {cc['base']/n:.2f} → +proteína {cc['protein']/n:.2f}  (n={n}, palabra en 6 tokens)", flush=True)
    save_protein(v_cc, f"composer_country.{short}.L{L}.s{a.scale}",
                 {"model": a.model, "layer": L, "scale": a.scale, "capability": "composer_country",
                  "extraction": {"method": "mean-activation v0", "n_extract": a.n_extract, "k_shot": a.k_shot, "seed": a.seed},
                  "efficacy": {"accuracy_base": round(cc["base"]/n, 4), "accuracy_protein": round(cc["protein"]/n, 4),
                               "metric": "expected word in 6 generated tokens", "n_test": n}}, pdir)
    proteins = {"A": v_cc, "B": v_cap, "C": v_tr, "D": None}

    # 2. Enrutador: decisión tipada con probabilidades.
    print("\n== Enrutador (decisión tipada A/B/C/D) ==", flush=True)
    router_rows, router_hits = [], 0
    questions = [("A", f"Country of composer {x}:") for x, _ in test] + \
                [("B", f"Capital of {y}:") for _, y in test] + \
                [("C", f"English to Spanish: {c} =") for c in list(CAPITAL_ES)[:n]]
    for expected, q in questions:
        choice, probs = route(model, tok, q, use_chat)
        ok = choice == expected
        router_hits += int(ok)
        router_rows.append({"question": q, "expected": expected, "choice": choice, "probs": probs, "hit": ok})
    print(f"acierto del enrutador: {router_hits}/{len(questions)} = {router_hits/len(questions):.2f}", flush=True)
    for r in router_rows[:6]:
        print(f"  {r['question']:<40} → {r['choice']} {r['probs']} {'OK' if r['hit'] else '--'}", flush=True)

    # 3. Cadena completa por compositor.
    print("\n== Cadena compositor → país → capital → español ==", flush=True)
    chain = {"orchestrated": [], "base": [], "wrong_fixed": []}
    for x, country in test:
        cap = COUNTRY_CAPITAL.get(country, "")
        cap_es = CAPITAL_ES.get(cap, cap)
        for mode in chain:
            steps, cur, hits = [], x, []
            for step, expected in [("composer_country", country), ("capital", cap), ("translate", cap_es)]:
                q = f"{STEPS[step]['prefix']}{cur}{STEPS[step]['sep']}"
                if mode == "orchestrated":
                    choice, probs = route(model, tok, q, use_chat)
                    vec = proteins[choice]
                elif mode == "base":
                    choice, probs, vec = "-", {}, None
                else:  # proteína fija equivocada: siempre la de traducción
                    choice, probs, vec = "C", {}, v_tr
                g = answer(model, layers, L, tok, use_chat, step, cur, vec, a.scale if vec is not None else 0.0)
                out = first_word(g)
                ok = bool(expected) and norm(expected) in norm(g)
                hits.append(ok)
                steps.append({"step": step, "input": cur, "route": choice, "probs": probs, "generated": g,
                              "parsed": out, "expected": expected, "hit": ok})
                cur = out
            chain[mode].append({"composer": x, "steps": steps, "end_to_end": all(hits), "step_hits": hits})
    summary = {}
    for mode, rows in chain.items():
        per_step = [sum(r["step_hits"][i] for r in rows) / len(rows) for i in range(3)]
        e2e = sum(r["end_to_end"] for r in rows) / len(rows)
        summary[mode] = {"step_country": per_step[0], "step_capital": per_step[1], "step_spanish": per_step[2], "end_to_end": e2e}
        print(f"{mode:<13} país {per_step[0]:.2f} · capital {per_step[1]:.2f} · español {per_step[2]:.2f} · extremo a extremo {e2e:.2f}", flush=True)
    print("\nEjemplos orquestados:", flush=True)
    for r in chain["orchestrated"][:5]:
        path = " → ".join(f"{s['parsed']}[{s['route']}]" for s in r["steps"])
        print(f"  {r['composer']:<12} → {path}   {'OK' if r['end_to_end'] else '--'}", flush=True)

    Path(a.report).write_text(json.dumps({
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": a.model, "layer": L,
        "scale": a.scale, "seed": a.seed, "n_test": n,
        "composer_country": {"base": cc["base"]/n, "protein": cc["protein"]/n, "rows": cc["rows"]},
        "router": {"accuracy": router_hits/len(questions), "rows": router_rows},
        "chain_summary": summary, "chain": chain, "elapsed_s": round(time.time()-t0)}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nInforme: {a.report}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
