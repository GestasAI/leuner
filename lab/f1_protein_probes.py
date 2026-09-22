"""F1 probes — la batería de pruebas antes del estructural.

Corre TODO lo que conviene saber antes de montar neuronas. Resultados malos o pequeños
también valen: el objetivo es **saber**, no aprobar. Pruebas:
  P1  Generación multi-token  — ¿da una respuesta completa y fluida, no solo 1 token?
  P2  Daño colateral          — inyectar la proteína de A, ¿empeora la tarea B?
  P3  Composición             — A + B inyectadas juntas, ¿siguen funcionando las dos?
  P4  Capacidad propia (F2)   — una proteína de Launer (traducción EN→ES), base vs vector.

Uso:
  python f1_protein_probes.py --model google/gemma-2b-it
  python f1_protein_probes.py --model google/gemma-4-E2B-it --layer 20 --scale 4.0

(Inyección on-device en LiteRT = prueba #1 del plan; NO se cubre aquí, va en el móvil.)
"""
import argparse
import random
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import protein_core as pc
from f0_datasets import TASKS as BASE

# Capacidad propia para P4 (F2): traducción EN→ES de una palabra (saber-hacer, no un hecho del usuario).
TRANSLATE = [
    ("dog", "perro"), ("cat", "gato"), ("house", "casa"), ("water", "agua"),
    ("fire", "fuego"), ("sun", "sol"), ("book", "libro"), ("tree", "arbol"),
    ("red", "rojo"), ("blue", "azul"), ("green", "verde"), ("black", "negro"),
    ("big", "grande"), ("day", "dia"), ("night", "noche"), ("man", "hombre"),
    ("woman", "mujer"), ("food", "comida"), ("friend", "amigo"), ("love", "amor"),
    ("time", "tiempo"), ("road", "camino"), ("city", "ciudad"), ("sea", "mar"),
    ("hand", "mano"), ("eye", "ojo"), ("door", "puerta"), ("key", "llave"),
]
DATA = {"antonym": BASE["antonym"], "capital": BASE["capital"], "translate": TRANSLATE}
TPL = {
    "antonym":   {"prefix": "opposite of ", "sep": ":"},
    "capital":   {"prefix": "Capital of ",  "sep": ":"},
    "translate": {"prefix": "English to Spanish: ", "sep": "="},
}


def split(task, n_test, seed):
    d = list(DATA[task])
    random.Random(seed).shuffle(d)
    return d[n_test:], d[:n_test]          # demos, test


def vec_for(model, layers, L, task, k, n_ext, use_chat, tok, seed):
    demos, _ = split(task, args_ntest, seed)
    p, s = TPL[task]["prefix"], TPL[task]["sep"]
    return pc.extract_vector(model, layers, L, demos, n_ext, k, p, s, use_chat, tok, random.Random(seed))


def acc(model, layers, L, task, use_chat, tok, seed, vec=None, scale=0.0):
    _, test = split(task, args_ntest, seed)
    p, s = TPL[task]["prefix"], TPL[task]["sep"]
    hits = 0
    for x, y in test:
        ids = pc.encode(tok, [], x, p, s, use_chat)
        pred = pc.predict_first(model, layers, L, ids, vec, scale)
        if pred == pc.first_token(tok, y, use_chat):
            hits += 1
    return hits / float(len(test))


def main():
    global args_ntest
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-2b-it")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--scale", type=float, default=-1.0)
    ap.add_argument("--k-shot", type=int, default=5)
    ap.add_argument("--n-extract", type=int, default=20)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    args_ntest = a.n_test

    tok, model, use_chat = pc.load_model(a.model)
    layers = pc.get_decoder_layers(model)
    L = a.layer if a.layer >= 0 else pc.default_layer(len(layers))
    scale = a.scale if a.scale >= 0 else (4.0 if use_chat else 1.0)
    print(f"\nModelo: {a.model} | capas: {len(layers)} | capa L={L} | scale={scale} | "
          f"chat={'sí' if use_chat else 'no'} | seed={a.seed}\n")

    # ---------- P4 / F2: capacidad propia (traducción) ----------
    print("== P4 · Capacidad propia (F2): traducción EN→ES ==")
    v_tr = vec_for(model, layers, L, "translate", a.k_shot, a.n_extract, use_chat, tok, a.seed)
    base_tr = acc(model, layers, L, "translate", use_chat, tok, a.seed)
    vec_tr = acc(model, layers, L, "translate", use_chat, tok, a.seed, v_tr, scale)
    print(f"   base {base_tr:.2f} → +proteína {vec_tr:.2f}  (ganancia {vec_tr-base_tr:+.2f})\n")

    # ---------- P1: generación multi-token ----------
    print("== P1 · Generación multi-token (¿respuesta completa y fluida?) ==")
    _, test = split("translate", a.n_test, a.seed)
    for x, y in test[:5]:
        ids = pc.encode(tok, [], x, TPL["translate"]["prefix"], TPL["translate"]["sep"], use_chat)
        g0 = pc.generate(model, layers, L, ids, tok)
        g1 = pc.generate(model, layers, L, ids, tok, v_tr, scale)
        print(f"   {x:>8} → base:'{g0[:24]}' | +prot:'{g1[:24]}' | esperado:{y}")
    print()

    # ---------- P2: daño colateral ----------
    print("== P2 · Daño colateral (la proteína de 'capital', ¿estropea 'antonym'?) ==")
    v_cap = vec_for(model, layers, L, "capital", a.k_shot, a.n_extract, use_chat, tok, a.seed)
    ant_base = acc(model, layers, L, "antonym", use_chat, tok, a.seed)
    ant_with_cap = acc(model, layers, L, "antonym", use_chat, tok, a.seed, v_cap, scale)
    cap_with_cap = acc(model, layers, L, "capital", use_chat, tok, a.seed, v_cap, scale)
    print(f"   capital con su proteína: {cap_with_cap:.2f}")
    print(f"   antonym  base {ant_base:.2f} → con proteína-de-capital {ant_with_cap:.2f} "
          f"(caída {ant_with_cap-ant_base:+.2f})  ← si cae mucho = colateral\n")

    # ---------- P3: composición ----------
    print("== P3 · Composición (antonym + capital inyectadas juntas) ==")
    v_ant = vec_for(model, layers, L, "antonym", a.k_shot, a.n_extract, use_chat, tok, a.seed)
    comb = v_ant + v_cap
    ant_alone = acc(model, layers, L, "antonym", use_chat, tok, a.seed, v_ant, scale)
    cap_alone = acc(model, layers, L, "capital", use_chat, tok, a.seed, v_cap, scale)
    ant_comb = acc(model, layers, L, "antonym", use_chat, tok, a.seed, comb, scale)
    cap_comb = acc(model, layers, L, "capital", use_chat, tok, a.seed, comb, scale)
    print(f"   antonym: sola {ant_alone:.2f} → compuesta {ant_comb:.2f}")
    print(f"   capital: sola {cap_alone:.2f} → compuesta {cap_comb:.2f}")
    print("   (si las dos se mantienen ≈ → componen; si se hunden → interfieren)\n")

    print("== FIN. Resultados crudos arriba — buenos o malos, ahora lo SABEMOS. ==")


if __name__ == "__main__":
    main()
