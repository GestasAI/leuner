"""F0 — Viabilidad de la proteína (vector-tarea / task vector).

Pregunta que responde: ¿existe un vector que, **inyectado** en el residual stream
de un modelo, le haga resolver una tarea **en frío** (zero-shot)?  Si la respuesta
es sí y reproducible, la línea "proteína = function vector" es viable a esta escala.

Método (v0 — robusto, agnóstico de arquitectura):
  1. Construir prompts ICL (k demos + consulta) de la tarea.
  2. Con un forward hook, leer el hidden state del ÚLTIMO token en la capa L.
  3. Promediar entre prompts → el **vector-tarea** (proteína v0).
  4. En un prompt ZERO-SHOT, sumar el vector en la capa L (hook de inyección) y
     medir si el modelo acierta — comparado con no sumarlo.

Versión v0 usa media de activaciones (Hendel et al. 2023).
Versión v1 (Todd & Bau 2023) añade mediación causal por cabeza; ver Launer_Lab_Primera_Proteina.md.

Uso:
  python f0_function_vector.py --smoke                      # gpt2 en segundos
  python f0_function_vector.py --model google/gemma-2b-it   # baseline Gemma 1
  python f0_function_vector.py --model google/gemma-4-E2B-it --task antonym
"""

import argparse
import random
import sys
from contextlib import contextmanager

# Forzar UTF-8 en Windows (consola usa CP1252 por defecto)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoTokenizer

from f0_datasets import TASKS, TASK_TEMPLATES

_PROMPT_PREFIX = ""
_PROMPT_SEP = " ->"
_USE_CHAT_ALT = False  # alternating user/assistant turns para IT models

# ──────────────────────────────────────────────────────────────
# Carga del modelo (soporta CausalLM y ConditionalGeneration)
# ──────────────────────────────────────────────────────────────

def load_model(model_id: str):
    """Carga el tokenizer y el modelo.  Prueba CausalLM primero; si falla, ConditionalGeneration."""
    print(f"Cargando tokenizer …", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Cargando modelo {model_id} (CPU, dtype=bfloat16) …", flush=True)
    kw = dict(device_map="cpu", dtype=torch.bfloat16,
              low_cpu_mem_usage=True)
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except Exception:
        from transformers import AutoModelForSeq2SeqLM
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kw)
        except Exception:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_id, **kw)
    model.eval()
    return tok, model


# ──────────────────────────────────────────────────────────────
# Detección de capas (agnóstico de arquitectura)
# ──────────────────────────────────────────────────────────────

def get_decoder_layers(model):
    """Devuelve la lista de capas decoder sea cual sea la arquitectura."""
    import torch.nn as nn

    def _first_modulelist(root):
        """Busca el primer ModuleList con >1 elementos en los hijos directos."""
        for _, child in root.named_children():
            if isinstance(child, nn.ModuleList) and len(child) > 1:
                return child
        return None

    # Gemma 4: model.model.language_model.layers  (Gemma4ForConditionalGeneration)
    if hasattr(model, "model") and hasattr(model.model, "language_model") \
            and hasattr(model.model.language_model, "layers"):
        return model.model.language_model.layers

    # Gemma 4 alternativo: model.language_model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model") \
            and hasattr(model.language_model.model, "layers"):
        return model.language_model.model.layers

    # Gemma 1/2, Llama, Mistral, Phi: model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # GPT-2: transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    # Bloom / otros con transformer.*
    if hasattr(model, "transformer"):
        for attr in ("blocks", "layers", "h"):
            if hasattr(model.transformer, attr):
                return getattr(model.transformer, attr)

    raise ValueError(
        f"No se encontro la lista de capas en {type(model).__name__}. "
        "Anade el path en get_decoder_layers()."
    )


def layer_output_to_hidden(output):
    """Extrae el tensor de hidden states del output de una capa decoder."""
    if isinstance(output, tuple):
        return output[0]
    # Algunos modelos devuelven objects con .last_hidden_state
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    return output  # tensor directo


# ──────────────────────────────────────────────────────────────
# Hooks de extracción e inyección
# ──────────────────────────────────────────────────────────────

@contextmanager
def capture_hidden(layer, position=-1):
    """Hook que captura el hidden state de `position` en la capa `layer`."""
    captured = [None]

    def hook(module, inp, output):
        h = layer_output_to_hidden(output)
        captured[0] = h[0, position, :].detach().clone()  # [hidden]

    handle = layer.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def inject_hidden(layer, vec: torch.Tensor, scale: float, position=-1):
    """Hook que suma `scale * vec` en `position` del hidden state de `layer`."""
    v = vec.to(dtype=torch.bfloat16)

    def hook(module, inp, output):
        h = layer_output_to_hidden(output)
        h = h.clone()
        h[0, position, :] = h[0, position, :] + scale * v
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


# ──────────────────────────────────────────────────────────────
# Preparación de prompts y tokens
# ──────────────────────────────────────────────────────────────

def build_prompt(pairs, query):
    """Arma: 'PREFIX x SEP y\n...\nPREFIX query SEP'  (pares vacío = zero-shot)."""
    s = "".join(f"{_PROMPT_PREFIX}{x}{_PROMPT_SEP} {y}\n" for x, y in pairs)
    return s + f"{_PROMPT_PREFIX}{query}{_PROMPT_SEP}"


def _encode_dispatch(tokenizer, pairs, query) -> torch.Tensor:
    """Tokeniza ICL: en modo chat-alternado usa turnos user/assistant; si no, texto plano."""
    if _USE_CHAT_ALT:
        msgs = []
        for x, y in pairs:
            msgs.append({"role": "user",      "content": f"{_PROMPT_PREFIX}{x}{_PROMPT_SEP}"})
            msgs.append({"role": "assistant", "content": y})
        msgs.append({"role": "user", "content": f"{_PROMPT_PREFIX}{query}{_PROMPT_SEP}"})
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt").input_ids
    return tokenizer(build_prompt(pairs, query), return_tensors="pt").input_ids


def first_token_id(tokenizer, answer: str) -> int:
    """Id del primer token de la respuesta.

    Chat-alternado: el modelo arranca el turno sin espacio inicial (nuevo turno de asistente).
    Texto plano: añadimos espacio para coincidir con el tokenizador SPM.
    """
    prefix = "" if _USE_CHAT_ALT else " "
    ids = tokenizer.encode(prefix + answer, add_special_tokens=False)
    return ids[0]


def encode(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, return_tensors="pt").input_ids


# ──────────────────────────────────────────────────────────────
# Extracción del vector-tarea
# ──────────────────────────────────────────────────────────────

def extract_task_vector(model, tokenizer, layers, layer_idx, demos, n_prompts, k_shot, rng):
    """Vector-tarea = media del hidden state del último token en `layer_idx`."""
    layer = layers[layer_idx]
    total = None
    count = 0
    for _ in range(n_prompts):
        sample = rng.sample(demos, k_shot + 1)
        pairs, (q, _) = sample[:k_shot], sample[k_shot]
        ids = _encode_dispatch(tokenizer, pairs, q)
        with torch.no_grad(), capture_hidden(layer) as cap:
            model(ids)
        if cap[0] is not None:
            total = cap[0] if total is None else total + cap[0]
            count += 1
    if count == 0:
        raise RuntimeError(f"No se capturó ninguna activación en la capa {layer_idx}.")
    return total / float(count)


# ──────────────────────────────────────────────────────────────
# Evaluación de exactitud (first-token)
# ──────────────────────────────────────────────────────────────

def eval_accuracy(model, tokenizer, test, *, layer=None, vec=None, scale=0.0,
                  k_shot=0, demos=None, rng=None):
    """Exactitud por primer token. k_shot>0 => few-shot (cota superior)."""
    hits = 0
    for x, y in test:
        if k_shot and demos:
            pool = [d for d in demos if d[0] != x]
            pairs = rng.sample(pool, min(k_shot, len(pool)))
        else:
            pairs = []
        ids = _encode_dispatch(tokenizer, pairs, x)
        target = first_token_id(tokenizer, y)

        ctx = inject_hidden(layer, vec, scale) if (vec is not None and layer is not None) \
            else _null_ctx()
        with torch.no_grad(), ctx:
            out = model(ids)
        logits = out.logits[0, -1, :] if hasattr(out, "logits") else out[0][0, -1, :]
        pred = int(logits.argmax().item())
        if pred == target:
            hits += 1
    return hits / float(len(test))


@contextmanager
def _null_ctx():
    yield


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="F0 — viabilidad de la proteína (task vector).")
    ap.add_argument("--model", default="google/gemma-2b-it",
                    help="HF model id. Usa --smoke para gpt2 rápido.")
    ap.add_argument("--task",  default="antonym", choices=list(TASKS.keys()))
    ap.add_argument("--layers", default="",
                    help="Capas coma-separadas. Vacío = barrido automático franja media.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Lambda: multiplicador del vector inyectado.")
    ap.add_argument("--k-shot",  type=int, default=5,
                    help="Demostraciones por prompt ICL.")
    ap.add_argument("--n-extract", type=int, default=20,
                    help="Prompts para promediar el vector.")
    ap.add_argument("--n-test",    type=int, default=15,
                    help="Consultas de test.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="Modo smoke: gpt2 + 3 prompts + 3 test. Verifica la tubería en segundos.")
    args = ap.parse_args()

    if args.smoke:
        args.model    = "gpt2"
        args.n_extract = 3
        args.n_test   = 3
        args.k_shot   = 2
        print("-- SMOKE TEST (gpt2, mini-settings) --")

    # Aplicar la plantilla de la tarea (prefix + sep) a las globales de prompt.
    global _PROMPT_PREFIX, _PROMPT_SEP, _USE_CHAT_ALT
    tmpl = TASK_TEMPLATES.get(args.task, {})
    _PROMPT_PREFIX = tmpl.get("prefix", "")
    _PROMPT_SEP    = tmpl.get("sep", " ->")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    data = list(TASKS[args.task])
    rng.shuffle(data)
    test  = data[:args.n_test]
    demos = data[args.n_test:]
    if len(demos) <= args.k_shot:
        sys.exit(f"Pocos datos: baja --n-test o --k-shot "
                 f"(demos disponibles: {len(demos)}, k-shot: {args.k_shot}).")

    tokenizer, model = load_model(args.model)

    # Auto-detectar si el modelo usa chat template (modelo IT).
    _USE_CHAT_ALT = bool(getattr(tokenizer, "chat_template", None))
    if _USE_CHAT_ALT:
        print("  [modo chat-alternado] IT model detectado — usando turnos user/assistant")
        if args.scale == 1.0:  # si el usuario no cambió el default
            args.scale = 4.0
            print(f"  [modo chat-alternado] Escala ajustada a {args.scale} (óptimo para IT)")

    layers = get_decoder_layers(model)
    n_layers = len(layers)

    if args.layers:
        layer_idxs = [int(x) for x in args.layers.split(",")]
    else:
        lo, hi = n_layers // 4, (3 * n_layers) // 4
        # Paso más fino (//8) para no saltarse la capa óptima en modelos IT
        step = max(1, (hi - lo) // 8)
        layer_idxs = list(range(lo, hi + 1, step))

    print(f"\nModelo: {args.model} | capas totales: {n_layers} | "
          f"tarea: {args.task} | semilla: {args.seed}")
    print(f"Capas a barrer: {layer_idxs}")

    # ── Referencias ──────────────────────────────────────────────
    print("\nMidiendo referencias …", flush=True)
    base = eval_accuracy(model, tokenizer, test)
    few  = eval_accuracy(model, tokenizer, test,
                         k_shot=args.k_shot, demos=demos, rng=random.Random(args.seed))

    print(f"  zero-shot SIN vector : {base:.2f}")
    print(f"  few-shot ({args.k_shot} ejemplos): {few:.2f}   <- cota superior")

    # ── Barrido de capas ─────────────────────────────────────────
    print(f"\n  {'capa':>4} | {'zero+vector':>11} | {'ganancia':>10}")
    print("  " + "-" * 32)
    best_acc, best_layer = -1.0, layer_idxs[0]
    for L in layer_idxs:
        print(f"  extrayendo capa {L} …", end=" ", flush=True)
        vec = extract_task_vector(model, tokenizer, layers, L, demos,
                                  args.n_extract, args.k_shot, random.Random(args.seed))
        steered = eval_accuracy(model, tokenizer, test,
                                layer=layers[L], vec=vec, scale=args.scale)
        gain = steered - base
        marker = " << mejor" if steered > best_acc else ""
        print(f"\r  {L:>4} | {steered:>11.2f} | {gain:>+10.2f}{marker}   ")
        if steered > best_acc:
            best_acc, best_layer = steered, L

    # -- Resultado y gate -----------------------------------------
    print("\n== RESULTADO ==========================================")
    print(f"  Mejor capa    : {best_layer}")
    print(f"  zero-shot base: {base:.2f}")
    print(f"  few-shot (ref): {few:.2f}")
    print(f"  zero+vector   : {best_acc:.2f}")

    passed = best_acc >= max(0.50, base + 0.20)
    if passed:
        print("\n  GATE F0: PASA [OK]")
        print("  La proteina existe a esta escala -- la linea 'function vector' es viable.")
    else:
        print("\n  GATE F0: NO PASA")
        print("  Opciones: subir escala (--scale), barrer más capas, usar modelo más grande,")
        print("  o pasar al v1 (Todd & Bau, por cabezas con mediación causal).")
        print("  Honesto: si Gemma 4 (PLE) no captura señal, ajustar get_decoder_layers().")

    print("\n  Recordatorio: v0 = media de activaciones.  v1 = función vector por cabezas.")
    print("  Repetir con google/gemma-4-E2B-it para validar sobre arquitectura PLE.")


if __name__ == "__main__":
    main()
