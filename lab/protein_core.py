"""protein_core — maquinaria compartida del laboratorio de proteínas (function/task vectors).

Una sola responsabilidad: cargar modelo, leer/inyectar activaciones, extraer un vector-tarea
y predecir/generar con o sin él. Reutiliza los hallazgos verificados en F0:
  - Gemma 4 (PLE) expone capas en `model.model.language_model.layers`.
  - Modelos IT necesitan **chat-alternado** (un turno user/assistant por demo).
Probado: torch 2.7 · transformers 5.x · Gemma 2B-IT y Gemma 4 E2B-IT (CPU, bfloat16).
"""
from contextlib import contextmanager
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_id, force_chat=None):
    """Carga tokenizer + modelo causal (CPU, bfloat16). Devuelve (tok, model, use_chat).

    force_chat: True/False para sobrescribir la auto-detección. None = auto.
    Auto-detección activa chat-alternado solo en modelos con fine-tuning fuerte
    (Gemma 4+). Gemma 2B-IT y modelos base funcionan mejor con texto plano.
    """
    print(f"Cargando {model_id} … (puede tardar)", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    if force_chat is not None:
        use_chat = force_chat
    else:
        model_lower = model_id.lower()
        strong_it = any(k in model_lower for k in ("gemma-4", "gemma4", "llama-3", "llama3"))
        use_chat = strong_it and getattr(tok, "chat_template", None) is not None
    return tok, model, use_chat


def get_decoder_layers(model):
    """Lista de capas decoder, agnóstico de arquitectura (incluye Gemma 4 PLE)."""
    m = model
    if hasattr(m, "model") and hasattr(m.model, "language_model") \
            and hasattr(m.model.language_model, "layers"):
        return m.model.language_model.layers           # Gemma 4
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers                          # Gemma 1/2, Llama, Mistral
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h                         # GPT-2
    raise ValueError(f"No encuentro las capas en {type(m).__name__}; añade el path.")


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


@contextmanager
def capture(layer, pos=-1):
    """Captura el hidden state de la posición `pos` en `layer`."""
    box = [None]

    def hook(_m, _i, output):
        box[0] = _hidden(output)[0, pos, :].detach().clone()
    h = layer.register_forward_hook(hook)
    try:
        yield box
    finally:
        h.remove()


@contextmanager
def inject(layer, vec, scale, pos=-1, once=True):
    """Suma `scale*vec` en la posición `pos` de `layer`. once=True: solo el primer forward
    (= el token de la consulta); once=False: cada paso (steering fuerte)."""
    v = vec.to(dtype=torch.bfloat16)
    state = {"fired": False}

    def hook(_m, _i, output):
        if once and state["fired"]:
            return output
        h = _hidden(output).clone()
        h[0, pos, :] = h[0, pos, :] + scale * v
        state["fired"] = True
        return (h,) + output[1:] if isinstance(output, tuple) else h
    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def encode(tok, pairs, query, prefix, sep, use_chat):
    """Tokeniza ICL. Chat-alternado para IT; texto plano si no. `pairs` vacío = zero-shot."""
    if use_chat:
        msgs = []
        for x, y in pairs:
            msgs.append({"role": "user", "content": f"{prefix}{x}{sep}"})
            msgs.append({"role": "assistant", "content": y})
        msgs.append({"role": "user", "content": f"{prefix}{query}{sep}"})
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt").input_ids
    text = "".join(f"{prefix}{x}{sep} {y}\n" for x, y in pairs) + f"{prefix}{query}{sep}"
    return tok(text, return_tensors="pt").input_ids


def first_token(tok, answer, use_chat):
    """Id del primer token esperado (sin espacio en chat; con espacio en texto plano)."""
    ids = tok.encode(("" if use_chat else " ") + answer, add_special_tokens=False)
    return ids[0]


def extract_vector(model, layers, L, demos, n_prompts, k_shot, prefix, sep, use_chat, tok, rng):
    """Vector-tarea = media del hidden del último token (capa L) sobre n_prompts ICL."""
    acc = None
    for _ in range(n_prompts):
        s = rng.sample(demos, k_shot + 1)
        pairs, (q, _) = s[:k_shot], s[k_shot]
        ids = encode(tok, pairs, q, prefix, sep, use_chat)
        with capture(layers[L]) as box, torch.no_grad():
            model(ids)
        acc = box[0] if acc is None else acc + box[0]
    return acc / float(n_prompts)


def predict_first(model, layers, L, ids, vec=None, scale=0.0):
    """Primer token predicho (greedy). Si vec!=None, lo inyecta en la capa L."""
    with torch.no_grad():
        if vec is not None:
            with inject(layers[L], vec, scale, once=False):
                logits = model(ids).logits[0, -1, :]
        else:
            logits = model(ids).logits[0, -1, :]
    return int(logits.argmax().item())


def generate(model, layers, L, ids, tok, vec=None, scale=0.0, max_new=12):
    """Genera texto (greedy). Si vec!=None, inyecta una vez en la consulta y deja generar."""
    kw = dict(max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
    with torch.no_grad():
        if vec is not None:
            with inject(layers[L], vec, scale, once=True):
                out = model.generate(ids, **kw)
        else:
            out = model.generate(ids, **kw)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def default_layer(n_layers):
    """Capa por defecto ≈ 57% de profundidad (coincide con F0: 20/35 y 10–11/18)."""
    return round(0.57 * n_layers)
