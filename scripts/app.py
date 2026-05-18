#!/usr/bin/env python3
"""
Web chat interface for nicolas-lm character-level language models.
Run with:  PYTHONPATH=src .venv/bin/python scripts/app.py
Then open: http://localhost:5000
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template_string, request

from nicolasm.models.bigram import BigramLanguageModel
from nicolasm.models.llama import LlamaStyleLanguageModel
from nicolasm.models.transformer import TinyTransformerLanguageModel
from nicolasm.tokenizer import CharTokenizer

app = Flask(__name__)

CHECKPOINTS = {
    "LLaMA — Medium (Austen)":        "experiments/runs/effective_tokens/medium/llama/model.pt",
    "LLaMA — High (10 autores)":      "experiments/runs/effective_tokens/high/llama/model.pt",
    "Transformer — Medium (Austen)":  "experiments/runs/effective_tokens/medium/transformer/model.pt",
    "Transformer — High (10 autores)":"experiments/runs/effective_tokens/high/transformer/model.pt",
}

_cache: dict = {}


def load_model(key: str):
    if key in _cache:
        return _cache[key]
    path = Path(CHECKPOINTS[key])
    ckpt = torch.load(path, map_location="cpu")
    tok = CharTokenizer(stoi=ckpt["stoi"], itos=ckpt["itos"])
    name = ckpt.get("model_name", "transformer")
    if name == "llama":
        model = LlamaStyleLanguageModel(
            vocab_size=ckpt["vocab_size"], block_size=ckpt["block_size"],
            embedding_dim=ckpt["embedding_dim"], num_heads=ckpt["num_heads"],
            num_layers=ckpt["num_layers"], dropout=0.0)
    elif name == "transformer":
        model = TinyTransformerLanguageModel(
            vocab_size=ckpt["vocab_size"], block_size=ckpt["block_size"],
            embedding_dim=ckpt["embedding_dim"], num_heads=ckpt["num_heads"],
            num_layers=ckpt["num_layers"], dropout=0.0)
    else:
        model = BigramLanguageModel(vocab_size=ckpt["vocab_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    _cache[key] = (model, tok, ckpt["block_size"])
    return _cache[key]


@torch.no_grad()
def generate(model, tok, block_size, prompt, max_new_tokens, temperature, top_k):
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            vals, _ = torch.topk(logits, k)
            logits[logits < vals[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_id], dim=1)
    return tok.decode(idx[0].tolist())


def model_weight_info(key: str) -> dict:
    model, tok, block_size = load_model(key)
    layers = []
    total_params = 0
    for name, param in model.named_parameters():
        n = param.numel()
        total_params += n
        shape = list(param.shape)
        # human description
        if len(shape) == 1:
            desc = f"{n} escalares en ℝ^{shape[0]}"
        elif len(shape) == 2:
            desc = f"{shape[0]} vectores en ℝ^{shape[1]}  ({shape[0]}×{shape[1]})"
        else:
            desc = "×".join(str(s) for s in shape)
        # first few values
        flat = param.detach().flatten()
        sample = [round(x, 5) for x in flat[:8].tolist()]
        layers.append({
            "name": name,
            "shape": shape,
            "numel": n,
            "desc": desc,
            "sample": sample,
            "has_more": flat.numel() > 8,
        })
    return {
        "model_key": key,
        "total_params": total_params,
        "vocab_size": tok.vocab_size,
        "block_size": block_size,
        "num_layers": len(layers),
        "layers": layers,
    }


HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nicolas-lm</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #212121; color: #ececec; height: 100vh; display: flex; flex-direction: column; }

  /* top bar */
  #topbar { display: flex; align-items: center; gap: 12px; padding: 12px 20px;
            background: #2f2f2f; border-bottom: 1px solid #3a3a3a; flex-shrink: 0; flex-wrap: wrap; }
  #topbar h1 { font-size: 15px; font-weight: 600; color: #ececec; flex: 1; min-width: 100px; }
  #model-select { background: #3a3a3a; color: #ececec; border: 1px solid #555;
                  border-radius: 8px; padding: 6px 10px; font-size: 13px; cursor: pointer; }
  .slider-group { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #aaa; }
  .slider-group input[type=range] { width: 80px; accent-color: #19c37d; }
  .slider-group .val { min-width: 28px; color: #ececec; }
  .top-btn { background: none; border: 1px solid #555; color: #aaa; border-radius: 8px;
             padding: 5px 12px; font-size: 12px; cursor: pointer; white-space: nowrap; }
  .top-btn:hover { border-color: #aaa; color: #ececec; }
  #weights-btn { border-color: #7c6fcd; color: #7c6fcd; }
  #weights-btn:hover { border-color: #a89ee8; color: #a89ee8; background: #7c6fcd11; }
  #lang-btn { border-color: #19c37d55; color: #19c37d; font-weight: 600; letter-spacing: .03em; }
  #lang-btn:hover { border-color: #19c37d; background: #19c37d11; }

  /* messages */
  #messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 20px; }
  .msg { display: flex; gap: 12px; max-width: 820px; margin: 0 auto; width: 100%; }
  .msg.user { flex-direction: row-reverse; }
  .avatar { width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 700; }
  .avatar.u { background: #19c37d; color: #fff; }
  .avatar.a { background: #555; color: #ececec; }
  .bubble { background: #2f2f2f; border-radius: 14px; padding: 12px 16px;
            font-size: 14px; line-height: 1.6; white-space: pre-wrap; max-width: 720px; }
  .msg.user .bubble { background: #19c37d22; border: 1px solid #19c37d55; }
  .typing { color: #19c37d; font-size: 20px; animation: blink 1s infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

  /* input area */
  #input-area { padding: 16px 20px; background: #212121; border-top: 1px solid #3a3a3a; flex-shrink: 0; }
  #form { display: flex; gap: 10px; max-width: 820px; margin: 0 auto; }
  #prompt { flex: 1; background: #2f2f2f; border: 1px solid #555; border-radius: 12px;
            color: #ececec; padding: 12px 16px; font-size: 14px; resize: none;
            outline: none; font-family: inherit; min-height: 48px; max-height: 160px; }
  #prompt:focus { border-color: #19c37d; }
  #send-btn { background: #19c37d; color: #fff; border: none; border-radius: 10px;
              padding: 12px 18px; cursor: pointer; font-size: 18px; flex-shrink: 0; }
  #send-btn:disabled { background: #3a3a3a; cursor: not-allowed; }
  #note { text-align: center; font-size: 11px; color: #666; margin: 6px auto 0; max-width: 820px; }

  /* weights modal */
  #w-overlay { display: none; position: fixed; inset: 0; background: #000a;
               z-index: 100; align-items: flex-start; justify-content: center;
               padding: 40px 20px; overflow-y: auto; }
  #w-overlay.open { display: flex; }
  #w-panel { background: #1a1a2e; border: 1px solid #7c6fcd55; border-radius: 16px;
             width: 100%; max-width: 760px; padding: 28px; position: relative; }
  #w-close { position: absolute; top: 16px; right: 20px; background: none; border: none;
              color: #aaa; font-size: 22px; cursor: pointer; line-height: 1; }
  #w-close:hover { color: #fff; }
  #w-panel h2 { font-size: 16px; font-weight: 700; color: #a89ee8; margin-bottom: 18px; }

  /* stats grid */
  .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 22px; }
  @media (max-width: 540px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } }
  .stat-card { background: #252540; border-radius: 10px; padding: 12px 14px; text-align: center; }
  .stat-val { font-size: 17px; font-weight: 700; color: #a89ee8; }
  .stat-lbl { font-size: 11px; color: #888; margin-top: 3px; }

  /* group accordion */
  .w-group { margin-bottom: 5px; }
  .group-header { display: flex; align-items: center; gap: 9px; cursor: pointer;
                  padding: 10px 14px; background: #252540; border-radius: 10px;
                  user-select: none; transition: background .15s; }
  .group-header:hover { background: #2d2d55; }
  .group-arrow { font-size: 9px; color: #7c6fcd; display: inline-block; transition: transform .2s; }
  .group-header.open .group-arrow { transform: rotate(90deg); }
  .group-title { font-size: 13px; font-weight: 600; color: #ccc; flex: 1; }
  .group-meta { font-size: 11px; color: #666; }
  .group-body { display: none; padding: 5px 0 6px 0; }
  .group-body.open { display: block; }

  /* layer card */
  .layer-card { background: #1e1e38; border-left: 2px solid #7c6fcd44; border-radius: 0 8px 8px 0;
                padding: 11px 14px; margin: 3px 0 3px 14px; }
  .layer-name { font-family: 'SFMono-Regular', Consolas, monospace; font-size: 11.5px;
                color: #7c6fcd; margin-bottom: 3px; }
  .layer-desc { font-size: 12px; color: #bbb; margin-bottom: 7px; }
  .layer-toggle { background: none; border: 1px solid #444; border-radius: 5px;
                  color: #888; font-size: 11px; padding: 3px 9px; cursor: pointer; }
  .layer-toggle:hover { border-color: #7c6fcd; color: #a89ee8; }
  .layer-values { display: none; margin-top: 8px; font-family: 'SFMono-Regular', Consolas, monospace;
                  font-size: 11px; color: #19c37d; background: #111128;
                  border-radius: 5px; padding: 9px 12px; line-height: 1.8; word-break: break-all; }
  .layer-values.open { display: block; }
  #w-loading { text-align: center; color: #7c6fcd; padding: 40px; font-size: 14px; }
</style>
</head>
<body>
<div id="topbar">
  <h1>nicolas-lm</h1>
  <select id="model-select">
    {% for name in models %}<option value="{{ name }}">{{ name }}</option>{% endfor %}
  </select>
  <div class="slider-group"><span data-i18n="temp">🌡 Temp</span>
    <input type="range" id="temp" min="0.3" max="1.5" step="0.05" value="0.8">
    <span class="val" id="temp-val">0.8</span>
  </div>
  <div class="slider-group">Top-k
    <input type="range" id="topk" min="0" max="80" step="5" value="20">
    <span class="val" id="topk-val">20</span>
  </div>
  <div class="slider-group">Tokens
    <input type="range" id="tokens" min="50" max="600" step="50" value="200">
    <span class="val" id="tokens-val">200</span>
  </div>
  <button id="weights-btn" class="top-btn" data-i18n="weightsBtn">⚛ Ver pesos</button>
  <button id="clear-btn"   class="top-btn" data-i18n="clearBtn">Limpiar</button>
  <button id="lang-btn"    class="top-btn">EN</button>
</div>

<div id="messages"></div>

<div id="input-area">
  <div id="form">
    <textarea id="prompt" rows="1" placeholder="Escribe algo y el modelo lo continúa…"></textarea>
    <button id="send-btn">&#9650;</button>
  </div>
  <div id="note" data-i18n="note">Estos modelos completan texto en estilo literario inglés — no responden preguntas.</div>
</div>

<!-- weights modal -->
<div id="w-overlay">
  <div id="w-panel">
    <button id="w-close">×</button>
    <h2 id="w-title"></h2>
    <div id="w-stats"></div>
    <div id="w-layers"></div>
    <div id="w-loading" style="display:none"></div>
  </div>
</div>

<script>
// ── i18n ──────────────────────────────────────────────────────────────────────
const STRINGS = {
  es: {
    temp:       '🌡 Temp',
    weightsBtn: '⚛ Ver pesos',
    clearBtn:   'Limpiar',
    placeholder:'Escribe algo y el modelo lo continúa…',
    note:       'Estos modelos completan texto en estilo literario inglés — no responden preguntas.',
    modalTitle: k => 'Pesos de ' + k,
    loading:    'Cargando pesos…',
    totalParams:'parámetros totales',
    tensorsLbl: 'tensores',
    vocabLbl:   'caracteres (vocab)',
    contextLbl: 'tokens de contexto',
    groupEmb:   'Embedding',
    groupBlock: n => 'Bloque ' + n,
    groupNorm:  'Normalización final',
    groupHead:  'Cabeza de predicción',
    scalarDesc: (n, d) => n.toLocaleString() + ' escalares en ℝ^' + d,
    matrixDesc: (r, c) => r + ' vectores en ℝ^' + c + '  (' + r + '×' + c + ')',
    paramsOf:   n => n.toLocaleString() + ' parámetros',
    groupOf:    n => n + ' ' + (n === 1 ? 'tensor' : 'tensores'),
    showVals:   'Ver primeros valores',
    hideVals:   'Ocultar valores',
    langNext:   'EN',
  },
  en: {
    temp:       '🌡 Temp',
    weightsBtn: '⚛ View weights',
    clearBtn:   'Clear',
    placeholder:'Type something and the model continues it…',
    note:       'These models complete text in English literary style — they do not answer questions.',
    modalTitle: k => 'Weights of ' + k,
    loading:    'Loading weights…',
    totalParams:'total parameters',
    tensorsLbl: 'tensors',
    vocabLbl:   'chars (vocab)',
    contextLbl: 'context tokens',
    groupEmb:   'Embedding',
    groupBlock: n => 'Block ' + n,
    groupNorm:  'Final normalization',
    groupHead:  'Prediction head',
    scalarDesc: (n, d) => n.toLocaleString() + ' scalars in ℝ^' + d,
    matrixDesc: (r, c) => r + ' vectors in ℝ^' + c + '  (' + r + '×' + c + ')',
    paramsOf:   n => n.toLocaleString() + ' parameters',
    groupOf:    n => n + ' ' + (n === 1 ? 'tensor' : 'tensors'),
    showVals:   'Show first values',
    hideVals:   'Hide values',
    langNext:   'ES',
  }
};

let lang = 'es';
const T = () => STRINGS[lang];
let lastData = null;
let lastKey  = null;

function applyLang() {
  document.documentElement.lang = lang;
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const v = T()[el.dataset.i18n];
    if (typeof v === 'string') el.textContent = v;
  });
  document.getElementById('prompt').placeholder = T().placeholder;
  document.getElementById('lang-btn').textContent = T().langNext;
  if (lastData) renderWeights(lastData);
}

document.getElementById('lang-btn').addEventListener('click', () => {
  lang = lang === 'es' ? 'en' : 'es';
  applyLang();
});

applyLang();

// ── sliders ───────────────────────────────────────────────────────────────────
['temp', 'topk', 'tokens'].forEach(id => {
  const el  = document.getElementById(id);
  const val = document.getElementById(id + '-val');
  val.textContent = el.value;
  el.addEventListener('input', () => val.textContent = el.value);
});

document.getElementById('clear-btn').addEventListener('click',
  () => document.getElementById('messages').innerHTML = '');

// ── chat ──────────────────────────────────────────────────────────────────────
const msgs     = document.getElementById('messages');
const promptEl = document.getElementById('prompt');
const sendBtn  = document.getElementById('send-btn');

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
  div.innerHTML = '<div class="avatar ' + (role==='user'?'u':'a') + '">'
                + (role==='user' ? 'N' : '🤖') + '</div>'
                + '<div class="bubble">' + text + '</div>';
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'msg assistant'; div.id = 'typing-indicator';
  div.innerHTML = '<div class="avatar a">🤖</div><div class="bubble"><span class="typing">●●●</span></div>';
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

async function send() {
  const text = promptEl.value.trim();
  if (!text) return;
  promptEl.value = ''; promptEl.style.height = 'auto';
  sendBtn.disabled = true;
  addMsg('user', text);
  addTyping();
  const resp = await fetch('/generate', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      prompt: text,
      model: document.getElementById('model-select').value,
      temperature: parseFloat(document.getElementById('temp').value),
      top_k: parseInt(document.getElementById('topk').value),
      max_new_tokens: parseInt(document.getElementById('tokens').value),
    })
  });
  const data = await resp.json();
  document.getElementById('typing-indicator')?.remove();
  addMsg('assistant', data.error || data.generated);
  sendBtn.disabled = false; promptEl.focus();
}

sendBtn.addEventListener('click', send);
promptEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
promptEl.addEventListener('input', () => {
  promptEl.style.height = 'auto';
  promptEl.style.height = Math.min(promptEl.scrollHeight, 160) + 'px';
});

// ── weights panel ──────────────────────────────────────────────────────────────
const overlay  = document.getElementById('w-overlay');
const wTitle   = document.getElementById('w-title');
const wStats   = document.getElementById('w-stats');
const wLayers  = document.getElementById('w-layers');
const wLoading = document.getElementById('w-loading');
document.getElementById('weights-btn').addEventListener('click', openWeights);
document.getElementById('w-close').addEventListener('click',
  () => overlay.classList.remove('open'));
overlay.addEventListener('click',
  e => { if (e.target === overlay) overlay.classList.remove('open'); });

function getGroup(name) {
  const parts = name.split('.');
  for (let i = 0; i < parts.length; i++)
    if (/^\\d+$/.test(parts[i])) return parts.slice(0, i + 1).join('.');
  return parts[0];
}

function groupLabel(g) {
  if (g === 'embedding') return T().groupEmb;
  if (g === 'norm')      return T().groupNorm;
  if (g === 'lm_head')   return T().groupHead;
  const m = g.match(/^blocks\\.(\\d+)$/);
  if (m) return T().groupBlock(m[1]);
  return g;
}

function layerDesc(shape, numel) {
  if (shape.length === 1) return T().scalarDesc(numel, shape[0]);
  if (shape.length === 2) return T().matrixDesc(shape[0], shape[1]);
  return shape.join('×');
}

function renderWeights(d) {
  wTitle.textContent = T().modalTitle(lastKey);
  wLoading.style.display = 'none';

  wStats.innerHTML =
    '<div class="stats-grid">'
    + '<div class="stat-card"><div class="stat-val">' + d.total_params.toLocaleString()
    + '</div><div class="stat-lbl" data-i18n="totalParams">' + T().totalParams + '</div></div>'
    + '<div class="stat-card"><div class="stat-val">' + d.num_layers
    + '</div><div class="stat-lbl" data-i18n="tensorsLbl">' + T().tensorsLbl + '</div></div>'
    + '<div class="stat-card"><div class="stat-val">' + d.vocab_size
    + '</div><div class="stat-lbl" data-i18n="vocabLbl">' + T().vocabLbl + '</div></div>'
    + '<div class="stat-card"><div class="stat-val">' + d.block_size
    + '</div><div class="stat-lbl" data-i18n="contextLbl">' + T().contextLbl + '</div></div>'
    + '</div>';

  // group by module prefix
  const groupMap = new Map();
  d.layers.forEach(layer => {
    const g = getGroup(layer.name);
    if (!groupMap.has(g)) groupMap.set(g, []);
    groupMap.get(g).push(layer);
  });

  wLayers.innerHTML = '';
  let first = true;
  groupMap.forEach((layers, g) => {
    const groupParams = layers.reduce((s, l) => s + l.numel, 0);

    const section = document.createElement('div');
    section.className = 'w-group';

    const header = document.createElement('div');
    header.className = 'group-header' + (first ? ' open' : '');
    header.innerHTML =
      '<span class="group-arrow">▶</span>'
      + '<span class="group-title">' + groupLabel(g) + '</span>'
      + '<span class="group-meta">' + T().groupOf(layers.length)
      + ' &nbsp;·&nbsp; ' + T().paramsOf(groupParams) + '</span>';

    const body = document.createElement('div');
    body.className = 'group-body' + (first ? ' open' : '');

    layers.forEach(layer => {
      const sampleStr = layer.sample.map(v => v.toFixed(5)).join(', ')
                      + (layer.has_more ? ', …' : '');
      const card = document.createElement('div');
      card.className = 'layer-card';
      card.innerHTML =
        '<div class="layer-name">' + layer.name + '</div>'
        + '<div class="layer-desc">' + layerDesc(layer.shape, layer.numel)
        + ' &nbsp;·&nbsp; ' + T().paramsOf(layer.numel) + '</div>'
        + '<button class="layer-toggle" onclick="toggleVals(this)">' + T().showVals + '</button>'
        + '<div class="layer-values">[' + sampleStr + ']</div>';
      body.appendChild(card);
    });

    header.addEventListener('click', () => {
      header.classList.toggle('open');
      body.classList.toggle('open');
    });

    section.appendChild(header);
    section.appendChild(body);
    wLayers.appendChild(section);
    first = false;
  });
}

async function openWeights() {
  overlay.classList.add('open');
  const key = document.getElementById('model-select').value;

  if (lastData && lastKey === key) { renderWeights(lastData); return; }

  wStats.innerHTML = ''; wLayers.innerHTML = '';
  wLoading.textContent = T().loading;
  wLoading.style.display = 'block';
  wTitle.textContent = T().modalTitle(key);
  lastKey = key;

  const resp = await fetch('/weights?model=' + encodeURIComponent(key));
  lastData = await resp.json();
  renderWeights(lastData);
}

function toggleVals(btn) {
  const vals = btn.nextElementSibling;
  const open = vals.classList.toggle('open');
  btn.textContent = open ? T().hideVals : T().showVals;
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML, models=list(CHECKPOINTS.keys()))


@app.route("/generate", methods=["POST"])
def generate_endpoint():
    data = request.get_json()
    key = data.get("model", list(CHECKPOINTS.keys())[0])
    prompt = data.get("prompt", "")
    temperature = float(data.get("temperature", 0.8))
    top_k = int(data.get("top_k", 20))
    max_new_tokens = int(data.get("max_new_tokens", 200))
    try:
        model, tok, block_size = load_model(key)
        known = set(tok.itos.values())
        prompt_clean = "".join(c for c in prompt if c in known) or " "
        full = generate(model, tok, block_size, prompt_clean,
                        max_new_tokens, temperature, top_k)
        return jsonify({"generated": full[len(prompt_clean):]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/weights")
def weights_endpoint():
    key = request.args.get("model", list(CHECKPOINTS.keys())[0])
    try:
        return jsonify(model_weight_info(key))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    print("Cargando modelos…")
    for k in CHECKPOINTS:
        try:
            load_model(k)
            print(f"  ✓ {k}")
        except Exception as e:
            print(f"  ✗ {k}: {e}")
    port = int(os.environ.get("PORT", 5000))
    print(f"\nAbre http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
