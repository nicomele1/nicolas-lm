#!/usr/bin/env python3
"""
Web chat interface for nicolas-lm character-level language models.
Run with:  PYTHONPATH=src .venv/bin/python scripts/app.py
Then open: http://localhost:5000
"""

from __future__ import annotations

import json
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
    "LLaMA — Medium (Austen)":      "experiments/runs/effective_tokens/medium/llama/model.pt",
    "LLaMA — High (10 autores)":    "experiments/runs/effective_tokens/high/llama/model.pt",
    "Transformer — Medium (Austen)": "experiments/runs/effective_tokens/medium/transformer/model.pt",
    "Transformer — High (10 autores)": "experiments/runs/effective_tokens/high/transformer/model.pt",
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


HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>nicolas-lm chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #212121; color: #ececec; height: 100vh; display: flex; flex-direction: column; }

  /* top bar */
  #topbar { display: flex; align-items: center; gap: 12px; padding: 12px 20px;
            background: #2f2f2f; border-bottom: 1px solid #3a3a3a; flex-shrink: 0; }
  #topbar h1 { font-size: 15px; font-weight: 600; color: #ececec; flex: 1; }
  #model-select { background: #3a3a3a; color: #ececec; border: 1px solid #555;
                  border-radius: 8px; padding: 6px 10px; font-size: 13px; cursor: pointer; }
  .slider-group { display: flex; align-items: center; gap: 6px; font-size: 12px; color: #aaa; }
  .slider-group input[type=range] { width: 80px; accent-color: #19c37d; }
  .slider-group span { min-width: 28px; color: #ececec; }
  #clear-btn { background: none; border: 1px solid #555; color: #aaa; border-radius: 8px;
               padding: 5px 12px; font-size: 12px; cursor: pointer; }
  #clear-btn:hover { border-color: #aaa; color: #ececec; }

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
  #note { text-align: center; font-size: 11px; color: #666; margin-top: 8px; max-width: 820px; margin: 6px auto 0; }
</style>
</head>
<body>
<div id="topbar">
  <h1>nicolas-lm</h1>
  <select id="model-select">
    {% for name in models %}<option value="{{ name }}">{{ name }}</option>{% endfor %}
  </select>
  <div class="slider-group">
    🌡 Temp
    <input type="range" id="temp" min="0.3" max="1.5" step="0.05" value="0.8">
    <span id="temp-val">0.8</span>
  </div>
  <div class="slider-group">
    Top-k
    <input type="range" id="topk" min="0" max="80" step="5" value="20">
    <span id="topk-val">20</span>
  </div>
  <div class="slider-group">
    Tokens
    <input type="range" id="tokens" min="50" max="600" step="50" value="200">
    <span id="tokens-val">200</span>
  </div>
  <button id="clear-btn">Limpiar</button>
</div>

<div id="messages"></div>

<div id="input-area">
  <div id="form">
    <textarea id="prompt" rows="1" placeholder="Escribe algo y el modelo lo continúa…"></textarea>
    <button id="send-btn">&#9650;</button>
  </div>
  <div id="note">Estos modelos completan texto en estilo literario inglés — no responden preguntas.</div>
</div>

<script>
const msgs = document.getElementById('messages');
const promptEl = document.getElementById('prompt');
const sendBtn = document.getElementById('send-btn');

// sliders
['temp','topk','tokens'].forEach(id => {
  const el = document.getElementById(id);
  const val = document.getElementById(id+'-val');
  val.textContent = el.value;
  el.addEventListener('input', () => val.textContent = el.value);
});

document.getElementById('clear-btn').addEventListener('click', () => {
  msgs.innerHTML = '';
});

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + (role === 'user' ? 'user' : 'assistant');
  div.innerHTML = `<div class="avatar ${role==='user'?'u':'a'}">${role==='user'?'N':'🤖'}</div>
                   <div class="bubble">${text}</div>`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
  return div;
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'msg assistant';
  div.id = 'typing-indicator';
  div.innerHTML = '<div class="avatar a">🤖</div><div class="bubble"><span class="typing">●●●</span></div>';
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

async function send() {
  const text = promptEl.value.trim();
  if (!text) return;
  promptEl.value = '';
  promptEl.style.height = 'auto';
  sendBtn.disabled = true;

  addMsg('user', text);
  addTyping();

  const resp = await fetch('/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
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
  sendBtn.disabled = false;
  promptEl.focus();
}

sendBtn.addEventListener('click', send);
promptEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
promptEl.addEventListener('input', () => {
  promptEl.style.height = 'auto';
  promptEl.style.height = Math.min(promptEl.scrollHeight, 160) + 'px';
});
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
        # filter prompt to known chars
        known = set(tok.itos.values())
        prompt_clean = "".join(c for c in prompt if c in known) or " "
        full = generate(model, tok, block_size, prompt_clean,
                        max_new_tokens, temperature, top_k)
        generated = full[len(prompt_clean):]
        return jsonify({"generated": generated})
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
