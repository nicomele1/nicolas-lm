import torch
import torch.nn.functional as F
from nicolasm.models.transformer import TinyTransformerLanguageModel
from nicolasm.tokenizer import CharTokenizer

ckpt = torch.load('experiments/runs/effective_tokens/medium/transformer/model.pt', map_location='cpu')
tok = CharTokenizer(stoi=ckpt['stoi'], itos=ckpt['itos'])
model = TinyTransformerLanguageModel(
    vocab_size=ckpt['vocab_size'], block_size=ckpt['block_size'],
    embedding_dim=ckpt['embedding_dim'], num_heads=ckpt['num_heads'],
    num_layers=ckpt['num_layers'], dropout=0.0)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

idx = torch.tensor([tok.encode('hello')], dtype=torch.long)
with torch.no_grad():
    for _ in range(300):
        logits, _ = model(idx[:, -ckpt['block_size']:])
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
print(tok.decode(idx[0].tolist()))
