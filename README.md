## NicolásLM-0: Bigram Language Model

The first model in the project is a character-level bigram language model. Given a token \(x_t\), it learns a conditional distribution

\[
p_\theta(x_{t+1}\mid x_t).
\]

It is implemented as a learnable matrix of logits \(A \in \mathbb{R}^{|V|\times |V|}\), where each row defines a next-token distribution through a softmax.

The model can be trained with:

```bash
PYTHONPATH=src python scripts/train.py