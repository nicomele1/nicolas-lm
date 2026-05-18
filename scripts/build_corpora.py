#!/usr/bin/env python3
"""
Descarga textos de Project Gutenberg y construye dos corpus:
  - corpus_medium.txt : un solo autor, diversidad moderada
  - corpus_high.txt   : varios autores/géneros, intercalados

Uso:
    PYTHONPATH=src .venv/bin/python scripts/build_corpora.py

Los corpus se guardan en data/corpora/ y se truncan a MAX_CHARS caracteres.
"""

import re
import sys
import urllib.request
from pathlib import Path

# ── Tamaño objetivo ───────────────────────────────────────────────────────────
MAX_CHARS = 1_000_000   # 1M caracteres por corpus

# ── Textos de Gutenberg ───────────────────────────────────────────────────────
# Cada entrada: (ID, descripción)
# Medium: un solo autor (Jane Austen, inglés limpio y homogéneo)
MEDIUM_BOOKS = [
    ("1342",  "Pride and Prejudice — Austen"),
    ("161",   "Sense and Sensibility — Austen"),
    ("105",   "Persuasion — Austen"),
    ("121",   "Northanger Abbey — Austen"),
    ("141",   "Mansfield Park — Austen"),
]

# High: autores y géneros distintos (novela, ciencia ficción, aventura, ensayo)
HIGH_BOOKS = [
    ("2701",  "Moby Dick — Melville"),
    ("84",    "Frankenstein — Shelley"),
    ("98",    "Tale of Two Cities — Dickens"),
    ("1260",  "Jane Eyre — Brontë"),
    ("76",    "Adventures of Huckleberry Finn — Twain"),
    ("174",   "Picture of Dorian Gray — Wilde"),
    ("35",    "The Time Machine — Wells"),
    ("1232",  "The Prince — Machiavelli"),
    ("1661",  "Sherlock Holmes — Doyle"),
    ("2554",  "Crime and Punishment — Dostoevsky"),
]

BASE_URL = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
FALLBACK  = "https://gutenberg.org/cache/epub/{id}/pg{id}.txt"

OUT_DIR = Path("data/corpora")


def fetch(book_id: str, desc: str) -> str:
    for tmpl in (BASE_URL, FALLBACK):
        url = tmpl.format(id=book_id)
        try:
            print(f"  Descargando {desc} … ", end="", flush=True)
            with urllib.request.urlopen(url, timeout=20) as r:
                raw = r.read().decode("utf-8", errors="replace")
            print(f"{len(raw):,} chars")
            return raw
        except Exception as e:
            print(f"fallo ({e}), probando alternativa…")
    raise RuntimeError(f"No se pudo descargar el libro {book_id}")


def strip_gutenberg_header_footer(text: str) -> str:
    """Elimina los encabezados y pies de página estándar de Gutenberg."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
    ]
    for m in start_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[text.find("\n", idx) + 1:]
            break
    for m in end_markers:
        idx = text.find(m)
        if idx != -1:
            text = text[:idx]
            break
    return text.strip()


def clean(text: str) -> str:
    text = strip_gutenberg_header_footer(text)
    # Normalizar saltos de línea
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Colapsar líneas en blanco excesivas
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def build_corpus(books: list[tuple[str, str]], max_chars: int) -> str:
    """Descarga y concatena libros hasta alcanzar max_chars."""
    parts = []
    total = 0
    for book_id, desc in books:
        if total >= max_chars * 1.1:   # margen para truncar limpio
            break
        text = clean(fetch(book_id, desc))
        parts.append(text)
        total += len(text)
        print(f"    → acumulado: {total:,} chars")
    combined = "\n\n".join(parts)
    return combined[:max_chars]


def interleave(books: list[tuple[str, str]], max_chars: int,
               chunk: int = 5_000) -> str:
    """
    Descarga todos los libros y los intercala en fragmentos de `chunk` chars.
    Produce mayor diversidad local que la concatenación simple.
    """
    texts = []
    for book_id, desc in books:
        texts.append(clean(fetch(book_id, desc)))

    # Fragmentar cada texto en chunks
    fragments: list[str] = []
    for text in texts:
        fragments += [text[i:i+chunk] for i in range(0, len(text), chunk)]

    # Intercalar round-robin entre libros (ya están fragmentados)
    from itertools import zip_longest
    book_chunks = [
        [text[i:i+chunk] for i in range(0, len(text), chunk)]
        for text in texts
    ]
    interleaved: list[str] = []
    for group in zip_longest(*book_chunks, fillvalue=""):
        interleaved.extend(g for g in group if g)

    combined = "\n".join(interleaved)
    return combined[:max_chars]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Corpus MEDIUM ({MAX_CHARS:,} chars) — un solo autor")
    print(f"{'='*60}")
    medium = build_corpus(MEDIUM_BOOKS, MAX_CHARS)
    out_m = OUT_DIR / "corpus_medium.txt"
    out_m.write_text(medium, encoding="utf-8")
    print(f"✓ Guardado: {out_m}  ({len(medium):,} chars)\n")

    print(f"{'='*60}")
    print(f"Corpus HIGH ({MAX_CHARS:,} chars) — múltiples autores, intercalado")
    print(f"{'='*60}")
    high = interleave(HIGH_BOOKS, MAX_CHARS, chunk=5_000)
    out_h = OUT_DIR / "corpus_high.txt"
    out_h.write_text(high, encoding="utf-8")
    print(f"✓ Guardado: {out_h}  ({len(high):,} chars)\n")

    print("Corpora listos. Para correr el experimento:")
    print(f"""
  PYTHONPATH=src .venv/bin/python scripts/run_effective_tokens.py \\
    --input {out_m} --name medium \\
    --input {out_h} --name high \\
    --model-name transformer \\
    --model-name llama \\
    --max-chars {MAX_CHARS} \\
    --max-steps 20000 \\
    --output experiments/results/effective_tokens_1M.csv
""")


if __name__ == "__main__":
    main()
