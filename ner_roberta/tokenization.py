"""Custom word tokenizer.

Rules (from the task spec):
  * whitespace is normalized — any run of whitespace separates tokens and is
    otherwise discarded;
  * every punctuation character is separated from words into its own token, e.g.
    ``"(hello.there!"`` -> ``["(", "hello", ".", "there", "!"]``.

We tokenize directly over the ORIGINAL text (rather than building a normalized
copy) so that each token keeps its character offsets into the source document.
Those offsets are what let us align MEDDOCAN entity spans (which are character
offsets) onto words for BIO labelling.
"""

import unicodedata
from typing import List, Tuple

Token = Tuple[str, int, int]  # (text, start_char, end_char)


def _is_punct(ch: str) -> bool:
    """True for any Unicode punctuation character (categories P*).

    Covers ASCII punctuation plus Spanish marks such as ``¿`` / ``¡`` and
    dashes/quotes, so each is split off into its own token.
    """
    return unicodedata.category(ch).startswith("P")


def tokenize_with_offsets(text: str) -> List[Token]:
    """Split ``text`` into ``(token, start, end)`` triples.

    A token is either a maximal run of non-whitespace, non-punctuation
    characters (a "word") or a single punctuation character. Whitespace is
    skipped, which implements whitespace normalization for tokenization.
    """
    tokens: List[Token] = []
    buf: List[str] = []
    word_start = -1

    def flush(end: int) -> None:
        nonlocal buf, word_start
        if buf:
            tokens.append(("".join(buf), word_start, end))
            buf = []
            word_start = -1

    for i, ch in enumerate(text):
        if ch.isspace():
            flush(i)
        elif _is_punct(ch):
            flush(i)
            tokens.append((ch, i, i + 1))
        else:
            if not buf:
                word_start = i
            buf.append(ch)
    flush(len(text))
    return tokens


def tokenize(text: str) -> List[str]:
    """Convenience: just the token strings."""
    return [t for t, _, _ in tokenize_with_offsets(text)]
