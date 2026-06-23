"""MEDDOCAN -> windowed, subword-tokenized BIO samples.

Pipeline per document:
  1. reconstruct full text from bigbio passages;
  2. map MEDDOCAN entity types to {PER, GPE, STREET} (drop the rest);
  3. word-tokenize with the custom punctuation tokenizer (keeps char offsets);
  4. assign BIO labels to words from the entity char spans;
  5. subword-tokenize words with the XLM-R tokenizer, aligning labels to subwords
     (continuation subwords get the I- label);
  6. slide a window of WINDOW_SIZE (incl. <s>/</s>) with WINDOW_STRIDE over the
     subword stream, padding the tail.

This module imports the heavy ML stack and is used only by prepare_data.py /
train.py — NOT by leap_integration.py.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import (
    BASE_MODEL,
    CONTENT_SIZE,
    DATASET_DIR,
    LABEL2ID,
    MEDDOCAN_TYPE_MAP,
    NO_WORD,
    O_ID,
    WINDOW_SIZE,
    WINDOW_STRIDE,
)
from .tokenization import Token, tokenize_with_offsets

BIGBIO_CONFIG = "meddocan_bigbio_kb"

# bigbio split name -> our state key
SPLIT_MAP = {"train": "training", "validation": "validation"}


# --------------------------------------------------------------------------- #
# Document loading
# --------------------------------------------------------------------------- #
def reconstruct_text(passages: List[dict]) -> str:
    """Rebuild the document text from bigbio passages, honoring char offsets."""
    spans: List[Tuple[int, int, str]] = []
    for p in passages:
        for text, (start, end) in zip(p["text"], p["offsets"]):
            spans.append((start, end, text))
    if not spans:
        return ""
    spans.sort()
    total = max(end for _, end, _ in spans)
    buf = [" "] * total
    for start, end, text in spans:
        buf[start:end] = list(text)
    return "".join(buf)


def load_documents(split: str) -> List[dict]:
    """Return ``[{document_id, text, entities:[(cls, start, end)]}]`` for a split."""
    ds = load_dataset("bigbio/meddocan", name=BIGBIO_CONFIG,
                      split=split, trust_remote_code=True)
    docs: List[dict] = []
    for ex in ds:
        text = reconstruct_text(ex["passages"])
        entities: List[Tuple[str, int, int]] = []
        for ent in ex["entities"]:
            cls = MEDDOCAN_TYPE_MAP.get(ent["type"])
            if cls is None:
                continue
            offs = ent["offsets"]
            if not offs:
                continue
            start = min(o[0] for o in offs)
            end = max(o[1] for o in offs)
            entities.append((cls, start, end))
        docs.append({
            "document_id": ex["document_id"],
            "text": text,
            "entities": entities,
        })
    return docs


# --------------------------------------------------------------------------- #
# Word-level BIO labelling
# --------------------------------------------------------------------------- #
def assign_bio(words: List[Token],
               entities: List[Tuple[str, int, int]]) -> List[str]:
    """BIO-label each word from entity char spans (first overlapping word = B-)."""
    labels = ["O"] * len(words)
    for cls, es, ee in entities:
        first = True
        for i, (_, ws, we) in enumerate(words):
            if ws < ee and we > es:  # char-span overlap
                if labels[i] != "O":  # don't overwrite an earlier entity
                    continue
                labels[i] = ("B-" if first else "I-") + cls
                first = False
    return labels


# --------------------------------------------------------------------------- #
# Subword tokenization + windowing
# --------------------------------------------------------------------------- #
def _continuation_label(word_label: str) -> str:
    """Label for a non-first subword of a word: I- of the same class, else O."""
    if word_label == "O":
        return "O"
    return "I-" + word_label[2:]


def encode_document(doc: dict, tokenizer) -> List[Tuple[int, int, int, str]]:
    """Flatten a document to ``[(subword_id, label_id, word_start, token_str)]``.

    ``word_start`` is 1 for the first subword of each word, else 0.
    """
    words_with_off = tokenize_with_offsets(doc["text"])
    if not words_with_off:
        return []
    words = [w for w, _, _ in words_with_off]
    word_labels = assign_bio(words_with_off, doc["entities"])

    enc = tokenizer(words, is_split_into_words=True, add_special_tokens=False)
    word_ids = enc.word_ids()
    input_ids = enc["input_ids"]

    flat: List[Tuple[int, int, int, str]] = []
    prev_w: Optional[int] = None
    for sub_id, w in zip(input_ids, word_ids):
        if w is None:
            continue
        is_start = w != prev_w
        prev_w = w
        wl = word_labels[w]
        label = wl if is_start else _continuation_label(wl)
        token_str = tokenizer.convert_ids_to_tokens(sub_id)
        flat.append((sub_id, LABEL2ID[label], 1 if is_start else 0, token_str))
    return flat


def make_windows(doc_id: str, split_key: str, flat, tokenizer) -> List[dict]:
    """Slide WINDOW_SIZE windows (incl. <s>/</s>) over the subword stream."""
    bos, eos, pad = (tokenizer.bos_token_id, tokenizer.eos_token_id,
                     tokenizer.pad_token_id)
    bos_tok = tokenizer.convert_ids_to_tokens(bos)
    eos_tok = tokenizer.convert_ids_to_tokens(eos)
    pad_tok = tokenizer.convert_ids_to_tokens(pad)

    n = len(flat)
    windows: List[dict] = []
    starts = list(range(0, max(1, n), WINDOW_STRIDE))
    win_idx = 0
    for start in starts:
        if start > 0 and start >= n:
            break
        chunk = flat[start:start + CONTENT_SIZE]

        ids = [bos] + [c[0] for c in chunk] + [eos]
        labels = [O_ID] + [c[1] for c in chunk] + [O_ID]
        starts_mask = [0] + [c[2] for c in chunk] + [0]
        word_idx = [NO_WORD] + list(range(start, start + len(chunk))) + [NO_WORD]
        toks = [bos_tok] + [c[3] for c in chunk] + [eos_tok]
        attn = [1] * len(ids)

        pad_n = WINDOW_SIZE - len(ids)
        ids += [pad] * pad_n
        labels += [O_ID] * pad_n
        starts_mask += [0] * pad_n
        word_idx += [NO_WORD] * pad_n
        toks += [pad_tok] * pad_n
        attn += [0] * pad_n

        windows.append({
            "sample_id": f"{split_key}-{doc_id}-w{win_idx}",
            "doc_id": doc_id,
            "window_index": win_idx,
            "input_ids": np.asarray(ids, dtype=np.int64),
            "attention_mask": np.asarray(attn, dtype=np.int64),
            "labels": np.asarray(labels, dtype=np.int64),
            "word_start_mask": np.asarray(starts_mask, dtype=np.int64),
            "word_ids": np.asarray(word_idx, dtype=np.int64),
            "tokens": toks,
        })
        win_idx += 1
    return windows


def build_split(split: str, tokenizer=None) -> Dict[str, dict]:
    """Build all windowed samples for a bigbio split. Returns {sample_id: sample}."""
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    split_key = SPLIT_MAP[split]
    samples: Dict[str, dict] = {}
    for doc in load_documents(split):
        flat = encode_document(doc, tokenizer)
        if not flat:
            continue
        for win in make_windows(doc["document_id"], split_key, flat, tokenizer):
            samples[win["sample_id"]] = win
    return samples


def prepare_dataset(out_dir: str = DATASET_DIR, verbose: bool = True) -> str:
    """Fetch MEDDOCAN, build windows, and write one pickle per split to out_dir.

    Reusable entry point for both scripts/prepare_data.py and the on-demand
    fallback in leap_integration.preprocess(). Each pickle is
    ``{"sample_ids": [...], "samples": {sid: {...}}}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    for split, key in SPLIT_MAP.items():
        samples = build_split(split, tokenizer=tokenizer)
        sample_ids = sorted(samples.keys())
        path = os.path.join(out_dir, f"{key}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"sample_ids": sample_ids, "samples": samples}, f)
        if verbose:
            print(f"{key:11s} -> {len(sample_ids):6d} windows -> {path}")
    return out_dir
