"""Build the processed MEDDOCAN windows and write them to the data volume.

Run from the repo root:

    poetry run python scripts/prepare_data.py

Output (one pickle per split) goes to ``config.DATASET_DIR`` — the Tensorleap
data volume. Each pickle is ``{"sample_ids": [...], "samples": {sid: {...}}}``.
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()  # silence the >512 subword length notice

from ner_roberta.config import BASE_MODEL, DATASET_DIR, LABELS  # noqa: E402
from ner_roberta.data import SPLIT_MAP, build_split  # noqa: E402


def main() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f"labels ({len(LABELS)}): {LABELS}")

    for split, key in SPLIT_MAP.items():
        samples = build_split(split, tokenizer=tokenizer)
        sample_ids = sorted(samples.keys())
        out = os.path.join(DATASET_DIR, f"{key}.pkl")
        with open(out, "wb") as f:
            pickle.dump({"sample_ids": sample_ids, "samples": samples}, f)

        n_docs = len({samples[s]["doc_id"] for s in sample_ids})
        n_ent = sum(int((samples[s]["word_start_mask"] *
                         (samples[s]["labels"] > 0)).sum() > 0)  # windows w/ entity
                    for s in sample_ids)
        print(f"{key:11s} -> {len(sample_ids):5d} windows "
              f"from {n_docs:3d} docs ({n_ent} windows contain an entity) "
              f"-> {out}")


if __name__ == "__main__":
    main()
