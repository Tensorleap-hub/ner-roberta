"""Build the processed MEDDOCAN windows and write them to the data volume.

Run from the repo root:

    poetry run python scripts/prepare_data.py

Output (one pickle per split) goes to ``config.DATASET_DIR`` — the Tensorleap
data volume. Each pickle is ``{"sample_ids": [...], "samples": {sid: {...}}}``.

This is a thin CLI wrapper; the actual build logic lives in
``ner_roberta.data.prepare_dataset`` (also used as the on-demand fallback in
``leap_integration.preprocess``).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()  # silence the >512 subword length notice

from ner_roberta.config import DATASET_DIR, LABELS  # noqa: E402
from ner_roberta.data import prepare_dataset  # noqa: E402


def main() -> None:
    print(f"labels ({len(LABELS)}): {LABELS}")
    prepare_dataset(DATASET_DIR, verbose=True)


if __name__ == "__main__":
    main()
