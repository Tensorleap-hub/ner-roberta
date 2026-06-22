"""Central, dependency-light configuration for the NER pipeline.

This module is imported by ``leap_integration.py`` at platform-parse time, so it
must stay import-light: standard library + (optionally) numpy only. Do NOT import
torch / transformers / datasets here.
"""

import os

# --- Paths -----------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)

# The processed dataset lives in the Tensorleap data volume. It is read at
# runtime from a CONFIG-DRIVEN path (a mount point on the platform), never a
# hardcoded one — override with NER_DATASET_DIR if the mount differs.
DATASET_DIR = os.environ.get(
    "NER_DATASET_DIR",
    "/Users/yotamazriel/tensorleap/data/ner-roberta",
)

# The ONNX model is uploaded to the platform SEPARATELY (not bundled, not in the
# data volume). This path is only used for local runs / the integration test.
MODEL_PATH = os.environ.get(
    "NER_MODEL_PATH",
    os.path.join(REPO_ROOT, "model", "ner_roberta.onnx"),
)

# --- Base model ------------------------------------------------------------
BASE_MODEL = "xlm-roberta-base"

# --- Label scheme (BIO over 3 entity classes) ------------------------------
ENTITY_CLASSES = ["PER", "GPE", "STREET"]

# BIO label list. Order is FIXED and must match the trained model's output axis.
LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-GPE", "I-GPE",
    "B-STREET", "I-STREET",
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
NUM_LABELS = len(LABELS)  # 7
O_ID = LABEL2ID["O"]

# --- MEDDOCAN entity-type -> our 3 classes ---------------------------------
# MEDDOCAN ships ~21 PHI types; we keep only those that map cleanly to
# PER / GPE / STREET and drop the rest (per the task spec). Adjust freely.
MEDDOCAN_TYPE_MAP = {
    # person names
    "NOMBRE_SUJETO_ASISTENCIA": "PER",
    "NOMBRE_PERSONAL_SANITARIO": "PER",
    # geo-political entities
    "TERRITORIO": "GPE",
    "PAIS": "GPE",
    # street / address
    "CALLE": "STREET",
}

# --- Windowing / model I/O -------------------------------------------------
# WINDOW_SIZE is the model's fixed sequence length (the ONNX input is [B, 32]),
# INCLUDING the two special tokens (<s> ... </s>). Content per window is
# WINDOW_SIZE - 2, advanced by WINDOW_STRIDE subword tokens each step.
WINDOW_SIZE = 32
WINDOW_STRIDE = 16
CONTENT_SIZE = WINDOW_SIZE - 2  # 30

# Sentinel for "no word" positions (special tokens / padding) in word_ids.
NO_WORD = -1
