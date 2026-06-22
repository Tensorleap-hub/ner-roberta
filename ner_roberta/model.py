"""Model construction + a float32-input wrapper for ONNX export."""

import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification

from .config import BASE_MODEL, ID2LABEL, LABEL2ID, NUM_LABELS


def build_model():
    """`xlm-roberta-base` + a NUM_LABELS-way token-classification head."""
    return AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )


class Float32Wrapper(nn.Module):
    """Expose float32 inputs so the exported ONNX matches Tensorleap encoders.

    Tensorleap input encoders return ``np.float32`` arrays. ONNX token-classifiers
    normally take int64 ``input_ids``/``attention_mask``; this wrapper accepts
    float32, casts to long internally, and returns the raw float32 logits
    (no softmax — postprocessing lives in visualizers/metrics).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(
            input_ids=input_ids.long(),
            attention_mask=attention_mask.long(),
        )
        return out.logits  # [B, T, NUM_LABELS] float32
