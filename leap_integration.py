"""Tensorleap integration for XLM-RoBERTa MEDDOCAN NER (3-class BIO).

Decorator-style integration. The model is a float32-input ONNX token classifier
(input_ids/attention_mask [B,32] -> logits [B,32,7]); the dataset is read from
the Tensorleap data volume via a config-driven path.
"""

import os
import pickle
from typing import List

import numpy as np

from code_loader.contract.datasetclasses import (
    PredictionTypeHandler,
    PreprocessResponse,
    SamplePreprocessResponse,
)
from code_loader.contract.enums import (
    DataStateType,
    DatasetMetadataType,
    LeapDataType,
    MetricDirection,
)
from code_loader.contract.visualizer_classes import LeapText, LeapTextMask
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_custom_loss,
    tensorleap_custom_metric,
    tensorleap_custom_visualizer,
    tensorleap_gt_encoder,
    tensorleap_input_encoder,
    tensorleap_integration_test,
    tensorleap_load_model,
    tensorleap_metadata,
    tensorleap_preprocess,
)

from ner_roberta.config import (
    DATASET_DIR,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    MODEL_PATH,
    NUM_LABELS,
    O_ID,
    WINDOW_SIZE,
)


# --------------------------------------------------------------------------- #
# Dataset access (read from the data volume via a config-driven path)
# --------------------------------------------------------------------------- #
_SPLIT_CACHE = {}


def _load_split(state_key: str) -> dict:
    """Load a processed split pickle ({"sample_ids": [...], "samples": {...}})."""
    if state_key not in _SPLIT_CACHE:
        with open(os.path.join(DATASET_DIR, f"{state_key}.pkl"), "rb") as f:
            _SPLIT_CACHE[state_key] = pickle.load(f)
    return _SPLIT_CACHE[state_key]


@tensorleap_preprocess()
def preprocess() -> List[PreprocessResponse]:
    train = _load_split("training")
    val = _load_split("validation")
    return [
        PreprocessResponse(
            sample_ids=train["sample_ids"], data=train["samples"],
            state=DataStateType.training,
        ),
        PreprocessResponse(
            sample_ids=val["sample_ids"], data=val["samples"],
            state=DataStateType.validation,
        ),
    ]


# --------------------------------------------------------------------------- #
# Input encoders (one per model input; unbatched [32] float32)
# --------------------------------------------------------------------------- #
@tensorleap_input_encoder(name="input_ids", channel_dim=-1)
def input_ids_encoder(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data[sample_id]["input_ids"].astype(np.float32)


@tensorleap_input_encoder(name="attention_mask", channel_dim=-1)
def attention_mask_encoder(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data[sample_id]["attention_mask"].astype(np.float32)


# --------------------------------------------------------------------------- #
# Model (single output: per-token logits over the 7 BIO labels)
# --------------------------------------------------------------------------- #
prediction_types = [
    PredictionTypeHandler(name="ner_logits", labels=LABELS, channel_dim=-1),
]


@tensorleap_load_model(prediction_types)
def load_model():
    import onnxruntime as ort

    return ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------- #
# Ground truth (one-hot per-subword BIO labels, [32, 7] — aligned to the logits)
# --------------------------------------------------------------------------- #
@tensorleap_gt_encoder(name="ner_labels")
def ner_labels_gt(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    labels = preprocess.data[sample_id]["labels"]
    onehot = np.zeros((WINDOW_SIZE, NUM_LABELS), dtype=np.float32)
    onehot[np.arange(WINDOW_SIZE), labels] = 1.0
    return onehot


# --------------------------------------------------------------------------- #
# Custom loss (per-sample masked softmax cross-entropy; batched in -> 1D out)
# --------------------------------------------------------------------------- #
@tensorleap_custom_loss("token_ce_loss")
def token_ce_loss(prediction: np.ndarray, ner_labels: np.ndarray,
                  attention_mask: np.ndarray) -> np.ndarray:
    # prediction [B,32,7] logits, ner_labels [B,32,7] one-hot, attention_mask [B,32]
    z = prediction - prediction.max(axis=-1, keepdims=True)
    log_prob = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))  # [B,32,7]
    ce = -(ner_labels * log_prob).sum(axis=-1)                    # [B,32]
    mask = (attention_mask > 0.5).astype(np.float32)             # ignore padding
    per_sample = (ce * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1.0, None)
    return per_sample.astype(np.float32)                          # [B]


# --------------------------------------------------------------------------- #
# Custom metric (per-sample token accuracy over real, non-pad tokens)
# --------------------------------------------------------------------------- #
@tensorleap_custom_metric("token_accuracy", direction=MetricDirection.Upward)
def token_accuracy(prediction: np.ndarray, ner_labels: np.ndarray,
                   attention_mask: np.ndarray) -> np.ndarray:
    pred = prediction.argmax(axis=-1)             # [B,32]
    gold = ner_labels.argmax(axis=-1)             # [B,32]
    mask = attention_mask > 0.5                   # [B,32] ignore padding
    correct = ((pred == gold) & mask).sum(axis=1)
    total = np.clip(mask.sum(axis=1), 1, None)
    return (correct / total).astype(np.float32)   # [B]


# --------------------------------------------------------------------------- #
# Metadata (scalars for slicing/analysis: document, window, entity counts)
# --------------------------------------------------------------------------- #
@tensorleap_metadata("meta", {
    "doc_id": DatasetMetadataType.string,
    "window_index": DatasetMetadataType.int,
    "num_real_tokens": DatasetMetadataType.int,
    "num_words": DatasetMetadataType.int,
    "num_per": DatasetMetadataType.int,
    "num_gpe": DatasetMetadataType.int,
    "num_street": DatasetMetadataType.int,
    "has_entity": DatasetMetadataType.boolean,
})
def metadata(sample_id: str, preprocess: PreprocessResponse) -> dict:
    s = preprocess.data[sample_id]
    labels = s["labels"]
    starts = s["word_start_mask"] == 1

    def count(begin_label: str) -> int:
        return int(((labels == LABEL2ID[begin_label]) & starts).sum())

    n_per, n_gpe, n_street = count("B-PER"), count("B-GPE"), count("B-STREET")
    return {
        "doc_id": s["doc_id"],
        "window_index": int(s["window_index"]),
        "num_real_tokens": int(s["attention_mask"].sum()),
        "num_words": int(starts.sum()),
        "num_per": n_per,
        "num_gpe": n_gpe,
        "num_street": n_street,
        "has_entity": bool(n_per + n_gpe + n_street > 0),
    }


# --------------------------------------------------------------------------- #
# Visualizers (NER token highlighting; tokens read from the SamplePreprocess-
# Response so no tokenizer is needed at runtime)
# --------------------------------------------------------------------------- #
def _spr_sample(spr: SamplePreprocessResponse) -> dict:
    sid = spr.sample_ids
    if isinstance(sid, np.ndarray):
        sid = sid.reshape(-1)[0]
    return spr.preprocess_response.data[sid]


@tensorleap_custom_visualizer("ner_predicted", LeapDataType.TextMask)
def ner_predicted_viz(prediction: np.ndarray,
                      spr: SamplePreprocessResponse) -> LeapTextMask:
    tokens = list(_spr_sample(spr)["tokens"])
    if prediction.ndim == 3:        # drop batch dim if present ([B,32,7] -> [32,7])
        prediction = prediction[0]
    mask = prediction.argmax(axis=-1).astype(np.uint8)  # [32] predicted label id
    return LeapTextMask(mask=mask, text=tokens, labels=LABELS)


@tensorleap_custom_visualizer("ner_ground_truth", LeapDataType.TextMask)
def ner_ground_truth_viz(spr: SamplePreprocessResponse) -> LeapTextMask:
    sample = _spr_sample(spr)
    tokens = list(sample["tokens"])
    mask = sample["labels"].astype(np.uint8)            # [32] gt label id
    return LeapTextMask(mask=mask, text=tokens, labels=LABELS)


# --------------------------------------------------------------------------- #
# Integration test (kept thin: only decorated calls + the ONNX inference)
# --------------------------------------------------------------------------- #
@tensorleap_integration_test()
def integration_test(sample_id: str, preprocess: PreprocessResponse):
    ids = input_ids_encoder(sample_id, preprocess)
    mask = attention_mask_encoder(sample_id, preprocess)
    gt = ner_labels_gt(sample_id, preprocess)
    model = load_model()
    in0 = model.get_inputs()[0].name
    in1 = model.get_inputs()[1].name
    logits = model.run(None, {in0: ids, in1: mask})[0]
    token_ce_loss(logits, gt, mask)
    token_accuracy(logits, gt, mask)
    metadata(sample_id, preprocess)
    spr = SamplePreprocessResponse(sample_id, preprocess)
    ner_predicted_viz(logits, spr)
    ner_ground_truth_viz(spr)


if __name__ == "__main__":
    subsets = preprocess()
    for subset in subsets:
        if subset.state not in (DataStateType.training, DataStateType.validation):
            continue
        for sid in subset.sample_ids[:3]:
            integration_test(sid, subset)
        print(f"ran integration_test on 3 {subset.state.value} samples")
