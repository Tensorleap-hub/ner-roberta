# ner-roberta

`xlm-roberta-base` token-classification **NER** on the **MEDDOCAN** clinical
dataset, restricted to 3 entity classes (`PER`, `GPE`, `STREET`) with a BIO
tagging scheme — built to be packaged as a **Tensorleap** integration.

Tensorleap is inference-only: it loads a *pre-trained* model and analyzes its
predictions over the data. This repo therefore produces (a) the processed
dataset, and (b) a trained ONNX model, which the integration then consumes.

## Status

| Piece | State |
|---|---|
| Poetry env (Python 3.10) + ML stack + `code-loader` | ✅ done |
| Data pipeline → processed windows in the data volume | ✅ done |
| Fine-tuned model → float32-input ONNX (val entity-F1 **0.96**) | ✅ done |
| `leap_integration.py` + `leap.yaml` (Tensorleap integration) | ⏳ **not yet created** |

## Layout

```
ner_roberta/
  config.py        # labels, entity mapping, window params, paths (import-light)
  tokenization.py  # custom punctuation-splitting word tokenizer (offset-preserving)
  data.py          # MEDDOCAN -> BIO -> subword -> sliding windows
  model.py         # model + float32-input ONNX export wrapper
  train.py         # fine-tune xlm-roberta-base (per-epoch checkpoints, live logs)
scripts/
  prepare_data.py  # build processed dataset into the Tensorleap data volume
  export_onnx.py   # export a trained checkpoint to ONNX (+ verify)
checkpoints/        # HF checkpoint written by training (gitignored)
model/              # ner_roberta.onnx (gitignored; uploaded to Tensorleap separately)
# leap_integration.py / leap.yaml  -> to be added (Tensorleap integration)
```

## Setup

Requires Python 3.10 (the active default 3.9 is too old for the stack). The
Poetry env is created in-project (`.venv/`).

```bash
poetry env use /path/to/python3.10
poetry install
```

Key pins: `datasets <3` (bigbio/meddocan uses a remote loading script via
`trust_remote_code`), `onnxruntime <1.20` (newer wheels drop Python 3.10),
`numpy <2`. Trained/verified with torch 2.12, transformers 4.57, code-loader
1.0.184; Apple-Silicon MPS used for training.

## Usage

```bash
# 1. Build the processed dataset into the data volume
poetry run python scripts/prepare_data.py

# 2. Fine-tune (Apple MPS / CUDA / CPU auto-detected)
poetry run python -m ner_roberta.train --epochs 3

# 3. Export the trained checkpoint to a float32-input ONNX (+ verify)
poetry run python scripts/export_onnx.py
```

## Pipeline

1. **Custom tokenizer** — whitespace is normalized; every punctuation character
   is split into its own token, e.g. `"(hello.there!"` -> `("," "hello", ".",
   "there", "!")`. Punctuation is **kept** as tokens (not dropped), since it
   carries NER signal. Character offsets are preserved so MEDDOCAN entity spans
   align onto words.
2. **BIO labelling** — entity char spans are projected onto words; the first word
   of an entity gets `B-`, the rest `I-`. Subword tokenization (XLM-R) then aligns
   labels to subwords (continuation subwords get the `I-` label).
3. **Windows** — the subword stream is split into sliding windows of length **32**
   (including `<s>`/`</s>`, so 30 content tokens), **stride 16**, tail padded.
4. **Model** — `xlm-roberta-base` + a 7-way token-classification head.
5. **Postprocess** — subwords merged back into words (belongs in
   visualizers/metrics, never baked into the model).

### Label scheme (7 labels)

`O`, `B-PER`/`I-PER`, `B-GPE`/`I-GPE`, `B-STREET`/`I-STREET`. Order is fixed in
`config.py` and matches the model's output axis.

### MEDDOCAN entity-type mapping (in `config.py`, adjustable)

| Our class | MEDDOCAN types |
|---|---|
| `PER` | `NOMBRE_SUJETO_ASISTENCIA`, `NOMBRE_PERSONAL_SANITARIO` |
| `GPE` | `TERRITORIO`, `PAIS` |
| `STREET` | `CALLE` |

All other MEDDOCAN PHI types are dropped.

## Dataset

`bigbio/meddocan` (`meddocan_bigbio_kb`): 500 train / 250 validation / 250 test
documents. After windowing: **26,152** training and **13,762** validation windows
(~23% contain an entity). Splits used: `train` -> training, `validation` ->
validation.

## Model I/O contract

The ONNX takes **float32** inputs (so Tensorleap's float32 encoders feed it
directly; the wrapper casts to int64 internally) and returns raw logits (no
softmax):

| Tensor | Shape | Dtype |
|---|---|---|
| `input_ids` (in) | `[batch, 32]` | float32 |
| `attention_mask` (in) | `[batch, 32]` | float32 |
| `logits` (out) | `[batch, 32, 7]` | float32 |

## Training results

3 epochs, AdamW (lr 5e-5), batch 32, on MPS. Loss is computed only on the first
subword of each word (continuation/special/pad positions masked to `-100`).

| Metric (validation) | Value |
|---|---|
| entity-F1 (micro) | **0.96** |
| PER / GPE / STREET F1 | 0.98 / 0.96 / 0.92 |
| token accuracy | 0.998 |

## Data / model placement (Tensorleap conventions)

- **Dataset** -> the Tensorleap **data volume** (`NER_DATASET_DIR`, default
  `/Users/yotamazriel/tensorleap/data/ner-roberta`), read via a config-driven
  path — not bundled in git or `leap.yaml`.
- **Model** -> `NER_MODEL_PATH` (default `model/ner_roberta.onnx`), uploaded to
  the platform **separately** — not bundled anywhere.
- **Code / assets** the integration reads -> `leap.yaml` `include` (once the
  integration is authored).
