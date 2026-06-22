# ner-roberta

XLM-RoBERTa token-classification NER on the **MEDDOCAN** dataset, restricted to 3
entity classes (`PER`, `GPE`, `STREET`) with a BIO tagging scheme, packaged as a
**Tensorleap** integration.

## Layout

```
ner_roberta/
  config.py        # labels, entity mapping, window params, paths (import-light)
  tokenization.py  # custom punctuation-splitting word tokenizer
  data.py          # MEDDOCAN -> BIO -> subword -> sliding windows
  model.py         # model + float32-input ONNX export wrapper
  train.py         # fine-tune xlm-roberta-base
scripts/
  prepare_data.py  # build processed dataset into the Tensorleap data volume
  export_onnx.py   # export trained model to ONNX
leap_integration.py  # Tensorleap integration (decorator style)
leap.yaml            # Tensorleap config
```

## Pipeline

1. **Preprocess** — custom tokenizer separates punctuation from words; whitespace
   normalized; subword tokenization via the XLM-R tokenizer.
2. **Windows** — documents are split into sliding windows of length 32
   (incl. `<s>`/`</s>`), stride 16.
3. **Model** — `xlm-roberta-base` + a 7-way token-classification head
   (`O`, `B-/I-` × {PER, GPE, STREET}).
4. **Postprocess** — subwords merged back into words (done in visualizers/metrics,
   not the model).

## Data / model placement (Tensorleap conventions)

- **Dataset** -> the Tensorleap **data volume** (`NER_DATASET_DIR`), read via a
  config-driven path — not bundled in git or `leap.yaml`.
- **Model** -> uploaded to the platform **separately** — not bundled.
- **Code / assets** the integration reads -> `leap.yaml` `include`.
