"""Export a trained checkpoint to a float32-input ONNX model.

    poetry run python scripts/export_onnx.py [checkpoint_dir]

Produces ``config.MODEL_PATH`` with:
  inputs : input_ids [B, 32] float32, attention_mask [B, 32] float32
  output : logits [B, 32, NUM_LABELS] float32
The batch axis is dynamic; the sequence length is fixed at WINDOW_SIZE.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoModelForTokenClassification  # noqa: E402

from ner_roberta.config import MODEL_PATH, NUM_LABELS, WINDOW_SIZE  # noqa: E402
from ner_roberta.model import Float32Wrapper  # noqa: E402


def main(ckpt: str = "checkpoints", out: str = MODEL_PATH) -> None:
    model = AutoModelForTokenClassification.from_pretrained(ckpt).eval()
    wrapper = Float32Wrapper(model).eval()

    os.makedirs(os.path.dirname(out), exist_ok=True)
    dummy_ids = torch.zeros(1, WINDOW_SIZE, dtype=torch.float32)
    dummy_mask = torch.ones(1, WINDOW_SIZE, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        out,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
        dynamo=False,
    )
    print(f"exported -> {out}")

    import onnxruntime as ort

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    print("inputs :", [(i.name, i.type, i.shape) for i in sess.get_inputs()])
    print("outputs:", [(o.name, o.type, o.shape) for o in sess.get_outputs()])
    logits = sess.run(None, {
        "input_ids": dummy_ids.numpy(),
        "attention_mask": dummy_mask.numpy(),
    })[0]
    print(f"onnx logits shape={logits.shape} dtype={logits.dtype} "
          f"(expected (1, {WINDOW_SIZE}, {NUM_LABELS}))")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoints")
