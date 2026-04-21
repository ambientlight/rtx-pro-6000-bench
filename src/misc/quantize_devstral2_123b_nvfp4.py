#!/usr/bin/env python3
"""
Quantize Devstral-2-123B to NVFP4 using LLM Compressor.

Based on the same approach used for Devstral-Small-2-24B-Instruct-2512-nvfp4:
- NVFP4 scheme (W4A4 with dual scaling)
- One-shot calibration with nvidia/OpenCodeInstruct
- Keep lm_head and multimodal components in high precision
- Sequential onloading for large models (default in LLM Compressor)
"""

import os
import json
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer

# Model paths - using local path
MODEL_PATH = "/mnt/hot/ambientlight/models/devstral-2-123b-instruct-2512"
OUT_DIR = "/mnt/hot/ambientlight/models/devstral-2-123b-instruct-2512-nvfp4"

# Calibration settings
# For 123B model, use longer sequences but fewer samples due to memory constraints
MAX_SEQ_LEN = 8192  # Can increase if you have enough VRAM
NUM_CALIB_SAMPLES = 128


def format_chat(tokenizer, user_text: str, assistant_text: str) -> str:
    """
    Build a chat transcript for calibration.
    """
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback: simple concatenation
        return f"User: {user_text}\nAssistant: {assistant_text}"


def main():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    # Load tokenizer.json directly with tokenizers library, then wrap in PreTrainedTokenizerFast
    # This bypasses the tokenizer_config.json which has format issues with this transformers version
    tokenizer_path = os.path.join(MODEL_PATH, "tokenizer.json")
    backend_tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend_tokenizer)

    # Set special tokens manually
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "<unk>"
    tokenizer.pad_token = "<pad>"

    print(f"Loading model from {MODEL_PATH}...")
    print("Using device_map=None for sequential onloading (CPU load, layers moved to GPU during calibration)")

    # IMPORTANT for huge models:
    # Load onto CPU (device_map=None) so LLM Compressor sequential onloading
    # can move layers to GPU one at a time during calibration.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map=None,  # CPU load for sequential onloading
    )

    print("Preparing calibration dataset from nvidia/OpenCodeInstruct...")
    # Stream OpenCodeInstruct so we don't materialize the whole dataset
    stream = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)

    samples = []
    for ex in stream:
        user_text = ex.get("input", "") or ""
        assistant_text = ex.get("output", "") or ""

        if not user_text.strip():
            continue

        # If outputs are missing, use a placeholder
        if not assistant_text.strip():
            assistant_text = "OK."

        text = format_chat(tokenizer, user_text, assistant_text)
        samples.append({"text": text})

        if len(samples) >= NUM_CALIB_SAMPLES:
            break

    print(f"Collected {len(samples)} calibration samples")
    calib_ds = Dataset.from_list(samples)

    # NVFP4 modifier - same as the working small model recipe
    # Ignore list matches the modules_to_not_convert from the original FP8 config
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="NVFP4",
        ignore=[
            "lm_head",                       # Keep output head unquantized
            "re:.*vision_tower.*",           # Vision components (if present)
            "re:.*multi_modal_projector.*",  # Multimodal projector (if present)
            "re:visual.*",                   # Other visual components
            "re:.*video_tower.*",            # Video components (harmless if not present)
            "re:.*audio_tower.*",            # Audio components (harmless if not present)
        ],
    )

    print("Starting one-shot quantization...")
    print(f"  - Max sequence length: {MAX_SEQ_LEN}")
    print(f"  - Calibration samples: {NUM_CALIB_SAMPLES}")

    # Run one-shot calibration + quantization
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        tokenizer=tokenizer,
        text_column="text",
        max_seq_length=MAX_SEQ_LEN,
        num_calibration_samples=NUM_CALIB_SAMPLES,
        pad_to_max_length=True,
        concatenate_data=True,
    )

    # Save compressed-tensors checkpoint
    print(f"Saving quantized model to {OUT_DIR}...")
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)

    # Patch config.json: remove the original FP8 quantization_config if it leaked through
    # The original 123B model has quant_method="fp8" which we want to replace with our NVFP4
    cfg_path = os.path.join(OUT_DIR, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        qc = cfg.get("quantization_config")
        if isinstance(qc, dict) and qc.get("quant_method") == "fp8":
            del cfg["quantization_config"]
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            print("Patched config.json: removed stale fp8 quantization_config")
    except Exception as e:
        print(f"(info) config patch skipped: {e}")

    print(f"\nDone! NVFP4 quantized model saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
