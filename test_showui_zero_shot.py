"""
Refactored & optimized tester for Qwen2VL models.

Key improvements:
- Cleaner structure & batching interface (keeps per-sample vision preprocessing to preserve compatibility).
- Robust prediction parsing with multiple fallbacks.
- Proper device handling and minimal .to(device) moves.
- Better logging, error capture, optional JSON export of errors.
- Command-line args for easy reuse and experimentation.
- Type hints and docstrings for clarity.

Usage:
  python test_showui_zero_shot.py --dataset_name dataset --split test --model_name showlab/ShowUI-2B --batch_size 1

Notes:
- Many Vision-Language pipelines require batch_size = 1 for stable behavior; keep that in mind.
- This script intentionally preserves `process_vision_info(...)` usage per-sample to avoid changing upstream I/O expectations.
"""

from __future__ import annotations
import argparse
import ast
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import Counter

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # type: ignore
from qwen_vl_utils import process_vision_info  # keep existing helper for vision preprocessing


# -------------------------
# Configuration dataclass
# -------------------------
@dataclass
class Config:
    model_name: str
    dataset_name: str
    split: str = "test"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16
    batch_size: int = 1
    max_new_tokens: int = 128
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None
    errors_output: Optional[str] = None
    no_gpu: bool = False
    verbose: bool = False


# -------------------------
# Utility functions
# -------------------------
def summarize_results_by_category(results, dataset):
    """
    Summarize test results by platform (from file_name prefix)
    and by data_type (from dataset entry).
    Returns a dict and also prints a pretty table.
    """
    # Counters for total and correct predictions by group
    group_total = Counter()
    group_correct = Counter()

    for entry in dataset:
        platform = entry["file_name"].split("_")[0]
        dtype = entry.get("data_type", "unknown")
        key = (platform, dtype)
        group_total[key] += 1

    for err in results["errors"]:
        platform = err["file_name"].split("_")[0]
        dtype = "unknown"
        # attempt to find data_type in error entry (if stored)
        if "data_type" in err:
            dtype = err["data_type"]
        else:
            # fallback: infer from dataset if available
            match = next((e for e in dataset if e["file_name"] == err["file_name"]), None)
            if match:
                dtype = match.get("data_type", "unknown")
        key = (platform, dtype)
        # mark as incorrect (no increment in correct)
        pass

    # derive correct = total - incorrect
    group_incorrect = Counter()
    for err in results["errors"]:
        platform = err["file_name"].split("_")[0]
        dtype = next((e.get("data_type", "unknown") for e in dataset if e["file_name"] == err["file_name"]), "unknown")
        group_incorrect[(platform, dtype)] += 1

    for key, total in group_total.items():
        correct = total - group_incorrect.get(key, 0)
        group_correct[key] = correct

    # compute accuracy
    rows = []
    for (platform, dtype), total in sorted(group_total.items()):
        correct = group_correct[(platform, dtype)]
        acc = correct / total if total > 0 else 0
        rows.append({
            "Platform": platform,
            "DataType": dtype,
            "Accuracy": round(acc * 100, 1),
            "Correct": correct,
            "Total": total,
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("Per-Platform and DataType Accuracy")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    avg_acc = df["Accuracy"].mean()
    print(f"\nOverall Average Accuracy: {avg_acc:.1f}%")

    return df

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s: %(message)s")


def safe_eval_pair(text: str) -> Optional[Tuple[float, float]]:
    """
    Try several strategies to parse a 2-number coordinate from model output.
    Returns a tuple (x, y) or None if parsing fails.
    """
    text = text.strip()
    # 1) literal eval (e.g. "[0.23, 0.45]" or "(0.23, 0.45)")
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            x, y = float(obj[0]), float(obj[1])
            return x, y
        if isinstance(obj, dict):
            # common keys
            for kx, ky in (("x", "y"), ("cx", "cy"), ("left", "top")):
                if kx in obj and ky in obj:
                    return float(obj[kx]), float(obj[ky])
    except Exception:
        pass

    # 2) regex float extractor (pick first two numbers)
    floats = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if len(floats) >= 2:
        try:
            x, y = float(floats[0]), float(floats[1])
            return x, y
        except Exception:
            pass

    # 3) try to find "x:..., y:..." style
    m = re.search(r"x[:=]\s*([-+]?\d*\.\d+|\d+).*?y[:=]\s*([-+]?\d*\.\d+|\d+)", text, flags=re.I | re.S)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except Exception:
            pass

    return None


def is_click_in_bbox(click_xy: Sequence[float], bbox: Sequence[float]) -> bool:
    """
    Check whether click (x, y) is inside bbox [x_min, y_min, x_max, y_max].
    Assumes normalized coordinates (0..1). If bbox values exceed 1, treat them as pixels
    relative to click being in the same coordinate space — caller should ensure consistent units.
    """
    if not click_xy or len(click_xy) != 2 or not bbox or len(bbox) != 4:
        return False
    try:
        x, y = float(click_xy[0]), float(click_xy[1])
        x_min, y_min, x_max, y_max = map(float, bbox)
    except Exception:
        return False

    return x_min <= x <= x_max and y_min <= y <= y_max


def move_tensor_dict_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move all torch.Tensor values in dict to device; return new dict (shallow copy)."""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


# -------------------------
# Core testing function
# -------------------------
def test_model(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    dataset: Dataset,
    cfg: Config,
) -> Dict[str, Any]:
    """
    Evaluate the model on the dataset.
    Returns a dict with metrics and errors list.
    """
    device = torch.device("cpu" if cfg.no_gpu or cfg.device == "cpu" or not torch.cuda.is_available() else cfg.device)
    logging.info(f"Using device: {device}")

    # Prepare dataloader
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=lambda b: b)

    model.to(device)
    model.eval()

    system_prompt = (
        "Based on the screenshot of the page, I give a text description and you give its corresponding location. "
        "The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
    )

    correct = 0
    total = 0
    errors: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            # batch is a list of samples (dict)
            for sample in batch:
                total += 1
                file_name = sample.get("file_name", "unknown")
                try:
                    instruction = sample["instruction"]
                    gt_bbox = sample["bbox"]

                    # Build the message structure expected by the processor / vision helper
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": system_prompt},
                                {"type": "image", "image": sample["image"], "min_pixels": cfg.min_pixels, "max_pixels": cfg.max_pixels},
                                {"type": "text", "text": instruction},
                            ],
                        }
                    ]

                    # prepare text prompt (keep tokenize=False as in original)
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    # process vision info (keep original helper)
                    image_inputs, video_inputs = process_vision_info(messages)

                    # prepare model inputs (processor will batch if we pass lists; here we pass a 1-element batch)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )

                    # move tensors to device
                    inputs = move_tensor_dict_to_device(inputs, device)

                    # generate
                    generated_ids = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens)

                    # generated_ids and inputs["input_ids"] are tensors of shape (batch, seq_len)
                    # compute trimmed outputs (remove prompt tokens)
                    input_id_lens = [len(in_ids) for in_ids in inputs["input_ids"]]
                    # convert generated to list-of-lists and trim per sample
                    generated_trimmed: List[List[int]] = []
                    gen_cpu = generated_ids.cpu().tolist()
                    for i, out_ids in enumerate(gen_cpu):
                        start = input_id_lens[i]
                        trimmed = out_ids[start:] if len(out_ids) > start else []
                        generated_trimmed.append(trimmed)

                    # decode predicted text
                    output_texts = processor.batch_decode(
                        generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )

                    output_text = output_texts[0].strip()
                    parsed = safe_eval_pair(output_text)
                    if not parsed:
                        raise ValueError(f"Could not parse coordinates from model output: {output_text!r}")

                    x, y = parsed
                    is_correct = is_click_in_bbox((x, y), gt_bbox)

                    if is_correct:
                        correct += 1
                    else:
                        errors.append({
                            "file_name": file_name,
                            "instruction": instruction,
                            "predicted": (x, y),
                            "ground_truth_bbox": gt_bbox,
                            "output_text": output_text,
                        })

                    if cfg.verbose:
                        logging.debug(f"{file_name}: pred={(x,y)} gt={gt_bbox} correct={is_correct}")

                except Exception as exc:
                    logging.exception(f"Error processing sample {file_name}: {exc}")
                    # capture the error with context
                    errors.append({
                        "file_name": file_name,
                        "instruction": sample.get("instruction"),
                        "error": str(exc),
                        "ground_truth_bbox": sample.get("bbox"),
                        "raw_output": output_text if 'output_text' in locals() else None,
                    })

    accuracy = (correct / total) if total > 0 else 0.0
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "incorrect": total - correct,
        "errors": errors,
    }

    # optionally persist errors
    if cfg.errors_output and errors:
        try:
            with open(cfg.errors_output, "w", encoding="utf-8") as fh:
                json.dump(results, fh, ensure_ascii=False, indent=2)
            logging.info(f"Wrote errors to {cfg.errors_output}")
        except Exception:
            logging.exception(f"Failed to write errors to {cfg.errors_output}")

    return results


# -------------------------
# Helper to load model & processor
# -------------------------
def load_model_and_processor(model_name: str, device: str, dtype: torch.dtype, min_pixels: Optional[int], max_pixels: Optional[int]):
    # load model
    logging.info(f"Loading model {model_name} (dtype={dtype})")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )

    # processor with limits
    logging.info("Loading processor")
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return model, processor


# -------------------------
# CLI and main
# -------------------------
def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2VL model on a dataset.")
    parser.add_argument("--model_name", default="showlab/ShowUI-2B")
    parser.add_argument("--dataset_name", default="dataset", help="Name or path accepted by datasets.load_dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1344 * 28 * 28)
    parser.add_argument("--errors_output", default=None, help="Optional JSON path to save errors and results")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    return Config(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        split=args.split,
        device=args.device,
        dtype=torch.float16,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        errors_output=args.errors_output,
        no_gpu=args.no_gpu,
        verbose=args.verbose,
    )


def main():
    cfg = parse_args()
    setup_logging(cfg.verbose)

    # load dataset
    logging.info(f"Loading dataset {cfg.dataset_name} split={cfg.split}")
    dataset = load_dataset(cfg.dataset_name, split=cfg.split)

    # load model & processor
    model, processor = load_model_and_processor(cfg.model_name, cfg.device, cfg.dtype, cfg.min_pixels, cfg.max_pixels)

    # run test
    results = test_model(model=model, processor=processor, dataset=dataset, cfg=cfg)

    # summary
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Total samples: {results['total']}")
    print(f"Correct predictions: {results['correct']}")
    print(f"Incorrect predictions: {results['incorrect']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print("=" * 60)


    # optional: print first few errors
    if results["errors"]:
        print("\nFirst 5 errors (if available):")
        for i, e in enumerate(results["errors"][:5], 1):
            print(f"\n{i}. File: {e.get('file_name')}")
            print(f"   Instruction: {e.get('instruction')}")
            if "predicted" in e:
                print(f"   Predicted: {e.get('predicted')}")
                print(f"   Ground truth bbox: {e.get('ground_truth_bbox')}")
                print(f"   Output text: {e.get('output_text')}")
            if "error" in e:
                print(f"   Error: {e.get('error')}")


if __name__ == "__main__":
    main()
