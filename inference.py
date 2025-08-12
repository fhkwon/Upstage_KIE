# -*- coding: utf-8 -*-
import argparse
import csv
import logging
import os
import shutil
from typing import List

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForTokenClassification, AutoTokenizer

from utils import evaluate

# NOTE: DO NOT MODIFY
data_dir = os.environ.get("SM_CHANNEL_EVAL", "../input/data")
model_dir = os.environ.get("SM_CHANNEL_MODEL", "./model")
output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "./output")
# -------------------

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    return labels if "O" in labels else ["O"] + labels


def _to_s_label(tag: str) -> str:
    t = (tag or "O").strip().upper()
    ent = t.split("-", 1)[-1] if "-" in t else t
    return f"S-{ent}" if ent in {"COMPANY", "DATE", "ADDRESS", "TOTAL"} else "O"


def main() -> None:
    parser = argparse.ArgumentParser()

    # 동일 CLI 유지
    parser.add_argument("--model_type", required=True, choices=["bert", "roberta", "layoutlm"])
    parser.add_argument("--model_name_or_path", required=True, type=str)

    parser.add_argument("--data_dir", default=data_dir, type=str)
    parser.add_argument("--mode", default="test", choices=["test", "op_test"], type=str)
    parser.add_argument("--model_dir", default=model_dir, type=str)
    parser.add_argument("--output_dir", default=output_dir, type=str)
    parser.add_argument("--labels", default=os.path.join(data_dir, "labels.txt"), type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_predict:
        if not args.overwrite_output_dir:
            raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. "
                             f"Use --overwrite_output_dir.")
        if args.local_rank in [-1, 0]:
            shutil.rmtree(args.output_dir)

    if not os.path.exists(args.output_dir) and args.do_predict and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    args.n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    args.device = device

    labels = get_labels(args.labels)
    pad_token_label_id = CrossEntropyLoss().ignore_index

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=(args.cache_dir or None), do_lower_case=args.do_lower_case
    )
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)

    if not args.do_predict:
        logger.info("do_predict가 설정되지 않아 종료합니다.")
        return

    result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode=args.mode)

    # {mode}_results.txt
    res_path = os.path.join(args.output_dir, f"{args.mode}_results.txt")
    with open(res_path, "w", encoding="utf-8") as w:
        for k in sorted(result.keys()):
            w.write(f"{k} = {result[k]}\n")

    # output.csv: 입력 파일 순회하며 예측 매핑
    data_path = os.path.join(args.data_dir, f"{args.mode}.txt")
    out_csv = os.path.join(args.output_dir, "output.csv")

    total_lines = written = skipped_no_pred = skipped_ctrl = 0
    sent_idx = tok_idx = 0

    with open(out_csv, "w", encoding="utf-8", newline="") as f_out, \
            open(data_path, "r", encoding="utf-8") as f_in:
        writer = csv.writer(f_out, lineterminator="\n")

        for raw in f_in:
            total_lines += 1
            line = raw.rstrip("\n")
            is_ctrl = raw.startswith("-DOCSTART-") or line.strip() == ""
            if is_ctrl:
                skipped_ctrl += 1
                # 문장 경계 이동
                if sent_idx < len(predictions) and tok_idx == len(predictions[sent_idx]):
                    sent_idx += 1
                tok_idx = 0
                continue

            parts = line.split("\t") if "\t" in line else line.split()
            token = parts[0]

            if sent_idx < len(predictions) and tok_idx < len(predictions[sent_idx]):
                pred_tag = predictions[sent_idx][tok_idx]
            else:
                pred_tag = "O"
                skipped_no_pred += 1
            tok_idx += 1

            if args.mode == "op_test":
                filename = parts[2] if len(parts) >= 3 else ""
                writer.writerow([token, _to_s_label(pred_tag), filename])
            else:
                writer.writerow([token, pred_tag])
            written += 1

    logger.info("Saved results: %s", os.path.abspath(res_path))
    logger.info("Saved predictions: %s", os.path.abspath(out_csv))
    logger.info("Lines total=%d, written=%d, skipped(no_pred)=%d, skipped(ctrl)=%d",
                total_lines, written, skipped_no_pred, skipped_ctrl)


if __name__ == "__main__":
    main()
