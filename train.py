#! /usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)

from utils import SROIEDataset, evaluate

logger = logging.getLogger(__name__)

# NOTE: DO NOT MODIFY THE FOLLOWING PATHS
# ---------------------------------------
data_dir = os.environ.get("SM_CHANNEL_TRAIN", "../input/data")
model_dir = os.environ.get("SM_MODEL_DIR", "./model")
output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "./results")
# ---------------------------------------


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    transposed = list(zip(*batch))  
    collated = []
    for items in transposed:
        first = items[0]
        if first is None:
            collated.append(None)
        elif isinstance(first, torch.Tensor):
            collated.append(torch.stack(items, dim=0))
        else:
            collated.append(items)
    return tuple(collated)


def get_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, writer=None):  # noqa: C901
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // max(1, (len(train_dataloader) // args.gradient_accumulation_steps)) + 1
    else:
        t_total = (len(train_dataloader) // max(1, args.gradient_accumulation_steps)) * int(args.num_train_epochs)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    scaler = GradScaler() if args.fp16 else None

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1
    total_bs = args.train_batch_size * args.gradient_accumulation_steps * world_size
    logger.info("  Total train batch size (parallel, distributed, accumulation) = %d", total_bs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()

            input_ids = batch[0].to(args.device)
            attention_mask = batch[1].to(args.device)
            token_type_ids = batch[2].to(args.device) if (batch[2] is not None and model.config.model_type in ["bert", "layoutlm"]) else None
            labels_tensor = batch[3].to(args.device)
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels_tensor}
            if token_type_ids is not None:
                inputs["token_type_ids"] = token_type_ids
            if model.config.model_type.startswith("layoutlm") and len(batch) > 4 and batch[4] is not None:
                inputs["bbox"] = batch[4].to(args.device)

            if args.fp16:
                with autocast():
                    loss = model(**inputs).loss
            else:
                loss = model(**inputs).loss

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and writer is not None:
                    writer.add_scalar("train/loss", loss.item(), global_step)

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 and args.evaluate_during_training:
                    results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                    if writer is not None:
                        for k, v in results.items():
                            writer.add_scalar(f"eval/{k}", v, global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    torch.save(args, os.path.join(ckpt_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", ckpt_dir)

            if args.max_steps > 0 and global_step >= args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step >= args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / max(global_step, 1)


def main():  # noqa: C901
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str, required=True)

    parser.add_argument("--data_dir", default=data_dir, type=str)
    parser.add_argument("--labels", default=os.path.join(data_dir, "labels.txt"), type=str)
    parser.add_argument("--output_dir", default=output_dir, type=str)
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--evaluate_during_training", action="store_true")
    parser.add_argument("--do_lower_case", action="store_true")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)

    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--save_steps", default=50, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fp16_opt_level", default="O1", type=str)  # kept for CLI compatibility
    parser.add_argument("--local_rank", default=-1, type=int)

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        if not args.overwrite_output_dir:
            raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir.")
        elif args.local_rank in [-1, 0]:
            shutil.rmtree(args.output_dir)

    if args.do_eval and not args.do_train and not os.path.exists(args.output_dir):
        raise ValueError(f"Output directory ({args.output_dir}) does not exist. Train and save the model before evaluation.")

    if not os.path.exists(args.output_dir) and args.do_train and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(
        filename=os.path.join(args.output_dir, "train.log") if args.local_rank in [-1, 0] else None,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, fp16: %s",
        args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16,
    )

    set_seed(args)

    labels = get_labels(args.labels)
    num_labels = len(labels)
    pad_token_label_id = CrossEntropyLoss().ignore_index

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, cache_dir=args.cache_dir or None)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path if not args.tokenizer_name else args.tokenizer_name,
        use_fast=True,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir or None,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        cache_dir=args.cache_dir or None,
        use_safetensors=True,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    tb_log_dir = os.path.join(args.output_dir, "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=tb_log_dir) if args.local_rank in [-1, 0] else None

    if args.do_train:
        train_dataset = SROIEDataset(args, tokenizer, labels, pad_token_label_id, mode="train")
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, writer)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if args.do_eval and (args.local_rank in [-1, 0]):
        results, report = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")
        if writer is not None:
            for k, v in results.items():
                writer.add_scalar(f"eval/{k}", v, 0)
        logger.info("***** Eval results *****")
        for key, value in results.items():
            logger.info("  %s = %s", key, value)
        if report:
            logger.info("\n%s", report)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
