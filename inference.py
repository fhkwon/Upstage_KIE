import argparse
import logging
import os
import csv
import shutil

import torch
from torch.nn import CrossEntropyLoss
from transformers import (BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig,
                          BertForTokenClassification, BertTokenizer,
                          LayoutLMConfig, LayoutLMForTokenClassification,
                          RobertaConfig, RobertaForTokenClassification,
                          RobertaTokenizer)
from utils import evaluate

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf_archive_map.keys())
        for conf_archive_map in (
            BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
            ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
            LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "layoutlm": (LayoutLMConfig, LayoutLMForTokenClassification, BertTokenizer),
}

# NOTE: DO NOT MODIFY THE FOLLOWING PATHS
# ---------------------------------------
data_dir = os.environ.get("SM_CHANNEL_EVAL", "../input/data")
model_dir = os.environ.get("SM_CHANNEL_MODEL", "./model")
output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "./output")
# ---------------------------------------


def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def main():  # noqa C901
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS),
    )

    ## Other parameters
    parser.add_argument(
        "--data_dir",
        default=data_dir,
        type=str,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--mode",
        default="test",
        type=str,
        choices=["test", "op_test"],
        help="The type of inference. The `test` mode indicates the f1 score of the bbox unit of the referenced BIO tag, "
             "and the `op_test` mode indicates the entity f1 score of the final result."
    )
    parser.add_argument(
        "--model_dir",
        default=model_dir,
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--output_dir",
        default=output_dir,
        type=str,
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--labels",
        default=os.path.join(data_dir, "labels.txt"),
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_predict
    ):
        if not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir
                )
            )
        else:
            if args.local_rank in [-1, 0]:
                shutil.rmtree(args.output_dir)

    if (
        not os.path.exists(args.output_dir)
        and args.do_predict
        and args.local_rank in [-1, 0]
    ):
        os.makedirs(args.output_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.device = device

    # Set seed
    labels = get_labels(args.labels)

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # Predict!
    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case
    )
    model = model_class.from_pretrained(args.model_dir)
    model.to(args.device)
    result, predictions = evaluate(
        args, model, tokenizer, labels, pad_token_label_id, mode=args.mode
    )
    # Save results
    output_test_results_file = os.path.join(args.output_dir, f"{args.mode}_results.txt")
    with open(output_test_results_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("{} = {}\n".format(key, str(result[key])))
    # Save predictions
    output_test_predictions_file = os.path.join(args.output_dir, f"output.csv")
    with open(output_test_predictions_file, "w", encoding="utf8") as writer:
        csv_writer = csv.writer(writer, lineterminator='\n')
        with open(os.path.join(args.data_dir, f"{args.mode}.txt"), "r", encoding="utf8") as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = [line.split()[0], predictions[example_id].pop(0)]
                    if args.mode == "op_test":
                        output_line += [line.split()[-1]]
                    csv_writer.writerow(output_line)
                else:
                    logger.warning(
                        "Maximum sequence length exceeded: No prediction for '%s'.",
                        line.split()[0],
                    )


if __name__ == "__main__":
    main()
