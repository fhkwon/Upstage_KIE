import argparse
from collections import Counter, defaultdict
import csv
from glob import glob
import json
import os
import re
import string

# NOTE: DO NOT MODIFY THE FOLLOWING PATHS
# ---------------------------------------
data_dir = os.environ.get("SM_CHANNEL_EVAL", "../input/data")
output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "./output")
# ---------------------------------------


def read_ground_truths(test_data_dir: str):
    gt_parses = {}
    for filename in glob(test_data_dir):
        with open(filename, "r") as f:
            data = json.load(f)
        filename = os.path.splitext(os.path.basename(filename))[0]
        gt_parses[filename] = data
    return gt_parses


def gen_parsers(output_path: str):
    f = open(output_path, "r", encoding="utf-8")
    pr_parses = defaultdict(lambda: {"company": [], "date": [], "address": [], "total": []})
    for line in csv.reader(f):
        if len(line) == 3:
            text, pred_label, filename = line
            if pred_label != "O":
                if pred_label == "S-COMPANY":
                    pred_label = "company"
                elif pred_label == "S-DATE":
                    pred_label = "date"
                elif pred_label == "S-ADDRESS":
                    pred_label = "address"
                elif pred_label == "S-TOTAL":
                    pred_label = "total"
                pr_parses[filename][pred_label].append(text)
        elif len(line) == 2:
            raise NotImplementedError(f"{output_path} is op test dataset.")

    for (filename, pr_parse) in pr_parses.items():
        for (pred_label, value) in pr_parse.items():
            pr_parse[pred_label] = " ".join(value)
    f.close()
    return pr_parses


def normalize_answer(s, remove_whitespace: bool = False):
    def remove_(text):
        """불필요한 기호 제거"""
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub("《", " ", text)
        text = re.sub("》", " ", text)
        text = re.sub("<", " ", text)
        text = re.sub(">", " ", text)
        text = re.sub("〈", " ", text)
        text = re.sub("〉", " ", text)
        text = re.sub("\(", " ", text)
        text = re.sub("\)", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        return text

    def white_space_fix(text):
        return " ".join(text.split())

    def white_space_remove(text):
        return "".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if remove_whitespace:
        return white_space_remove(remove_punc(lower(remove_(s))))
    else:
        return white_space_fix(remove_punc(lower(remove_(s))))


def get_char_level_f1_score(prediction, ground_truth, remove_whitespace: bool = False):
    prediction_tokens = normalize_answer(prediction, remove_whitespace).split()
    ground_truth_tokens = normalize_answer(ground_truth, remove_whitespace).split()

    # F1 by character
    prediction_Char = []
    for tok in prediction_tokens:
        now = [a for a in tok]
        prediction_Char.extend(now)

    ground_truth_Char = []
    for tok in ground_truth_tokens:
        now = [a for a in tok]
        ground_truth_Char.extend(now)

    common = Counter(prediction_Char) & Counter(ground_truth_Char)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(prediction_Char)
    recall = 1.0 * num_same / len(ground_truth_Char)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def exact_match_score(prediction, ground_truth, remove_whitespace: bool = False):
    return normalize_answer(prediction, remove_whitespace) == normalize_answer(
        ground_truth, remove_whitespace
    )


def evaluation(gt_parses, pr_parses: str, **kwargs):
    assert len(gt_parses) == len(pr_parses)
    parses = defaultdict(lambda: {"gold": dict, "infer": dict})
    f1 = exact_match = exact_match_no_space = total = 0
    entity_score_per_entity = defaultdict(
        lambda: {
            "entity_em": 0.0,
            "entity_em_no_space": 0.0,
            "entity_f1": 0.0,
        }
    )
    total_per_entity = defaultdict(int)

    filenames = list(gt_parses.keys())
    for filename in filenames:
        gt_parse = gt_parses[filename]
        pr_parse = pr_parses[filename]

        for key in ["company", "date", "address", "total"]:
            total += 1
            total_per_entity[key] += 1
            ground_truths = " ".join(gt_parse[key])
            try:
                prediction = " ".join(pr_parse[key])
            except KeyError:
                prediction = ""
            parses[filename][key] = {"gold": ground_truths, "infer": prediction}
            exact_match += exact_match_score(prediction, ground_truths)
            f1 += get_char_level_f1_score(prediction, ground_truths)
            exact_match_no_space += exact_match_score(
                prediction, ground_truths, remove_whitespace=True
            )

            entity_score_per_entity[key]["entity_em"] += exact_match_score(
                prediction, ground_truths
            )
            entity_score_per_entity[key]["entity_em_no_space"] += exact_match_score(
                prediction, ground_truths, remove_whitespace=True
            )
            entity_score_per_entity[key]["entity_f1"] += get_char_level_f1_score(
                prediction, ground_truths
            )

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    exact_match_no_space = 100.0 * exact_match_no_space / total

    # get entity score per entities
    assert len(entity_score_per_entity.keys()) == len(total_per_entity.keys())
    for key in entity_score_per_entity.keys():
        entity_score_per_entity[key]["entity_em"] = (
            100.0 * entity_score_per_entity[key]["entity_em"] / total_per_entity[key]
        )
        entity_score_per_entity[key]["entity_f1"] = (
            100.0 * entity_score_per_entity[key]["entity_f1"] / total_per_entity[key]
        )
        entity_score_per_entity[key]["entity_em_no_space"] = (
            100.0 * entity_score_per_entity[key]["entity_em_no_space"] / total_per_entity[key]
        )

    result = {
        'entity_f1': {
            'value': f1,
            'rank': True,
            'decs': True
        },
        "entity_em" : {
            'value': exact_match,
            'rank': False,
            'decs': True
        },
        "entity_em_no_space": {
            'value': exact_match_no_space,
            'rank': False,
            'decs': True
        }
    }

    return json.dumps(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=data_dir, help="path to read the test data")
    parser.add_argument(
        "--output_dir", type=str, default=output_dir, help="path to read the inference result"
    )
    args = parser.parse_args()

    gt_parses = read_ground_truths(f"{args.data_dir}/test/entities/*")
    pr_parses = gen_parsers(os.path.join(args.output_dir, "output.csv"))

    eval_result = evaluation(gt_parses, pr_parses)
    print(eval_result)


if __name__ == "__main__":
    main()