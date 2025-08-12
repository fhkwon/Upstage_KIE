# utils.py
import logging
import os
from typing import List

import numpy as np
import torch
from torch import serialization
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _infer_model_type(args, tokenizer=None, model=None) -> str:
    """args/tokenizer/model에서 model_type을 유추합니다."""
    if getattr(args, "model_type", None):
        return args.model_type.lower()
    name = None
    if getattr(getattr(model, "config", None), "model_type", None):
        name = model.config.model_type
    if name is None and tokenizer is not None:
        name = getattr(tokenizer, "model_type", None) or tokenizer.__class__.__name__.lower()
    name = (name or "bert").lower()
    for key in ("layoutlm", "roberta", "xlnet", "bert"):
        if key in name:
            return key
    return name

def _filter_inputs_for_model(model, batch):
    allowed = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    mt = getattr(model.config, "model_type", "")
    if mt in {"layoutlm"}:
        allowed |= {"bbox"}
    elif mt in {"layoutlmv2"}:
        allowed |= {"bbox", "image"}          # v2는 feature_extractor 사용 시 image 텐서
    elif mt in {"layoutlmv3"}:
        allowed |= {"bbox", "pixel_values"}   # v3는 pixel_values 사용

    pruned = {k: v for k, v in batch.items() if k in allowed and v is not None}
    # token_type_ids가 없는 토크나이저의 경우 제거
    if "token_type_ids" in pruned and pruned["token_type_ids"] is None:
        pruned.pop("token_type_ids")
    return pruned

class SROIEDataset(Dataset):
    """SROIE 포맷 데이터셋 → 텐서 특징으로 변환합니다."""

    def __init__(self, args, tokenizer, labels, pad_token_label_id, mode):
        if args.local_rank not in [-1, 0] and mode == "train":
            torch.distributed.barrier()

        cached_features_file = os.path.join(
            args.data_dir,
            f"cached_{mode}_{list(filter(None, getattr(args, 'model_name_or_path', 'model').split('/'))).pop()}_{args.max_seq_length}",
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with serialization.safe_globals([InputFeatures]):
                features = torch.load(cached_features_file, weights_only=False)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples = read_examples_from_file(args.data_dir, mode)
            model_type = _infer_model_type(args, tokenizer=tokenizer)
            features = convert_examples_to_features(
                examples,
                labels,
                args.max_seq_length,
                tokenizer,
                cls_token_at_end=(model_type == "xlnet"),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if model_type == "xlnet" else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=(model_type == "roberta"),
                pad_on_left=(model_type == "xlnet"),
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if model_type == "xlnet" else 0,
                pad_token_label_id=pad_token_label_id,
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(features, cached_features_file)

        if args.local_rank == 0 and mode == "train":
            torch.distributed.barrier()

        self.features = features
        self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        self.all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        self.all_bboxes = torch.tensor([f.boxes for f in features], dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_segment_ids[index],
            self.all_label_ids[index],
            self.all_bboxes[index],
        )


class InputExample:
    """단일 문서 예제"""

    def __init__(self, guid, words, labels, boxes, actual_bboxes, file_name, page_size):
        self.guid = guid
        self.words = words
        self.labels = labels
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


class InputFeatures:
    """토큰화/패딩된 입력 특징"""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        label_ids,
        boxes,
        actual_bboxes,
        file_name,
        page_size,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size


def _split_token_label(line: str):
    """'토큰<TAB>라벨' 또는 '토큰 라벨' 파싱. 라벨 누락 시 'O'."""
    s = line.rstrip("\n").lstrip("\ufeff").strip()
    if not s:
        return None, None
    parts = s.split("\t") if "\t" in s else s.split(maxsplit=1)
    token = parts[0].strip()
    label = parts[1].strip() if len(parts) > 1 and parts[1].strip() else "O"
    return token, label


def _split_box_line(bline: str):
    """'토큰<TAB>x y w h' 파싱."""
    s = bline.rstrip("\n").lstrip("\ufeff").strip()
    if not s:
        return None, None
    parts = s.split("\t") if "\t" in s else s.split()
    token, last = parts[0].strip(), parts[-1].strip()
    try:
        box = [int(b) for b in last.split()]
        if len(box) != 4:
            raise ValueError
    except Exception:
        return token, None
    return token, box


def _split_image_line(iline: str):
    """'토큰<TAB>x1 y1 x2 y2<TAB>W H<TAB>filename' 파싱."""
    s = iline.rstrip("\n").lstrip("\ufeff").strip()
    if not s:
        return None, None, None, None
    parts = s.split("\t") if "\t" in s else s.split()
    if len(parts) < 4:
        return parts[0], None, None, None
    token, bbox_str, size_str, fname = parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()
    try:
        actual_bbox = [int(b) for b in bbox_str.split()]
        page_size = [int(i) for i in size_str.split()]
        if len(actual_bbox) != 4 or len(page_size) != 2:
            raise ValueError
    except Exception:
        return token, None, None, fname
    return token, actual_bbox, page_size, fname


def read_examples_from_file(data_dir, mode):
    """세 개의 어노테이션 파일에서 예제를 읽습니다."""
    file_path = os.path.join(data_dir, f"{mode}.txt")
    box_file_path = os.path.join(data_dir, f"{mode}_box.txt")
    image_file_path = os.path.join(data_dir, f"{mode}_image.txt")

    guid_index, examples = 1, []
    with open(file_path, encoding="utf-8") as f, \
         open(box_file_path, encoding="utf-8") as fb, \
         open(image_file_path, encoding="utf-8") as fi:

        words, boxes, actual_bboxes, labels = [], [], [], []
        file_name = page_size = None

        for lineno, (line, bline, iline) in enumerate(zip(f, fb, fi), start=1):
            if line.startswith("-DOCSTART-") or line.strip() == "":
                if words:
                    examples.append(
                        InputExample(
                            guid=f"{mode}-{guid_index}",
                            words=words, labels=labels,
                            boxes=boxes, actual_bboxes=actual_bboxes,
                            file_name=file_name, page_size=page_size,
                        )
                    )
                    guid_index += 1
                    words, boxes, actual_bboxes, labels = [], [], [], []
                    file_name = page_size = None
                continue

            tok, lbl = _split_token_label(line)
            b_tok, box = _split_box_line(bline)
            i_tok, actual_bbox, pg_size, fname = _split_image_line(iline)

            if tok is None:
                logger.warning("[L%d] Empty/invalid token line skipped", lineno)
                continue
            if b_tok and tok != b_tok:
                logger.warning("[L%d] Token mismatch (.txt vs _box.txt): %r vs %r", lineno, tok, b_tok)
            if i_tok and tok != i_tok:
                logger.warning("[L%d] Token mismatch (.txt vs _image.txt): %r vs %r", lineno, tok, i_tok)
            if box is None or actual_bbox is None or pg_size is None:
                logger.warning("[L%d] Incomplete annotation, skipped", lineno)
                continue

            words.append(tok)
            labels.append(lbl or "O")
            boxes.append(box)
            actual_bboxes.append(actual_bbox)
            page_size, file_name = pg_size, fname

        if words:
            examples.append(
                InputExample(
                    guid=f"{mode}-{guid_index}",
                    words=words, labels=labels,
                    boxes=boxes, actual_bboxes=actual_bboxes,
                    file_name=file_name, page_size=page_size,
                )
            )
    return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
    pad_token_segment_id=0,
    pad_token_label_id=-1,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """예제를 토큰/라벨/박스 텐서로 변환합니다."""
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for ex_index, example in enumerate(examples):
        file_name, page_size = example.file_name, example.page_size
        width, height = page_size

        if ex_index % 10000 == 0:
            logger.info("Writing example %d / %d", ex_index, len(examples))

        tokens, token_boxes, actual_bboxes, label_ids = [], [], [], []
        for word, label, box, actual_bbox in zip(example.words, example.labels, example.boxes, example.actual_bboxes):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            actual_bboxes.extend([actual_bbox] * len(word_tokens))
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            cut = max_seq_length - special_tokens_count
            tokens, token_boxes, actual_bboxes, label_ids = tokens[:cut], token_boxes[:cut], actual_bboxes[:cut], label_ids[:cut]

        tokens += [sep_token]
        token_boxes += [sep_token_box]
        actual_bboxes += [[0, 0, width, height]]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            tokens += [sep_token]
            token_boxes += [sep_token_box]
            actual_bboxes += [[0, 0, width, height]]
            label_ids += [pad_token_label_id]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            token_boxes += [cls_token_box]
            actual_bboxes += [[0, 0, width, height]]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_boxes = [cls_token_box] + token_boxes
            actual_bboxes = [[0, 0, width, height]] + actual_bboxes
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
            token_boxes = ([pad_token_box] * padding_length) + token_boxes
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length
            token_boxes += [pad_token_box] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example %s ***", example.guid)
            logger.info("tokens: %s", " ".join(map(str, tokens)))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                boxes=token_boxes,
                actual_bboxes=actual_bboxes,
                file_name=file_name,
                page_size=page_size,
            )
        )
    return features


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    """평가 수행: 결과 dict와 예측 시퀀스 리스트를 반환합니다.
    - 모델 실제 타입(model.config.model_type)에 따라 입력 키를 필터링합니다.
    - BERT/Roberta 등은 텍스트 키만, LayoutLM 계열은 bbox(+image/pixel_values)까지 허용합니다.
    """
    # 데이터셋/로더
    eval_dataset = SROIEDataset(args, tokenizer, labels, pad_token_label_id, mode=mode)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 실제 모델 타입을 단일 진실로 사용
    real_model_type = getattr(getattr(model, "config", None), "model_type", "")
    if not real_model_type:
        logger.warning("model.config.model_type 미검출: 기본값으로 텍스트 모델로 처리합니다.")
        real_model_type = "bert"

    def _filter_inputs_for_model(model_type: str, batch_inputs: dict) -> dict:
        """모델 타입에 맞춰 허용 키만 통과시키는 안전 가드입니다."""
        # 공통 허용
        allowed = {"input_ids", "attention_mask", "labels"}
        # token_type_ids는 존재할 때만 포함
        if batch_inputs.get("token_type_ids", None) is not None:
            allowed.add("token_type_ids")
        # LayoutLM 계열 확장
        if model_type == "layoutlm":
            allowed |= {"bbox"}
        elif model_type == "layoutlmv2":
            allowed |= {"bbox", "image"}          # v2: feature_extractor -> image
        elif model_type == "layoutlmv3":
            allowed |= {"bbox", "pixel_values"}   # v3: image_processor -> pixel_values
        # 필터링
        pruned = {k: v for k, v in batch_inputs.items() if k in allowed and v is not None}
        return pruned

    eval_loss, nb_eval_steps = 0.0, 0
    preds, out_label_ids = None, None
    model.eval()

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        # 배치 → 기본 입력 구성
        inputs = {
            "input_ids": batch[0].to(args.device),
            "attention_mask": batch[1].to(args.device),
            "labels": batch[3].to(args.device),
        }
        # token_type_ids는 있을 때만
        if len(batch) > 2 and batch[2] is not None:
            inputs["token_type_ids"] = batch[2].to(args.device)
        # LayoutLM 계열에서만 bbox 고려(배치 4번 인덱스에 있다고 가정)
        if real_model_type in ["layoutlm", "layoutlmv2", "layoutlmv3"] and len(batch) > 4 and batch[4] is not None:
            inputs["bbox"] = batch[4].to(args.device)
        # v2/v3의 이미지 텐서가 배치에 포함되는 경우(선택적)
        # 관례상 batch[5] 또는 dict 기반 collate에서 제공될 수 있으므로 존재 검사만 수행합니다.
        if real_model_type == "layoutlmv2" and len(batch) > 5 and batch[5] is not None:
            inputs["image"] = batch[5].to(args.device)
        if real_model_type == "layoutlmv3" and len(batch) > 5 and batch[5] is not None:
            inputs["pixel_values"] = batch[5].to(args.device)

        # 최종 안전 가드: 모델 타입별 허용 키만 통과
        inputs = _filter_inputs_for_model(real_model_type, inputs)

        # 첫 배치 로깅으로 가시성 확보
        if step == 0:
            logger.info("model_type=%s, input_keys=%s", real_model_type, sorted(list(inputs.keys())))

        # 추론
        with torch.no_grad():
            outputs = model(**inputs)
            tmp_eval_loss = outputs.loss
            logits = outputs.logits

        # 멀티 GPU 평균
        if args.n_gpu > 1:
            tmp_eval_loss = tmp_eval_loss.mean()

        eval_loss += float(tmp_eval_loss.item())
        nb_eval_steps += 1

        # 누적 결과
        logits_np = logits.detach().cpu().numpy()
        labels_np = inputs["labels"].detach().cpu().numpy()
        preds = logits_np if preds is None else np.append(preds, logits_np, axis=0)
        out_label_ids = labels_np if out_label_ids is None else np.append(out_label_ids, labels_np, axis=0)

    # 집계
    eval_loss = eval_loss / max(1, nb_eval_steps)
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}
    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i, j]])
                preds_list[i].append(label_map[preds[i, j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list, zero_division=0),
        "recall": recall_score(out_label_list, preds_list, zero_division=0),
        "f1": f1_score(out_label_list, preds_list, zero_division=0),
    }
    try:
        report = classification_report(out_label_list, preds_list, zero_division=0)
        logger.info("\n%s", report)
    except Exception as e:
        logger.warning("classification_report skipped: %s", e)

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list
