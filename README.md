# SROIE 영수증 개체명 인식 (NER) - LayoutLM 기반

## 1. 요약 (Summary)  
이 프로젝트는 SROIE(Scanned Receipts OCR and Information Extraction) 데이터셋을 활용하여 스캔된 영수증에서 개체명을 인식하는 작업을 수행합니다.  
기본 제공된 베이스라인 코드를 최신 Hugging Face `transformers` 및 PyTorch 버전에 맞게 수정하였으며, 자동 혼합 정밀도(AMP) 학습을 적용하여 학습 속도를 개선하였습니다.  
실험은 원본 데이터 처리 로직을 유지하며, 하이퍼파라미터 조정 및 모델 학습 효율성 개선에 집중하였습니다.  
최종 보고서는 성능 지표뿐 아니라, 모델의 한계와 이를 개선할 수 있는 다양한 접근 방안을 중점적으로 다룹니다.  
이 보고서에서는 모델 결과 분석과 향후 발전 가능성을 함께 제시합니다.  

---

## 2. 실험 결과 (Experimental Results)

| 실험 ID  | 모델명              | Epochs | Batch Size | Learning Rate | F1 점수 (Dev) | F1 점수 (Test) | 비고                  |
|----------|--------------------|--------|------------|---------------|---------------|----------------|-----------------------|
| exp01    | layoutlm-base      | 3      | 8          | 5e-5          | 94.2          | 93.7           | 베이스라인 설정       |
| exp02    | layoutlm-base      | 5      | 8          | 3e-5          | 94.5          | 93.8           | Epoch 증가 + LR 감소 |
| exp03    | layoutlm-base      | 3      | 8          | 5e-5          | 94.1          | 93.5           | FP16 적용, 속도 향상 |

---

## 3. 실행 방법 (Instructions)

### 환경 구성
- Python: 3.10.6
- CUDA: 12.6

### 라이브러리 설치

Conda 환경 생성 및 활성화
```
conda create -n sroie_ner python=3.9 -y
conda activate sroie_ner
requirements.txt 설치
```

### 학습 실행
```
python train.py \
    --model_name_or_path "microsoft/layoutlm-base-uncased" \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size 4 \
    --num_train_epochs 10 \
    --output_dir "results/train_v1.0" \
    --data_dir "../data" \
    --overwrite_output_dir
```

### 테스트 실행
```
python inference.py \
    --model_type layoutlm \
    --model_name_or_path mp-02/layoutlmv3-base-sroie \
    --model_dir results/layoutlmv3_v2.0 \
    --mode op_test \
    --do_predict \
    --overwrite_output_dir \
    --output_dir results/infer_lvm_op_v2.0
```

### 평가 실행
```
python train.py \
  --model_name_or_path microsoft/layoutlm-base-uncased \
  --output_dir ./output \
  --do_train --do_eval \
  --evaluate_during_training \
  --num_train_epochs 3 \
  --per_gpu_train_batch_size 8 \
  --learning_rate 5e-5 \
  --fp16
```

## 4. 접근 방법 (Approach)
### 4.1 데이터 탐색 (EDA)
- OCR로 추출된 Bounding Box 좌표 분포를 확인하여 문서 내 텍스트 배치 특성을 파악함.
- 개체명 라벨 분포를 분석한 결과, O 태그(비개체)가 전체 토큰의 대다수를 차지하여 심한 클래스 불균형이 존재함.
- 금액, 날짜, 전화번호 등 숫자 기반 필드에서 OCR 오인식과 잘못된 좌표 매핑이 빈번하게 발생함.
- 일부 영수증은 폰트 크기·정렬이 다른 항목이 섞여 있어 레이아웃 정보 해석에 혼선을 줌.

### 4.2 모델 및 학습 전략
- 모델 구조: 문서 레이아웃과 텍스트를 동시에 처리할 수 있는 LayoutLM 기반 토큰 분류 모델.
- 학습 최적화: AMP(FP16) 적용으로 GPU 메모리 사용량을 줄이고 학습 속도를 향상.
- 스케줄러: Linear Learning Rate Scheduler와 Warmup 단계 적용, 초반 학습 안정성 확보.
- Batch 처리: Gradient Accumulation을 사용하여 GPU 메모리 한계 내에서 더 큰 효과적 Batch Size로 학습.
- 다중 GPU 학습 지원: torch.nn.DataParallel 및 분산 학습(DistributedDataParallel) 구조 호환.

### 4.3 설계 근거
- LayoutLM은 OCR 텍스트 토큰과 위치(Bounding Box) 정보를 함께 입력받아 문서 구조적 문맥을 이해할 수 있음.
- FP16은 동일 하드웨어 자원으로 더 많은 Epoch 또는 하이퍼파라미터 조합 실험 가능.
- Warmup 스케줄링은 학습 초기의 급격한 손실 변화를 완화해 안정적인 수렴을 유도.

### 4.4 모델 한계
- OCR 품질이 낮거나 좌표가 잘못 추출된 경우, 모델이 잘못된 공간 정보를 학습하여 성능이 저하됨.
- Bounding Box 좌표 표준화 미흡으로, 다른 해상도의 영수증 이미지에서 레이아웃 이해도가 떨어짐.
- Epoch 수를 늘리면 특정 엔티티(예: 상호명, 주소)에 과적합이 발생하는 경향이 있음.
- 클래스 불균형으로 인해 드물게 등장하는 엔티티 라벨 인식률이 낮음.

### 4.5 향후 개선 방향 (Future Work)
- 데이터 증강: OCR 결과에 랜덤 노이즈 추가, Bounding Box 위치 변형, 배경/해상도 변화 등 실험.
- 모델 업그레이드: LayoutLMv2, LayoutXLM 등 시각·언어 융합 구조가 강화된 최신 모델 적용.
- 외부 데이터 활용: 영수증, 송장, 전표 등 유사 문서 데이터셋을 통한 추가 사전 학습.
- 후처리 규칙 적용: 금액·날짜 등 패턴 기반 오류 교정(Post-processing).
- 오류 분석 기반 튜닝: Confusion Matrix, 잘못 예측된 케이스의 공통 패턴 분석 후, 맞춤형 전처리 설계.

제출자 정보
이름: [권효은]

이메일: [fhkwonb@gmail.com]

제출일: 2025-08-12

