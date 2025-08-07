# Mistral-7B 파인튜닝: 유지보수 요청 파싱

이 프로젝트는 Mistral-7B-Instruct-v0.3 모델을 사용하여 유지보수 요청 문장을 파싱하는 모델을 파인튜닝하는 프로젝트입니다.

## 목표

자연어로 작성된 작업 요청 문장을 입력받아, 위치, 설비유형, 현상코드, 우선순위 정보를 정확히 식별하여 key:value 형태로 반환하는 모델을 학습합니다.

## 기술 스택

- **QLoRA**: 4비트 양자화 + LoRA를 통한 효율적인 파인튜닝
- **LoRA**: Low-Rank Adaptation으로 메모리 효율성 확보
- **Mistral-7B-Instruct-v0.3**: 기본 모델
- **PyTorch**: 딥러닝 프레임워크
- **Transformers**: Hugging Face 라이브러리
- **WandB**: 실험 추적 및 로깅

## 프로젝트 구조

```
mistral_finetuning/
├── README.md                    # 프로젝트 설명
├── requirements.txt             # 필요한 라이브러리
├── config.py                   # 설정 관리
├── data_preprocessing.py        # 데이터 전처리
├── train.py                    # 파인튜닝 메인 스크립트
├── evaluate.py                  # 모델 평가
├── inference.py                 # 추론 스크립트
├── run_preprocessing.py         # 전처리 실행 스크립트
├── run_training.py             # 파인튜닝 실행 스크립트
├── finetuning_notebook.ipynb   # Jupyter 노트북
├── data/                       # 전처리된 데이터
├── checkpoints/                # 모델 체크포인트
├── logs/                       # 로그 파일
├── results/                    # 평가 결과
└── inference_results/          # 추론 결과
```

## 설치 및 설정

### 1. 환경 설정

```bash
# 필요한 라이브러리 설치
pip install -r requirements.txt

# GPU 메모리 확인 (L40 권장)
nvidia-smi
```

### 2. 데이터 준비

```bash
# 데이터 전처리 실행
python run_preprocessing.py
```

## 사용법

### 1. 데이터 전처리

```bash
python run_preprocessing.py
```

이 스크립트는 `../output/generated_dataset.jsonl` 파일을 읽어서 파인튜닝에 적합한 형태로 변환합니다.

### 2. 파인튜닝 실행

#### 기본 실행
```bash
python run_training.py
```

#### 옵션과 함께 실행
```bash
python run_training.py \
    --epochs 5 \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --lora_r 32 \
    --lora_alpha 64
```

#### 체크포인트에서 재시작
```bash
python run_training.py --resume checkpoints/checkpoint-1000
```

### 3. 모델 평가

```bash
python evaluate.py
```

### 4. 추론 테스트

#### 대화형 모드
```bash
python inference.py --interactive
```

#### 단일 텍스트 파싱
```bash
python inference.py --text "No.2 PP의 기타기기 CCTV에서 SHE 발생. 주기작업 필요."
```

#### 배치 파싱
```bash
python inference.py --input_file test_data.jsonl --output_file results.json
```

### 5. Jupyter 노트북 사용

```bash
jupyter notebook finetuning_notebook.ipynb
```

## 설정 옵션

### 학습 설정 (`config.py`)

- **모델**: `mistralai/Mistral-7B-Instruct-v0.3`
- **학습률**: 2e-4 (기본값)
- **배치 크기**: 4 (기본값)
- **에포크**: 3 (기본값)
- **LoRA rank**: 16 (기본값)
- **LoRA alpha**: 32 (기본값)

### QLoRA 설정

- **4비트 양자화**: 활성화
- **양자화 타입**: nf4
- **더블 양자화**: 활성화

## 체크포인트 및 복구

### 체크포인트 저장

- `save_steps`: 500 스텝마다 체크포인트 저장
- `save_total_limit`: 최대 3개 체크포인트 유지
- `load_best_model_at_end`: 최고 성능 모델 자동 로드

### 학습 재시작

```bash
python run_training.py --resume checkpoints/checkpoint-1000
```

## 성능 모니터링

### WandB 통합

학습 과정을 WandB에서 실시간으로 모니터링할 수 있습니다:

```python
# WandB 프로젝트 설정
wandb.init(
    project="mistral-maintenance-parsing",
    name="qlora-finetuning"
)
```

### 로그 파일

- `logs/`: TensorBoard 로그
- `results/`: 평가 결과 및 성능 리포트

## 평가 메트릭

### 정확도 측정

- **전체 정확도**: 모든 필드가 정확한 비율
- **필드별 정확도**: 위치, 설비유형, 현상코드, 우선순위별 정확도

### 결과 파일

- `evaluation_results.json`: 상세 평가 결과
- `detailed_evaluation.csv`: 샘플별 상세 결과
- `error_analysis.json`: 오류 분석
- `performance_report.md`: 성능 리포트

## 메모리 최적화

### L40 GPU 권장 사항

- **배치 크기**: 4 (기본값)
- **그래디언트 누적**: 4 스텝
- **4비트 양자화**: 메모리 사용량 75% 감소
- **LoRA**: 학습 가능한 파라미터 1% 미만

### 메모리 사용량

- **모델 로드**: ~14GB (4비트)
- **학습 중**: ~20GB (배치 크기 4 기준)
- **추론**: ~14GB

## 문제 해결

### 일반적인 문제

1. **CUDA 메모리 부족**
   - 배치 크기 줄이기
   - 그래디언트 누적 스텝 증가
   - 4비트 양자화 확인

2. **학습이 느림**
   - 배치 크기 증가
   - 그래디언트 누적 스텝 감소
   - GPU 메모리 확인

3. **모델 성능 저하**
   - 학습률 조정
   - LoRA rank 증가
   - 데이터 품질 확인

### 로그 확인

```bash
# 실시간 로그 확인
tail -f logs/trainer_state.json

# GPU 사용량 확인
watch -n 1 nvidia-smi
```

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 기여

버그 리포트나 기능 제안은 이슈를 통해 제출해 주세요.

## 참고 자료

- [Mistral AI](https://mistral.ai/)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) 