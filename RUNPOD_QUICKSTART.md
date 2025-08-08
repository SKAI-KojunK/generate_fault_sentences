# RunPod 파인튜닝 완전 가이드

## 📋 준비사항
- RunPod GPU Pod (RTX L40 48GB 또는 RTX 4090 24GB 권장)
- PyTorch 이미지 (CUDA 12.x 계열)
- Hugging Face 토큰 (Mistral 모델 접근용)

## 🚀 빠른 시작 (5단계)

### 1️⃣ 프로젝트 업로드

**방법 1: Git 클론 (권장)**
```bash
cd /workspace
git clone <YOUR_REPO_URL> generate_fault_sentences
cd generate_fault_sentences
```

**방법 2: 직접 업로드**
- 프로젝트 폴더 전체를 RunPod의 `/workspace/`에 업로드
- 특히 `output/generated_dataset.jsonl` 파일이 포함되어야 함

### 2️⃣ HF 토큰 설정

```bash
cd /workspace/generate_fault_sentences

# .env 파일 생성 및 토큰 입력
cp .env.example .env
nano .env  # 또는 vi .env

# 내용 예시:
# HUGGING_FACE_HUB_TOKEN=hf_your_actual_token_here
```

### 3️⃣ 환경 설정 및 전처리 (원샷)

```bash
# 설치 + GPU 확인 + 전처리 모두 자동 실행
python mistral_finetuning/runpod_setup.py
```

**자동 수행 작업:**
- ✅ .env 파일에서 HF 토큰 자동 로드
- ✅ 시스템 의존성 설치 (tmux 등)
- ✅ GPU 확인 및 메모리 정보 출력
- ✅ requirements.txt 기반 패키지 설치
- ✅ HF 토큰으로 비대화형 로그인
- ✅ 디렉토리 생성 (`data/`, `checkpoints/`, `logs/`, `results/`)
- ✅ `output/generated_dataset.jsonl` 탐지 및 train/validation/test 분할

### 4️⃣ 파인튜닝 실행

**옵션 1: CLI 인자로 커스터마이즈 (권장)**
```bash
# 백그라운드 실행 (세션 유지)
tmux new -s finetuning

python mistral_finetuning/run_training.py \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32
```

**옵션 2: 기본 설정으로 빠른 실행**
```bash
python mistral_finetuning/train.py
```

**옵션 3: nohup 백그라운드**
```bash
nohup python mistral_finetuning/run_training.py \
  --epochs 3 --batch_size 4 --learning_rate 2e-4 \
  > training.log 2>&1 &

# 로그 실시간 확인
tail -f training.log
```

### 5️⃣ 모니터링 및 평가

**실시간 모니터링**
```bash
# GPU 사용률 확인
watch -n 2 nvidia-smi

# 체크포인트 확인
ls -la checkpoints/

# 로그 확인
ls -la logs/
```

**학습 완료 후 평가**
```bash
python mistral_finetuning/evaluate.py

# 결과 확인
cat results/performance_report.json
```

## 🔄 고급 사용법

### 체크포인트 재시작
```bash
python mistral_finetuning/run_training.py \
  --resume checkpoints/checkpoint-1500 \
  --epochs 5
```

### WandB 모니터링 (선택)
```bash
# .env 파일에 추가
echo "WANDB_API_KEY=your_wandb_key" >> .env

# config.py에서 report_to="wandb"로 변경하거나
# 환경변수로 설정
export WANDB_PROJECT=mistral-finetuning
```

## 📂 출력 파일들

**학습 결과:**
- `checkpoints/`: 모델 체크포인트, LoRA 어댑터
- `logs/`: 학습 로그, TensorBoard 이벤트
- `checkpoints/model_info.json`: 모델 메타데이터

**평가 결과:**
- `results/evaluation_results.json`: 전체 평가 메트릭
- `results/detailed_evaluation.csv`: 샘플별 상세 결과
- `results/performance_report.json`: 성능 요약
- `results/error_analysis.json`: 오류 분석

## 🎯 전체 실행 예시

```bash
# 1. 환경 설정
cd /workspace/generate_fault_sentences
cp .env.example .env
# .env 파일에 실제 토큰 입력

# 2. 원샷 설정
python mistral_finetuning/runpod_setup.py

# 3. 파인튜닝 (tmux 세션)
tmux new -s finetuning
python mistral_finetuning/run_training.py --epochs 3 --batch_size 4

# 4. 모니터링 (다른 터미널)
watch -n 2 nvidia-smi

# 5. 평가
python mistral_finetuning/evaluate.py
```

## 🔧 문제 해결

### GPU 메모리 부족
```bash
# 배치 크기 줄이기
python mistral_finetuning/run_training.py --batch_size 2

# 또는 config.py에서 per_device_train_batch_size 조정
```

### 모델 다운로드 실패
- `.env` 파일의 HF 토큰 확인
- 토큰 권한 확인 (Mistral 모델 접근 가능한지)

### 데이터 파일 없음
- `output/generated_dataset.jsonl` 파일 존재 확인
- 파일 크기 확인 (`ls -la output/`)

---

> **💡 팁**: HF 토큰만 설정하면 나머지는 모두 자동화되어 있습니다!