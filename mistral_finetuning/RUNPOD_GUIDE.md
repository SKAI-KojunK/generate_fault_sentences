# RunPod에서 Mistral-7B 파인튜닝 가이드

이 가이드는 RunPod 클라우드 GPU 플랫폼에서 Mistral-7B 모델을 파인튜닝하는 방법을 설명합니다.

## 1. RunPod 설정

### 1.1 GPU Pod 생성
1. [RunPod](https://runpod.io)에 로그인
2. "GPU Pods" 섹션으로 이동
3. 다음 사양의 Pod 선택:
   - **권장 GPU**: RTX L40 (48GB VRAM)
   - **최소 GPU**: RTX 4090 (24GB VRAM)
   - **템플릿**: PyTorch 또는 Jupyter Notebook
   - **스토리지**: 최소 50GB

### 1.2 Jupyter Lab 실행
1. Pod가 시작되면 "Connect" 버튼 클릭
2. "Jupyter Lab" 옵션 선택
3. 새 브라우저 탭에서 Jupyter Lab 열림

## 2. 프로젝트 파일 업로드

### 2.1 필수 파일 업로드
Jupyter Lab 파일 브라우저를 사용하여 다음 파일들을 업로드:

```
mistral_finetuning/
├── finetuning_notebook.ipynb    # 메인 노트북
├── config.py                   # 설정 파일
├── data_preprocessing.py        # 데이터 전처리
├── train.py                    # 파인튜닝 스크립트
├── evaluate.py                 # 평가 스크립트
├── inference.py                # 추론 스크립트
├── runpod_setup.py            # RunPod 설정 스크립트
└── generated_dataset.jsonl     # 학습 데이터 (중요!)
```

### 2.2 데이터 파일 업로드
- `generated_dataset.jsonl` 파일을 반드시 업로드
- 파일 크기: 약 2-3MB (10,000개 샘플)
- 파일 위치: `mistral_finetuning/` 디렉토리 안

## 3. 환경 설정

### 3.1 자동 설정 (권장)
터미널에서 다음 명령 실행:

```bash
cd mistral_finetuning
python runpod_setup.py
```

이 스크립트는 자동으로:
- GPU 환경 확인
- 필요한 라이브러리 설치
- 디렉토리 생성
- 데이터 전처리 실행

### 3.2 수동 설정
필요한 라이브러리만 설치:

```bash
pip install -q peft bitsandbytes trl wandb datasets accelerate
```

## 4. 파인튜닝 실행

### 4.1 Jupyter 노트북 사용 (권장)
1. `finetuning_notebook.ipynb` 파일 열기
2. 셀을 순서대로 실행
3. 각 단계별로 결과 확인

**예상 소요 시간**:
- L40 GPU: 2-3시간
- RTX 4090: 3-4시간

### 4.2 Python 스크립트 사용
터미널에서 실행:

```bash
# 데이터 전처리
python run_preprocessing.py

# 파인튜닝 실행
python run_training.py
```

## 5. 모니터링 및 관리

### 5.1 학습 진행 상황 확인
```bash
# 로그 실시간 확인
tail -f logs/trainer_state.json

# GPU 메모리 사용량 확인
nvidia-smi

# 체크포인트 확인
ls -la checkpoints/
```

### 5.2 연결 끊김 대비
RunPod 연결이 끊어져도 학습이 계속되도록:

```bash
# tmux 세션 시작
tmux new -s finetuning

# 학습 실행
python run_training.py

# 세션에서 나가기 (학습은 계속됨)
Ctrl+B, D

# 나중에 세션 재연결
tmux attach -t finetuning
```

## 6. 체크포인트 및 복구

### 6.1 중간 저장
- 500 스텝마다 자동 저장
- `checkpoints/checkpoint-{step}` 형태로 저장
- 최대 3개 체크포인트 유지

### 6.2 학습 재개
연결이 끊어진 경우:

```bash
python run_training.py --resume checkpoints/checkpoint-1500
```

## 7. 결과 다운로드

### 7.1 모델 파일 압축
```bash
# 체크포인트 폴더 압축
tar -czf mistral_finetuned_model.tar.gz checkpoints/

# 또는 ZIP 형태
zip -r mistral_finetuned_model.zip checkpoints/
```

### 7.2 파일 다운로드
1. Jupyter Lab 파일 브라우저에서 압축 파일 선택
2. 우클릭 → "Download" 선택
3. 로컬 컴퓨터에 저장

## 8. 비용 최적화 팁

### 8.1 Pod 관리
- 파인튜닝 완료 후 즉시 Pod 종료
- 필요시 스냅샷 저장하여 재사용
- 장시간 사용하지 않을 때는 "Pause" 사용

### 8.2 설정 최적화
```python
# 메모리 절약 설정
training_args = TrainingArguments(
    per_device_train_batch_size=2,  # 배치 크기 줄이기
    gradient_accumulation_steps=8,  # 그래디언트 누적 증가
    fp16=True,                      # 16비트 정밀도 사용
    dataloader_num_workers=0,       # 워커 수 줄이기
)
```

## 9. 문제 해결

### 9.1 일반적인 문제

**CUDA 메모리 부족**
```python
# config.py에서 배치 크기 줄이기
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
```

**라이브러리 설치 오류**
```bash
pip install --upgrade pip
pip install --no-cache-dir peft bitsandbytes
```

**데이터 파일 없음**
```bash
# 파일 위치 확인
find . -name "generated_dataset.jsonl"

# 파일 형식 확인
head -3 generated_dataset.jsonl
```

### 9.2 성능 확인
```python
# GPU 메모리 사용량
print(f"메모리 사용량: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 학습 속도
print(f"초당 샘플 수: {samples_per_second}")
```

## 10. 로컬 테스트 (옵션)

로컬에서 간단히 테스트하려면:

```bash
# 작은 데이터셋으로 테스트
head -100 generated_dataset.jsonl > test_data.jsonl

# 빠른 테스트 설정
python run_training.py \
    --epochs 1 \
    --batch_size 1 \
    --learning_rate 1e-3
```

## 지원 및 문의

- RunPod 공식 문서: [docs.runpod.io](https://docs.runpod.io)
- Discord 커뮤니티: RunPod 공식 Discord
- 기술적 문제: GitHub Issues