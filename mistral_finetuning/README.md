# Mistral-7B Fine-tuning Project

이 프로젝트는 Mistral-7B 모델을 사용하여 유지보수 데이터 파싱을 위한 파인튜닝을 수행합니다.

## 🚀 빠른 시작

**📖 최신 가이드**: [RUNPOD_QUICKSTART.md](../RUNPOD_QUICKSTART.md) - **가장 간단한 실행 방법**

### RunPod에서 실행 (권장)

```bash
# 1. 프로젝트 업로드 후
cd /workspace/generate_fault_sentences

# 2. HF 토큰 설정 (.env 파일)
cp .env.example .env
# .env 파일에 실제 토큰 입력

# 3. 원샷 설정 (설치+전처리)
python mistral_finetuning/runpod_setup.py

# 4. 파인튜닝 실행
python mistral_finetuning/run_training.py --epochs 3 --batch_size 4
```

자세한 내용은 [RUNPOD_QUICKSTART.md](../RUNPOD_QUICKSTART.md)를 참조하세요.

### 로컬에서 실행

1. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

2. **데이터 준비**
   ```bash
   python data_preprocessing.py
   ```

3. **파인튜닝 실행**
   ```bash
   python train.py
   ```

## 📁 프로젝트 구조

```
mistral_finetuning/
├── train.py                    # 파인튜닝 메인 스크립트
├── config.py                   # 설정 파일 (RunPod 최적화 포함)
├── data_preprocessing.py       # 데이터 전처리
├── evaluate.py                 # 평가 스크립트
├── inference.py                # 추론 스크립트
├── runpod_setup.py            # RunPod 환경 설정
├── requirements.txt            # 의존성 목록
├── finetuning_notebook.ipynb   # Jupyter 노트북
├── RUNPOD_GUIDE.md            # RunPod 상세 가이드
└── generated_dataset.jsonl     # 학습 데이터
```

## ⚙️ 설정

### 모델 설정
- **기본 모델**: `mistralai/Mistral-7B-v0.1` (공개 모델)
- **파인튜닝 방법**: LoRA (Low-Rank Adaptation)
- **양자화**: 4비트 QLoRA

### 학습 설정 (RunPod 최적화)
- **배치 크기**: 2 (메모리 최적화)
- **그래디언트 누적**: 8 (실제 배치 크기 16 유지)
- **시퀀스 길이**: 1024 (실제 데이터 최대 39 토큰)
- **학습률**: 2e-4
- **에포크**: 3

### 메모리 최적화 설정
- **그래디언트 체크포인팅**: 활성화
- **데이터로더 워커**: 0 (메모리 절약)
- **시퀀스 길이**: 1024 (실제 데이터 대비 26배 여유)

## 🔧 주요 기능

### 1. 자동 환경 설정
```bash
python runpod_setup.py
```
- GPU 환경 확인
- 필요한 라이브러리 설치
- Hugging Face 인증 설정
- 디렉토리 생성
- 데이터 전처리

### 2. 파인튜닝 실행
```bash
python train.py
```
- LoRA 설정
- 데이터셋 로딩
- 학습 실행
- 체크포인트 저장

### 3. 모델 평가
```bash
python evaluate.py --model_path checkpoints/best
```

### 4. 추론 테스트
```bash
python inference.py --model_path checkpoints/best
```

## 📊 모니터링

### 실시간 모니터링
```bash
# GPU 사용량
watch -n 1 nvidia-smi

# 학습 로그
tail -f training.log

# 체크포인트 확인
watch -n 10 ls -la checkpoints/
```

### tmux 세션 관리
```bash
# 세션 시작
tmux new -s finetuning

# 세션 재연결
tmux attach -t finetuning

# 세션 목록
tmux list-sessions
```

## 🛠️ 문제 해결

### 일반적인 문제들

1. **CUDA 메모리 부족**
   ```bash
   # 환경 변수 설정
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   export TOKENIZERS_PARALLELISM=false
   
   # 메모리 정리
   python -c "import torch; torch.cuda.empty_cache()"
   ```

2. **Hugging Face 인증 오류**
   ```bash
   # 공개 모델 사용 (권장)
   # 또는 토큰 설정
   export HUGGING_FACE_HUB_TOKEN=your_token
   ```

3. **토크나이저 경고**
   ```bash
   export TOKENIZERS_PARALLELISM=false
   ```

4. **라이브러리 설치 오류**
   ```bash
   pip install --upgrade pip
   pip install --no-cache-dir -r requirements.txt
   ```

## 📈 성능 최적화

### GPU 메모리 최적화
- 4비트 양자화 (QLoRA)
- 그래디언트 체크포인팅
- 배치 크기 조정 (2)
- 시퀀스 길이 최적화 (1024)

### 학습 속도 최적화
- LoRA 파인미터: 0.5754%만 훈련
- 그래디언트 누적 (8)
- 메모리 효율적인 설정

## 💰 비용 최적화 (RunPod)

1. **학습 완료 후 즉시 Pod 종료**
2. **필요시 스냅샷 저장**
3. **적절한 GPU 선택** (L40 vs 4090)

## 📋 체크리스트

### 사전 준비
- [ ] RunPod 계정 생성
- [ ] GPU Pod 생성
- [ ] 프로젝트 파일 준비
- [ ] 데이터 파일 준비

### 환경 설정
- [ ] 파일 업로드 완료
- [ ] `python runpod_setup.py` 실행
- [ ] GPU 환경 확인
- [ ] 라이브러리 설치 완료

### 파인튜닝 실행
- [ ] tmux 세션 시작
- [ ] `python train.py` 실행
- [ ] GPU 사용률 확인 (80-90%)
- [ ] 체크포인트 생성 확인

## 📞 지원

- **RunPod 가이드**: [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md)
- **문제 해결**: 이 README의 문제 해결 섹션
- **GitHub Issues**: 프로젝트 저장소

## ⏱️ 예상 소요 시간

- **환경 설정**: 10-15분
- **모델 다운로드**: 5-10분
- **파인튜닝**: 2-3시간 (L40), 3-4시간 (4090)
- **결과 처리**: 5-10분

## 🔧 RunPod 최적화 사항

### 메모리 최적화
- **배치 크기**: 4 → 2 (50% 메모리 절약)
- **그래디언트 누적**: 4 → 8 (실제 배치 크기 16 유지)
- **시퀀스 길이**: 2048 → 1024 (실제 데이터 최대 39 토큰)
- **워커 수**: 4 → 0 (메모리 절약)
- **그래디언트 체크포인팅**: 활성화

### 호환성 수정
- **evaluation_strategy** → **eval_strategy**
- **공개 모델 사용**: 인증 불필요
- **환경 변수 설정**: 메모리 단편화 방지

---

**이 프로젝트는 RunPod에서 안정적으로 실행되도록 최적화되었습니다!** 🚀 