# RunPod에서 Mistral-7B 파인튜닝 완전 가이드

이 가이드는 RunPod 클라우드 GPU 플랫폼에서 Mistral-7B 모델을 파인튜닝하는 방법을 단계별로 설명합니다.

## 📋 사전 준비

### 1.1 RunPod 계정 및 GPU Pod
1. [RunPod](https://runpod.io)에 로그인
2. "GPU Pods" 섹션으로 이동
3. **권장 사양**:
   - **GPU**: RTX L40 (48GB VRAM) 또는 RTX 4090 (24GB VRAM)
   - **템플릿**: PyTorch 또는 Jupyter Notebook
   - **스토리지**: 최소 50GB
   - **RAM**: 최소 32GB

### 1.2 프로젝트 파일 준비
다음 파일들을 준비하여 업로드:
```
mistral_finetuning/
├── train.py                    # 파인튜닝 메인 스크립트
├── config.py                   # 설정 파일 (메모리 최적화 포함)
├── data_preprocessing.py       # 데이터 전처리
├── evaluate.py                 # 평가 스크립트
├── inference.py                # 추론 스크립트
├── runpod_setup.py            # RunPod 설정 스크립트
├── requirements.txt            # 의존성 목록
├── finetuning_notebook.ipynb   # Jupyter 노트북
└── generated_dataset.jsonl     # 학습 데이터 (중요!)
```

## 🚀 1단계: RunPod Pod 생성 및 접속

### 1.1 Pod 생성
1. RunPod 대시보드에서 "GPU Pods" 클릭
2. **권장 설정**:
   - GPU: RTX L40 (48GB VRAM)
   - 템플릿: PyTorch 2.0
   - 스토리지: 50GB
3. "Deploy" 클릭하여 Pod 생성

### 1.2 Pod 접속
1. Pod가 "Running" 상태가 되면 "Connect" 클릭
2. **터미널 접속** (권장):
   - "Terminal" 탭 선택
   - SSH 또는 웹 터미널 사용

## 📁 2단계: 프로젝트 파일 업로드

### 2.1 파일 업로드 방법
**방법 1: Jupyter Lab 파일 브라우저**
1. "Connect" → "Jupyter Lab" 선택
2. 파일 브라우저에서 `mistral_finetuning/` 폴더 업로드

**방법 2: Git 클론**
```bash
git clone <your-repository-url>
cd mistral_finetuning
```

**방법 3: 직접 업로드**
```bash
# 로컬에서 파일 압축
tar -czf mistral_finetuning.tar.gz mistral_finetuning/

# RunPod에서 다운로드
wget <your-file-url>
tar -xzf mistral_finetuning.tar.gz
```

### 2.2 필수 파일 확인
```bash
cd mistral_finetuning
ls -la

# 다음 파일들이 있어야 함:
# - train.py
# - config.py
# - runpod_setup.py
# - generated_dataset.jsonl
```

## ⚙️ 3단계: 환경 설정

### 3.1 자동 설정 (권장)
```bash
cd mistral_finetuning
python runpod_setup.py
```

이 스크립트는 자동으로:
- ✅ 시스템 의존성 설치 (tmux 등)
- ✅ GPU 환경 확인
- ✅ 모든 필요한 라이브러리 설치
- ✅ Hugging Face 인증 설정
- ✅ 디렉토리 생성
- ✅ 데이터 전처리 실행

### 3.2 수동 설정 (필요시)
```bash
# 시스템 의존성
apt update && apt install -y tmux

# Python 라이브러리
pip install -r requirements.txt

# 환경 변수 설정
export TOKENIZERS_PARALLELISM=false
```

## 🔐 4단계: Hugging Face 인증 (필요시)

### 4.1 토큰 기반 인증
```bash
# Hugging Face 토큰으로 로그인
huggingface-cli login --token YOUR_TOKEN

# 또는 대화형 로그인
huggingface-cli login
```

### 4.2 환경 변수 설정
```bash
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

### 4.3 공개 모델 사용 (권장)
현재 설정은 공개 모델 `mistralai/Mistral-7B-v0.1`을 사용하므로 인증이 필요하지 않습니다.

## 🎯 5단계: 파인튜닝 실행

### 5.1 tmux 세션에서 실행 (권장)
```bash
# tmux 세션 시작
tmux new -s finetuning

# 파인튜닝 실행
python train.py

# 세션에서 나가기 (학습은 계속됨)
# Ctrl+B, D
```

### 5.2 백그라운드 실행
```bash
# 백그라운드에서 실행
nohup python train.py > training.log 2>&1 &

# 프로세스 확인
ps aux | grep python

# 로그 실시간 확인
tail -f training.log
```

### 5.3 Jupyter 노트북 실행 (선택사항)
```bash
# Jupyter Lab 실행
jupyter lab --allow-root --no-browser --port=8888

# 또는 Jupyter Notebook
jupyter notebook --allow-root --no-browser --port=8888
```

## 📊 6단계: 학습 모니터링

### 6.1 실시간 모니터링
```bash
# GPU 사용량 확인
watch -n 1 nvidia-smi

# 프로세스 확인
ps aux | grep python

# 체크포인트 확인
watch -n 10 ls -la checkpoints/

# 로그 확인
tail -f training.log
```

### 6.2 tmux 세션 관리
```bash
# 세션 목록 확인
tmux list-sessions

# 세션 재연결
tmux attach -t finetuning

# 새 세션에서 모니터링
tmux new -s monitoring
```

### 6.3 학습 진행 상황 확인
```bash
# 학습 로그 확인
tail -f logs/trainer_state.json

# 체크포인트 크기 확인
du -sh checkpoints/*

# GPU 메모리 사용량
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 🔄 7단계: 학습 중단 및 재개

### 7.1 안전한 중단
```bash
# tmux 세션에서 Ctrl+C
# 또는 프로세스 종료
pkill -f "python train.py"
```

### 7.2 체크포인트에서 재개
```bash
# 가장 최근 체크포인트 확인
ls -la checkpoints/

# 체크포인트에서 재시작
python train.py --resume checkpoints/checkpoint-1000
```

### 7.3 연결 끊김 대비
```bash
# tmux 세션 사용 (권장)
tmux new -s finetuning
python train.py

# 또는 nohup 사용
nohup python train.py > training.log 2>&1 &
```

## 📈 8단계: 학습 완료 및 결과 확인

### 8.1 학습 완료 확인
```bash
# 최종 체크포인트 확인
ls -la checkpoints/

# 학습 로그 확인
tail -20 training.log

# GPU 사용량 확인 (0%로 떨어짐)
nvidia-smi
```

### 8.2 모델 파일 압축
```bash
# 체크포인트 압축
tar -czf mistral_finetuned_model.tar.gz checkpoints/

# 또는 ZIP 형태
zip -r mistral_finetuned_model.zip checkpoints/
```

### 8.3 결과 다운로드
1. Jupyter Lab 파일 브라우저에서 압축 파일 선택
2. 우클릭 → "Download" 선택
3. 로컬 컴퓨터에 저장

## 🧪 9단계: 모델 테스트

### 9.1 간단한 추론 테스트
```bash
# 추론 스크립트 실행
python inference.py --model_path checkpoints/best

# 또는 간단한 테스트
python simple_test.py
```

### 9.2 성능 평가
```bash
# 평가 스크립트 실행
python evaluate.py --model_path checkpoints/best
```

## 💰 10단계: 비용 최적화

### 10.1 Pod 관리
- **학습 완료 후 즉시 Pod 종료**
- **필요시 스냅샷 저장**하여 재사용
- **장시간 사용하지 않을 때는 "Pause"** 사용

### 10.2 설정 최적화
```python
# 메모리 절약 설정 (config.py에서)
per_device_train_batch_size = 2  # 배치 크기 줄이기
gradient_accumulation_steps = 8  # 그래디언트 누적 증가
dataloader_num_workers = 0       # 워커 수 줄이기
```

## 🛠️ 11단계: 문제 해결

### 11.1 일반적인 문제들

**CUDA 메모리 부족**
```bash
# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# 메모리 정리
python -c "import torch; torch.cuda.empty_cache()"

# 배치 크기 줄이기 (config.py에서)
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
```

**Hugging Face 인증 오류**
```bash
# 공개 모델 사용 (권장)
# config.py에서 model_name을 "mistralai/Mistral-7B-v0.1"로 설정

# 또는 토큰 설정
export HUGGING_FACE_HUB_TOKEN=your_token
```

**토크나이저 경고**
```bash
# 환경 변수 설정
export TOKENIZERS_PARALLELISM=false
```

**연결 끊김**
```bash
# tmux 세션 사용
tmux new -s finetuning
python train.py

# 세션 재연결
tmux attach -t finetuning
```

### 11.2 성능 확인
```bash
# GPU 메모리 사용량
nvidia-smi

# 학습 속도 확인
tail -f training.log | grep "it/s"

# 체크포인트 생성 확인
ls -la checkpoints/
```

### 11.3 메모리 최적화 팁
```bash
# 메모리 단편화 방지
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 토크나이저 병렬화 비활성화
export TOKENIZERS_PARALLELISM=false

# 메모리 캐시 정리
python -c "import torch; torch.cuda.empty_cache()"
```

## 📋 12단계: 체크리스트

### 12.1 사전 준비
- [ ] RunPod 계정 생성
- [ ] GPU Pod 생성 (L40 또는 4090)
- [ ] 프로젝트 파일 준비
- [ ] 데이터 파일 준비

### 12.2 환경 설정
- [ ] 파일 업로드 완료
- [ ] `python runpod_setup.py` 실행
- [ ] GPU 환경 확인
- [ ] 라이브러리 설치 완료

### 12.3 파인튜닝 실행
- [ ] tmux 세션 시작
- [ ] `python train.py` 실행
- [ ] GPU 사용률 확인 (80-90%)
- [ ] 체크포인트 생성 확인

### 12.4 모니터링
- [ ] 실시간 로그 확인
- [ ] GPU 사용량 모니터링
- [ ] 체크포인트 백업
- [ ] 학습 완료 확인

### 12.5 결과 처리
- [ ] 모델 파일 압축
- [ ] 결과 다운로드
- [ ] Pod 종료
- [ ] 비용 확인

## 📞 지원 및 문의

- **RunPod 공식 문서**: [docs.runpod.io](https://docs.runpod.io)
- **Discord 커뮤니티**: RunPod 공식 Discord
- **GitHub Issues**: 프로젝트 저장소
- **기술적 문제**: 이 가이드의 문제 해결 섹션 참조

## ⏱️ 예상 소요 시간

- **환경 설정**: 10-15분
- **모델 다운로드**: 5-10분
- **파인튜닝**: 2-3시간 (L40), 3-4시간 (4090)
- **결과 처리**: 5-10분

## 💡 팁

1. **연결 안정성**: tmux 세션 사용으로 연결 끊김 방지
2. **모니터링**: 새 터미널에서 실시간 모니터링
3. **백업**: 정기적으로 체크포인트 백업
4. **비용**: 학습 완료 후 즉시 Pod 종료
5. **성능**: GPU 사용률 80-90%가 정상

## 🔧 메모리 최적화 설정

### 현재 최적화된 설정
- **배치 크기**: 2 (메모리 절약)
- **그래디언트 누적**: 8 (실제 배치 크기 16 유지)
- **시퀀스 길이**: 1024 (실제 데이터 최대 39 토큰)
- **워커 수**: 0 (메모리 절약)
- **그래디언트 체크포인팅**: 활성화

### 환경 변수 설정
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
```

---

**이 가이드를 따라하면 RunPod에서 안정적으로 Mistral-7B 파인튜닝을 완료할 수 있습니다!** 🚀