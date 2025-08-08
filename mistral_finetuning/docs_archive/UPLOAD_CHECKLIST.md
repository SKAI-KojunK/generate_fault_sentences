# RunPod 업로드 체크리스트

이 체크리스트를 따라 RunPod에 업로드할 파일들을 준비하세요.

## 📁 필수 파일 목록

### ✅ 핵심 스크립트
- [ ] `train.py` - 파인튜닝 메인 스크립트
- [ ] `config.py` - 설정 파일 (수정됨: eval_strategy, 공개 모델, 메모리 최적화)
- [ ] `data_preprocessing.py` - 데이터 전처리
- [ ] `evaluate.py` - 평가 스크립트
- [ ] `inference.py` - 추론 스크립트
- [ ] `runpod_setup.py` - RunPod 환경 설정 (업데이트됨)

### ✅ 설정 및 의존성
- [ ] `requirements.txt` - 의존성 목록 (업데이트됨)
- [ ] `finetuning_notebook.ipynb` - Jupyter 노트북

### ✅ 데이터 파일
- [ ] `generated_dataset.jsonl` - 학습 데이터 (중요!)

### ✅ 문서
- [ ] `README.md` - 프로젝트 설명 (업데이트됨)
- [ ] `RUNPOD_GUIDE.md` - RunPod 상세 가이드 (업데이트됨)
- [ ] `UPLOAD_CHECKLIST.md` - 이 파일

## 🔧 수정된 내용 확인

### 1. config.py 수정사항
- [ ] `evaluation_strategy` → `eval_strategy` 변경
- [ ] `model_name` → `"mistralai/Mistral-7B-v0.1"` (공개 모델)
- [ ] `per_device_train_batch_size` → `2` (메모리 최적화)
- [ ] `per_device_eval_batch_size` → `2` (메모리 최적화)
- [ ] `gradient_accumulation_steps` → `8` (실제 배치 크기 16 유지)
- [ ] `max_seq_length` → `1024` (실제 데이터 최대 39 토큰)
- [ ] `dataloader_num_workers` → `0` (메모리 절약)

### 2. train.py 수정사항
- [ ] `evaluation_strategy` → `eval_strategy` 변경
- [ ] `gradient_checkpointing=True` 추가 (메모리 최적화)

### 3. requirements.txt 수정사항
- [ ] 모든 의존성 포함
- [ ] 주석 추가로 가독성 향상

### 4. runpod_setup.py 수정사항
- [ ] 시스템 의존성 설치 (tmux)
- [ ] Hugging Face 인증 설정
- [ ] 환경 변수 설정
- [ ] 모든 라이브러리 설치

## 📦 업로드 준비

### 1. 파일 구조 확인
```
mistral_finetuning/
├── train.py
├── config.py
├── data_preprocessing.py
├── evaluate.py
├── inference.py
├── runpod_setup.py
├── requirements.txt
├── finetuning_notebook.ipynb
├── README.md
├── RUNPOD_GUIDE.md
├── UPLOAD_CHECKLIST.md
└── generated_dataset.jsonl
```

### 2. 파일 크기 확인
- [ ] `generated_dataset.jsonl`: 약 3MB (10,000개 샘플)
- [ ] 전체 프로젝트: 약 5-10MB

### 3. 압축 준비
```bash
# 로컬에서 압축
tar -czf mistral_finetuning.tar.gz mistral_finetuning/

# 또는 ZIP
zip -r mistral_finetuning.zip mistral_finetuning/
```

## 🚀 RunPod 실행 순서

### 1. Pod 생성
- [ ] GPU: RTX L40 (48GB VRAM) 또는 RTX 4090 (24GB VRAM)
- [ ] 템플릿: PyTorch 2.0
- [ ] 스토리지: 50GB

### 2. 파일 업로드
- [ ] Jupyter Lab 파일 브라우저 사용
- [ ] 또는 Git 클론
- [ ] 또는 직접 업로드

### 3. 환경 설정
```bash
cd mistral_finetuning
python runpod_setup.py
```

### 4. 메모리 최적화 설정
```bash
# 환경 변수 설정
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# 메모리 정리
python -c "import torch; torch.cuda.empty_cache()"
```

### 5. 파인튜닝 실행
```bash
# tmux 세션에서 실행
tmux new -s finetuning
python train.py
```

### 6. 모니터링
```bash
# 새 터미널에서
watch -n 1 nvidia-smi
tail -f training.log
```

## ⚠️ 주의사항

### 1. 호환성 문제 해결
- [ ] `evaluation_strategy` → `eval_strategy` 변경 완료
- [ ] 공개 모델 사용으로 인증 문제 해결
- [ ] 모든 의존성 포함

### 2. 메모리 최적화
- [ ] `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정
- [ ] `TOKENIZERS_PARALLELISM=false` 설정
- [ ] 배치 크기 2, 그래디언트 누적 8 설정
- [ ] 시퀀스 길이 1024 설정 (실제 데이터 최대 39 토큰)
- [ ] 그래디언트 체크포인팅 활성화

### 3. 환경 설정
- [ ] tmux 설치
- [ ] Hugging Face 인증 설정

### 4. 모니터링
- [ ] GPU 사용률 확인 (80-90%)
- [ ] 체크포인트 생성 확인
- [ ] 로그 실시간 확인

## 📋 최종 확인

### 업로드 전
- [ ] 모든 파일이 포함되었는지 확인
- [ ] 수정사항이 반영되었는지 확인
- [ ] 파일 크기가 적절한지 확인

### RunPod에서
- [ ] 파일 업로드 완료
- [ ] 환경 설정 성공
- [ ] 메모리 최적화 설정 완료
- [ ] 파인튜닝 시작 성공
- [ ] GPU 사용률 정상

## 🎯 성공 지표

### 환경 설정 성공
- ✅ GPU 환경 확인
- ✅ 라이브러리 설치 완료
- ✅ 데이터 전처리 완료
- ✅ 메모리 최적화 설정 완료

### 파인튜닝 성공
- ✅ 모델 로딩 완료
- ✅ LoRA 설정 완료
- ✅ 학습 시작
- ✅ GPU 사용률 80-90%
- ✅ 메모리 부족 없음

### 완료 지표
- ✅ 체크포인트 생성
- ✅ 손실값 감소
- ✅ 학습 완료

## 🔧 메모리 최적화 설정 확인

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

### 예상 메모리 사용량
- **기존**: 26GB (메모리 부족)
- **최적화 후**: 20-22GB (안전한 범위)

---

**이 체크리스트를 따라하면 RunPod에서 안정적으로 파인튜닝을 완료할 수 있습니다!** 🚀