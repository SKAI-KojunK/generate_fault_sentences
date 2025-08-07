# RunPod 업로드 체크리스트

## 📦 업로드할 파일들

### 필수 파일 ✅
- [ ] `mistral_finetuning/` 전체 폴더
- [ ] `generated_dataset.jsonl` (3.12MB)

### 업로드 확인 ✅
```bash
# RunPod에서 확인
ls -la
# 다음 파일들이 있어야 함:
# - mistral_finetuning/
# - generated_dataset.jsonl
```

## 🚀 실행 순서

### 1단계: 환경 설정
```bash
cd mistral_finetuning
python runpod_setup.py
```

### 2단계: 파인튜닝 (2가지 방법)

**방법 A: Jupyter 노트북 (권장)**
```bash
jupyter notebook finetuning_notebook.ipynb
```

**방법 B: Python 스크립트**
```bash
python run_training.py
```

## ⏱️ 예상 소요 시간
- **L40**: 2-3시간
- **RTX 4090**: 3-4시간

## 🔍 모니터링
```bash
# GPU 사용량 확인
nvidia-smi

# 로그 확인
tail -f logs/trainer_state.json

# 체크포인트 확인
ls -la checkpoints/
```

## 💾 결과 다운로드
```bash
# 모델 압축
tar -czf mistral_finetuned.tar.gz checkpoints/

# 또는
zip -r mistral_finetuned.zip checkpoints/
```

## 🆘 문제 해결

### CUDA 메모리 부족
```python
# config.py에서 배치 크기 줄이기
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
```

### 연결 끊김 대비
```bash
# tmux 사용
tmux new -s training
python run_training.py
# Ctrl+B, D로 세션 나가기

# 재연결
tmux attach -t training
```

## ✅ 완료 체크리스트
- [ ] 환경 설정 완료
- [ ] 데이터 전처리 완료
- [ ] 파인튜닝 시작
- [ ] 체크포인트 저장 확인
- [ ] 최종 모델 저장
- [ ] 결과 다운로드