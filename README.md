# 유지보수 문장 생성 및 Mistral 파인튜닝 프로젝트

유지보수 요청 문장을 생성하고 Mistral-7B 모델을 파인튜닝하여 설비 정보를 추출하는 프로젝트입니다.

## 📁 프로젝트 구조

```
generate_fault_sentences/
├── README.md                        # 이 파일
├── RUNPOD_QUICKSTART.md            # 🚀 RunPod 실행 가이드 (최신)
├── .env.example                     # 환경변수 템플릿
├── requirements.txt                 # 통합 패키지 목록
├── generate_fault_sentences.py     # 데이터셋 생성 스크립트
├── dictionary_data.xlsx            # 사전 데이터
├── output/
│   ├── generated_dataset.csv       # 생성된 데이터셋 (CSV)
│   └── generated_dataset.jsonl     # 생성된 데이터셋 (JSONL, 파인튜닝용)
└── mistral_finetuning/             # 파인튜닝 모듈
    ├── README.md                    # 파인튜닝 상세 가이드
    ├── requirements.txt             # requirements.txt 참조
    ├── config.py                    # 학습 설정 (RunPod 최적화)
    ├── runpod_setup.py             # 환경 설정 (.env 토큰 자동 로드)
    ├── run_training.py             # CLI 학습 실행
    ├── train.py                    # 파인튜닝 구현
    ├── data_preprocessing.py       # 데이터 전처리
    ├── evaluate.py                 # 모델 평가
    ├── inference.py               # 추론 스크립트
    └── finetuning_notebook.ipynb  # Jupyter 노트북
```

## 🚀 빠른 시작

### 1️⃣ 데이터 생성 (선택사항)
```bash
# 사전 데이터 기반 문장 생성
python generate_fault_sentences.py
```

### 2️⃣ RunPod 파인튜닝 (권장)

**📖 상세 가이드**: [RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md)

```bash
# 1. 프로젝트 업로드 후
cd /workspace/generate_fault_sentences

# 2. HF 토큰 설정
cp .env.example .env
# .env 파일에 실제 토큰 입력

# 3. 원샷 설정 (자동화)
python mistral_finetuning/runpod_setup.py

# 4. 파인튜닝 실행
python mistral_finetuning/run_training.py --epochs 3 --batch_size 4
```

### 3️⃣ 로컬 개발
```bash
# 의존성 설치
pip install -r requirements.txt

# 데이터 전처리
python mistral_finetuning/data_preprocessing.py

# 파인튜닝 실행
python mistral_finetuning/train.py
```

## 🔧 주요 기능

### 1. 데이터 생성
- OpenAI API를 사용한 유지보수 문장 생성
- 사전 데이터 기반 다양한 패턴 생성
- CSV/JSONL 형식 출력

### 2. 파인튜닝
- **모델**: Mistral-7B-Instruct-v0.3
- **방법**: LoRA (QLoRA 4비트)
- **최적화**: RunPod 환경 전용 설정
- **자동화**: .env 토큰 → 설치 → 전처리 → 학습

### 3. 평가 및 추론
- 정확도 메트릭 (필드별/전체)
- 오류 분석 및 성능 리포트
- 실시간 추론 테스트

## 📊 데이터 형식

**입력 예시:**
```
"No.2 PP의 기타기기 CCTV에서 SHE 발생. 주기작업 필요."
```

**출력 예시:**
```json
{
  "위치": "No.2 PP",
  "설비유형": "[IOCC]Other Instrument/ CCTV", 
  "현상코드": "SHE",
  "우선순위": "주기작업(TA.PM)"
}
```

## 🛠️ 환경 요구사항

### RunPod (권장)
- **GPU**: RTX L40 (48GB) 또는 RTX 4090 (24GB)
- **이미지**: PyTorch 2.x + CUDA 12.x
- **스토리지**: 50GB

### 로컬
- **GPU**: CUDA 지원 8GB+ VRAM
- **Python**: 3.8+
- **패키지**: requirements.txt 참조

## 📈 성능

### 메모리 최적화
- **배치 크기**: 2-4 (GPU 메모리에 따라)
- **그래디언트 누적**: 8 (실제 배치 16 유지)
- **시퀀스 길이**: 1024 (실제 데이터 최대 39토큰)
- **4비트 양자화**: 메모리 50% 절약

### 학습 시간
- **L40 48GB**: 2-3시간 (3 epochs)
- **RTX 4090**: 3-4시간 (3 epochs)
- **훈련 파라미터**: 전체의 0.5% (LoRA)

## 🔐 보안

**HF 토큰 관리:**
- `.env` 파일 사용 (Git 제외)
- 자동 로그인 지원
- 토큰 없이도 공개 모델 사용 가능

## 📞 지원

- **RunPod 가이드**: [RUNPOD_QUICKSTART.md](RUNPOD_QUICKSTART.md)
- **상세 문서**: [mistral_finetuning/README.md](mistral_finetuning/README.md)
- **문제 해결**: 각 README의 문제 해결 섹션

---

> **💡 중요**: HF 토큰만 `.env` 파일에 설정하면 나머지는 모두 자동화됩니다!