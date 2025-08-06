# 🚀 실행 준비 가이드

## 📊 현재 상태 확인 ✅

### 파일 구조
```
generate_fault_sentences/
├── generate_fault_sentences.py ✅ (321줄)
├── dictionary_data.xlsx ✅ (4개 시트)
├── requirements.txt ✅
├── README.md ✅
└── output/ (자동 생성됨)
```

### Excel 데이터 현황
- **설비유형**: 163개 항목
- **위치**: 429개 항목  
- **현상코드**: 12개 항목
- **우선순위**: 4개 항목

**총 조합 가능 수**: 163 × 429 × 12 × 4 = **3,368,436개** (10,000개 생성 여유롭게 가능!)

## 🔧 실행 단계

### 1단계: 패키지 설치
```bash
pip install -r requirements.txt
```

### 2단계: OpenAI API 키 설정
```bash
export OPENAI_API_KEY="your-api-key-here"
```

또는 현재 터미널에서만:
```bash
OPENAI_API_KEY="your-api-key-here" python generate_fault_sentences.py
```

### 3단계: 실행
```bash
python generate_fault_sentences.py
```

## 📁 출력 결과물 위치

**저장 경로**: `./output/` 폴더
- `output/generated_dataset.jsonl` - Fine-tuning용 JSONL 형식
- `output/generated_dataset.csv` - 검토용 CSV 형식

**절대 경로**: 
`/Users/YMARX/Dropbox/2025_ECMiner/C_P02_SKAI/03_진행/LLM_FineTuning/generate_fault_sentences/output/`

## ⚙️ 현재 설정값

- **모델**: `gpt-4o` ✅
- **생성 문장 수**: 10,000개
- **미입력 비율**: 15% (각 필드별)
- **배치 크기**: 50개씩 처리
- **예상 실행 시간**: 30-60분

## 🔧 새로운 기능: 미입력 처리

### 개요
실제 현장에서는 모든 필드가 완전히 입력되지 않는 경우가 많습니다. 이를 반영하여:

- **각 필드별 15% 확률**로 "미입력" 상태 생성
- **최소 1개 필드**는 항상 입력된 상태 보장
- **자연스러운 문장 생성**: 미입력 필드는 문장에서 자연스럽게 생략

### 예시
```
입력: 위치: 미입력, 설비유형: Pump, 현상코드: 이상음, 우선순위: 점검
출력: "펌프에서 이상음 발생. 점검 필요."

입력: 위치: 보일러실, 설비유형: Heat Exchanger, 현상코드: 온도이상, 우선순위: 미입력  
출력: "보일러실 열교환기 온도 이상 확인됨."
```

## 🔍 모니터링

실행 중 다음 정보가 실시간으로 표시됩니다:
- 배치 처리 진행률 (tqdm 진행바)
- 각 배치별 성공/실패 로그
- 최종 생성 통계 및 샘플 결과

## ⚠️ 주의사항

1. **API 키**: 환경변수 설정 필수
2. **인터넷 연결**: OpenAI API 접근 필요
3. **API 비용**: 약 $20-40 예상 (10,000개 문장 기준)
4. **실행 시간**: 중간에 중단하지 말고 완료까지 대기