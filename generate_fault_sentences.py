import os
import pandas as pd
from openai import OpenAI
import random
import json
import time
import re
from pathlib import Path
from tqdm import tqdm
import logging

# ✅ 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ✅ OpenAI API 클라이언트 설정
# 방법 1: 직접 입력 (아래 YOUR_API_KEY 부분에 실제 키 입력)
# client = OpenAI(api_key="YOUR_API_KEY")

# 방법 2: 환경변수 사용 (보안상 권장)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # 환경변수가 없으면 직접 입력된 키 사용
    api_key = "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"  # 여기에 실제 API 키를 입력하세요
    if api_key == "YOUR_API_KEY":
        raise ValueError("API 키를 입력해주세요. 코드 17번째 줄에서 'YOUR_API_KEY'를 실제 키로 변경하세요.")

client = OpenAI(api_key=api_key)

def load_excel_data(file_path="dictionary_data.xlsx"):
    """Excel 파일에서 데이터를 로드하고 검증합니다."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
    
    try:
        xls = pd.ExcelFile(file_path)
        required_sheets = ['위치', '설비유형', '현상코드', '우선순위']
        
        for sheet in required_sheets:
            if sheet not in xls.sheet_names:
                raise ValueError(f"필수 시트가 없습니다: {sheet}")
        
        df_location = pd.read_excel(xls, sheet_name='위치')
        df_equipment = pd.read_excel(xls, sheet_name='설비유형')
        df_phenomenon = pd.read_excel(xls, sheet_name='현상코드')
        df_priority = pd.read_excel(xls, sheet_name='우선순위')
        
        location_list = df_location.iloc[:, 0].dropna().tolist()
        equipment_list = df_equipment.iloc[:, 0].dropna().tolist()
        phenomenon_list = df_phenomenon.iloc[:, 0].dropna().tolist()
        priority_list = df_priority.iloc[:, 0].dropna().tolist()
        
        # 한국어 포함 설비유형 별도 추출
        korean_equipment_list = []
        for eq in equipment_list:
            has_korean = any('\uac00' <= char <= '\ud7a3' for char in str(eq))
            if has_korean:
                korean_equipment_list.append(eq)
        
        logger.info(f"한국어 포함 설비유형: {len(korean_equipment_list)}개")
        
        # 데이터 유효성 검증
        if not all([location_list, equipment_list, phenomenon_list, priority_list]):
            raise ValueError("하나 이상의 시트에 데이터가 없습니다.")
        
        logger.info(f"데이터 로드 완료 - 위치: {len(location_list)}, 설비: {len(equipment_list)}, "
                   f"현상: {len(phenomenon_list)}, 우선순위: {len(priority_list)}")
        
        return location_list, equipment_list, phenomenon_list, priority_list, korean_equipment_list
        
    except Exception as e:
        logger.error(f"Excel 파일 로드 중 오류: {e}")
        raise

# ✅ 사전 데이터 로드
location_list, equipment_list, phenomenon_list, priority_list, korean_equipment_list = load_excel_data()

def generate_combinations(target_num=10000, missing_ratio=0.15):
    """지정된 수만큼 고유한 조합을 생성합니다. 일부 필드는 '미입력'으로 설정합니다."""
    logger.info(f"총 {target_num}개의 조합 생성 시작 (미입력 비율: {missing_ratio*100:.1f}%)...")
    
    # 각 필드에 '미입력' 추가
    extended_location_list = location_list + ['미입력']
    extended_equipment_list = equipment_list + ['미입력']
    extended_phenomenon_list = phenomenon_list + ['미입력']
    extended_priority_list = priority_list + ['미입력']
    
    # 최대 가능한 조합 수 계산
    max_combinations = len(extended_location_list) * len(extended_equipment_list) * len(extended_phenomenon_list) * len(extended_priority_list)
    if target_num > max_combinations:
        logger.warning(f"요청한 조합 수({target_num})가 최대 가능 조합 수({max_combinations})보다 큽니다. "
                      f"최대 가능 수로 조정합니다.")
        target_num = max_combinations
    
    selected_combos = set()
    attempts = 0
    max_attempts = target_num * 10  # 무한 루프 방지
    
    while len(selected_combos) < target_num and attempts < max_attempts:
        # 각 필드별로 미입력 여부 결정
        use_missing_location = random.random() < missing_ratio
        use_missing_equipment = random.random() < missing_ratio
        use_missing_phenomenon = random.random() < missing_ratio
        use_missing_priority = random.random() < missing_ratio
        
        # 적어도 하나의 필드는 입력되어야 함
        if all([use_missing_location, use_missing_equipment, use_missing_phenomenon, use_missing_priority]):
            # 랜덤하게 하나의 필드는 입력되도록 함
            field_to_keep = random.choice(['location', 'equipment', 'phenomenon', 'priority'])
            if field_to_keep == 'location':
                use_missing_location = False
            elif field_to_keep == 'equipment':
                use_missing_equipment = False
            elif field_to_keep == 'phenomenon':
                use_missing_phenomenon = False
            else:
                use_missing_priority = False
        
        combo = (
            '미입력' if use_missing_location else random.choice(location_list),
            '미입력' if use_missing_equipment else random.choice(equipment_list),
            '미입력' if use_missing_phenomenon else random.choice(phenomenon_list),
            '미입력' if use_missing_priority else random.choice(priority_list)
        )
        selected_combos.add(combo)
        attempts += 1
    
    selected_combos = list(selected_combos)
    random.shuffle(selected_combos)
    
    # 미입력 통계
    missing_counts = {'위치': 0, '설비유형': 0, '현상코드': 0, '우선순위': 0}
    for combo in selected_combos:
        if combo[0] == '미입력':
            missing_counts['위치'] += 1
        if combo[1] == '미입력':
            missing_counts['설비유형'] += 1
        if combo[2] == '미입력':
            missing_counts['현상코드'] += 1
        if combo[3] == '미입력':
            missing_counts['우선순위'] += 1
    
    logger.info(f"조합 생성 완료: {len(selected_combos)}개")
    logger.info(f"미입력 통계 - 위치: {missing_counts['위치']}, 설비유형: {missing_counts['설비유형']}, "
               f"현상코드: {missing_counts['현상코드']}, 우선순위: {missing_counts['우선순위']}")
    
    return selected_combos

# ✅ 조합 수 설정 (10,000개로 변경)
target_num = 10000  # 본격 실행 준비 완료
missing_field_ratio = 0.15  # 각 필드별 미입력 비율 (15%)
selected_combos = generate_combinations(target_num, missing_field_ratio)

# ✅ GPT 프롬프트 구성 (다양한 오류 패턴 포함)
system_prompt = (
    "You are a maintenance worker writing equipment fault reports. Create natural Korean sentences with common errors based on the given location, equipment type, fault code, and priority information.\n\n"
    "IMPORTANT: Some fields may be marked as '미입력' (not entered). In these cases, create sentences that naturally omit or vaguely reference those missing fields.\n\n"
    "Include these error types:\n"
    "1. Korean-English mixing (한영 혼용)\n"
    "2. English to Korean translation (영어→한국어 번역: Pump→펌프, Valve→밸브)\n"
    "3. English to Korean phonetic transcription (영어→한국어 음차: Pressure→프레셔, Vessel→베젤)\n"
    "4. Spacing errors (띄어쓰기 오류)\n" 
    "5. Typos and phonetic variations (오타 및 음성학적 변형)\n"
    "6. Punctuation inconsistencies (구두점 불일치)\n"
    "7. Missing field handling (미입력 필드 자연스러운 처리)\n\n"
    
    "Examples:\n\n"
    
    "예시1 - 한영혼용 + 음성학적 변형:\n"
    "위치: No.1 PE, 설비유형: Pressure Vessel/Drum, 현상코드: 고장, 우선순위: 긴급\n"
    "문장: NO 1 pe의 프레셔 베젤 드럼에 고장 발생. 긴급조치 요망.\n\n"
    
    "예시2 - 오타 + 구분자 변형:\n"
    "위치: 석유제품배합/저장, 설비유형: Motor Operated Valve, 현상코드: 작동불량, 우선순위: 우선\n"
    "문장: 서규 제품배합-저장, 모터 벨브, 오작동 발생. 우선 작업 요청.\n\n"
    
    "예시3 - 띄어쓰기 오류 + 구두점 누락:\n"
    "위치: 합성수지 1창고, 설비유형: Conveyor Belt, 현상코드: 파손, 우선순위: 보통\n"
    "문장: 합성수지, 1창고 7번 라인 - 컨베이어 belt 파손 확인 보통.\n\n"
    
    "예시4 - 다양한 구분자 + 약어:\n"
    "위치: 중앙제어실, 설비유형: Control Panel, 현상코드: 이상, 우선순위: 점검\n"
    "문장: 중앙 제어실 컨트롤 패널 이상 . 점검 스케쥴링.\n\n"
    
    "예시5 - 영어의 한글 표기 + 혼합 구분자:\n"
    "위치: 원료저장탱크, 설비유형: Storage Tank, 현상코드: 누설, 우선순위: 긴급\n"
    "문장: 원료 저장 탱크 스토리지 탱크/ 누설 사고. 긴급 대응요망!.\n\n"
    
    "예시8 - 영어 설비명의 한국어 번역:\n"
    "위치: 중앙제어실, 설비유형: Heat Exchanger, 현상코드: 온도이상, 우선순위: 우선\n"
    "문장: 중앙제어실 열교환기 온도 이상 발견. 우선 점검 요청.\n\n"
    
    "예시9 - 영어 설비명의 음차 + 한국어 혼용:\n"
    "위치: 보일러실, 설비유형: Centrifugal Pump, 현상코드: 진동, 우선순위: 긴급\n"
    "문장: 보일러실 센트리퓨갈 펌프 진동 심함. 긴급 수리 필요.\n\n"
    
    "예시10 - 복합 변환 (번역+음차+약어):\n"
    "위치: 냉각탑, 설비유형: Motor Operated Valve, 현상코드: 작동불량, 우선순위: 점검\n"
    "문장: 냉각탑 모터 구동 밸브 MOV 작동 불량. 점검 스케줄링.\n\n"
    
    "예시6 - 미입력 필드 처리 (위치 미입력):\n"
    "위치: 미입력, 설비유형: Pump, 현상코드: 이상음, 우선순위: 점검\n"
    "문장: 펌프에서 이상음 발생. 점검 필요.\n\n"
    
    "예시7 - 미입력 필드 처리 (우선순위 미입력):\n"
    "위치: 보일러실, 설비유형: Heat Exchanger, 현상코드: 온도이상, 우선순위: 미입력\n"
    "문장: 보일러실 열교환기 온도 이상 확인됨.\n\n"
    
    "Generate sentences that reflect real-world maintenance log variations with these error patterns."
)

def call_openai_api(messages, max_retries=3, delay=1):
    """OpenAI API를 호출하고 재시도 로직을 포함합니다."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"API 호출 시도 {attempt + 1}/{max_retries} 실패: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (2 ** attempt))  # 지수 백오프
            else:
                logger.error(f"API 호출 최종 실패: {e}")
                raise

def process_batch(batch, batch_num, total_batches):
    """배치를 처리하고 결과를 반환합니다."""
    logger.info(f"배치 {batch_num}/{total_batches} 처리 중... ({len(batch)}개 문장)")
    
    prompt_lines = []
    for i, (loc, equip, phen, prio) in enumerate(batch, 1):
        prompt_lines.append(f"{i}. 위치: {loc}, 설비유형: {equip}, 현상코드: {phen}, 우선순위: {prio}")
    prompt = "\n".join(prompt_lines)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Create exactly {len(batch)} Korean maintenance report sentences with various error patterns. "
                                   f"Each sentence should be on a new line without numbers, maintaining input order. "
                                   f"Vary the error types across sentences (mixing, spacing, typos, punctuation):\n\n{prompt}"}
    ]

    try:
        output_text = call_openai_api(messages)
        lines = [line.strip() for line in output_text.splitlines() if line.strip()]
        
        # 번호 제거
        cleaned_lines = []
        for line in lines:
            # 다양한 번호 패턴 제거 (1., 1), 1-, 1. 등)
            cleaned_line = re.sub(r'^[\d]+[\.\)\-]\s*', '', line)
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        # 결과 수가 입력과 다른 경우 처리
        if len(cleaned_lines) != len(batch):
            logger.warning(f"배치 {batch_num}: 예상 {len(batch)}개, 실제 {len(cleaned_lines)}개 문장 생성됨")
            
            # 부족한 경우: 마지막 문장 복제
            while len(cleaned_lines) < len(batch):
                if cleaned_lines:
                    cleaned_lines.append(cleaned_lines[-1] + " (복제)")
                else:
                    cleaned_lines.append("기본 문장")
            
            # 초과한 경우: 뒤의 것들 제거
            cleaned_lines = cleaned_lines[:len(batch)]
        
        batch_results = []
        for combo, sentence in zip(batch, cleaned_lines):
            loc, equip, phen, prio = combo
            batch_results.append({
                "instruction": "다음 문장에서 설비 항목을 추출하라.",
                "input": sentence,
                "output": {
                    "위치": loc,
                    "설비유형": equip,
                    "현상코드": phen,
                    "우선순위": prio
                }
            })
        
        logger.info(f"배치 {batch_num} 완료: {len(batch_results)}개 문장 생성")
        return batch_results
        
    except Exception as e:
        logger.error(f"배치 {batch_num} 처리 실패: {e}")
        return []

# ✅ 배치 처리 개선
BATCH_SIZE = 50  # 본격 실행용
batches = [selected_combos[i:i+BATCH_SIZE] for i in range(0, len(selected_combos), BATCH_SIZE)]
total_batches = len(batches)

logger.info(f"총 {total_batches}개 배치로 처리 시작 (배치당 {BATCH_SIZE}개)")

results = []
failed_batches = []

# 진행률 표시와 함께 배치 처리
for i, batch in enumerate(tqdm(batches, desc="배치 처리 중"), 1):
    batch_results = process_batch(batch, i, total_batches)
    if batch_results:
        results.extend(batch_results)
    else:
        failed_batches.append(i)
        logger.warning(f"배치 {i} 실패 - 재시도 목록에 추가")

# 실패한 배치 재시도
if failed_batches:
    logger.info(f"실패한 {len(failed_batches)}개 배치 재시도 중...")
    for batch_num in failed_batches:
        batch = batches[batch_num - 1]
        logger.info(f"배치 {batch_num} 재시도 중...")
        batch_results = process_batch(batch, batch_num, total_batches)
        if batch_results:
            results.extend(batch_results)
        else:
            logger.error(f"배치 {batch_num} 재시도 실패")

# ✅ 결과 저장 (JSONL과 CSV 모두 지원)
def save_results(results, base_filename="generated_dataset"):
    """결과를 JSONL과 CSV 형식으로 저장합니다."""
    if not results:
        logger.error("저장할 결과가 없습니다.")
        return
    
    # JSONL 저장
    jsonl_path = f"{base_filename}.jsonl"
    try:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for item in results:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")
        logger.info(f"JSONL 파일 저장 완료: {jsonl_path}")
    except Exception as e:
        logger.error(f"JSONL 저장 실패: {e}")
    
    # CSV 저장 (분석 및 검토용)
    csv_path = f"{base_filename}.csv"
    try:
        csv_data = []
        for item in results:
            csv_row = {
                "instruction": item["instruction"],
                "input": item["input"],
                "위치": item["output"]["위치"],
                "설비유형": item["output"]["설비유형"],
                "현상코드": item["output"]["현상코드"],
                "우선순위": item["output"]["우선순위"]
            }
            csv_data.append(csv_row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"CSV 파일 저장 완료: {csv_path}")
    except Exception as e:
        logger.error(f"CSV 저장 실패: {e}")
    
    return jsonl_path, csv_path

# 결과 저장
if results:
    # 출력 폴더 생성 (필요시)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 결과 저장 (output 폴더에 저장)
    jsonl_file, csv_file = save_results(results, f"{output_dir}/generated_dataset")
    
    # 최종 통계
    logger.info("="*50)
    logger.info("📊 최종 생성 결과")
    logger.info("="*50)
    logger.info(f"목표 문장 수: {target_num:,}")
    logger.info(f"실제 생성 수: {len(results):,}")
    logger.info(f"성공률: {len(results)/target_num*100:.1f}%")
    logger.info(f"총 배치 수: {total_batches}")
    logger.info(f"실패 배치 수: {len(failed_batches)}")
    logger.info("="*50)
    logger.info(f"✅ JSONL 파일: {jsonl_file}")
    logger.info(f"✅ CSV 파일: {csv_file}")
    logger.info("="*50)
    
    # 샘플 결과 표시
    if len(results) >= 3:
        logger.info("📝 생성된 문장 샘플:")
        for i, sample in enumerate(results[:3], 1):
            logger.info(f"{i}. {sample['input']}")
            logger.info(f"   → 위치: {sample['output']['위치']}, "
                       f"설비: {sample['output']['설비유형']}, "
                       f"현상: {sample['output']['현상코드']}, "
                       f"우선순위: {sample['output']['우선순위']}")
    
else:
    logger.error("❌ 생성된 결과가 없습니다. 설정을 확인해주세요.")

logger.info("프로그램 실행 완료")
