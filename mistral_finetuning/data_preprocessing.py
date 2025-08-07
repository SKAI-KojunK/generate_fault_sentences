import json
import pandas as pd
from datasets import Dataset
import re
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, input_file: str):
        self.input_file = input_file
        
    def load_data(self) -> List[Dict[str, Any]]:
        """JSONL 파일을 로드합니다."""
        data = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        logger.info(f"로드된 데이터 수: {len(data)}")
        return data
    
    def format_prompt(self, instruction: str, input_text: str, output: Dict) -> str:
        """Mistral 모델에 적합한 프롬프트 형식으로 변환합니다."""
        # Mistral Instruct 형식: <s>[INST] instruction [/INST] response </s>
        prompt = f"<s>[INST] {instruction}\n\n{input_text} [/INST]\n"
        
        # output을 JSON 형태로 변환
        output_json = json.dumps(output, ensure_ascii=False, indent=2)
        prompt += f"{output_json}</s>"
        
        return prompt
    
    def clean_text(self, text: str) -> str:
        """텍스트를 정리합니다."""
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 줄바꿈 정리
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip()
    
    def validate_output(self, output: Dict) -> bool:
        """출력 데이터의 유효성을 검증합니다."""
        required_keys = ['위치', '설비유형', '현상코드', '우선순위']
        return all(key in output for key in required_keys)
    
    def process_data(self) -> Dataset:
        """데이터를 전처리하여 HuggingFace Dataset으로 변환합니다."""
        raw_data = self.load_data()
        processed_data = []
        
        for i, item in enumerate(raw_data):
            try:
                # 입력 텍스트 정리
                input_text = self.clean_text(item['input'])
                
                # 출력 데이터 검증
                if not self.validate_output(item['output']):
                    logger.warning(f"유효하지 않은 출력 데이터 건너뜀: {i}")
                    continue
                
                # 프롬프트 형식으로 변환
                prompt = self.format_prompt(
                    item['instruction'],
                    input_text,
                    item['output']
                )
                
                processed_data.append({
                    'text': prompt,
                    'instruction': item['instruction'],
                    'input': input_text,
                    'output': item['output']
                })
                
            except Exception as e:
                logger.error(f"데이터 처리 중 오류 발생 (인덱스 {i}): {e}")
                continue
        
        logger.info(f"처리된 데이터 수: {len(processed_data)}")
        
        # HuggingFace Dataset으로 변환
        dataset = Dataset.from_list(processed_data)
        return dataset
    
    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict[str, Dataset]:
        """데이터셋을 train/validation/test로 분할합니다."""
        total_size = len(dataset)
        
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        # 랜덤하게 섞기
        dataset = dataset.shuffle(seed=42)
        
        train_dataset = dataset.select(range(train_size))
        val_dataset = dataset.select(range(train_size, train_size + val_size))
        test_dataset = dataset.select(range(train_size + val_size, total_size))
        
        logger.info(f"데이터셋 분할 완료:")
        logger.info(f"  Train: {len(train_dataset)}")
        logger.info(f"  Validation: {len(val_dataset)}")
        logger.info(f"  Test: {len(test_dataset)}")
        
        return {
            'train': train_dataset,
            'validation': val_dataset,
            'test': test_dataset
        }

def main():
    """메인 실행 함수"""
    # 데이터 파일 경로
    input_file = "../output/generated_dataset.jsonl"
    
    # 전처리기 초기화
    preprocessor = DataPreprocessor(input_file)
    
    # 데이터 처리
    dataset = preprocessor.process_data()
    
    # 데이터셋 분할
    split_datasets = preprocessor.split_dataset(dataset)
    
    # 결과 저장
    for split_name, split_dataset in split_datasets.items():
        output_file = f"data/{split_name}_dataset.json"
        split_dataset.to_json(output_file, force_ascii=False)
        logger.info(f"{split_name} 데이터셋 저장: {output_file}")
    
    # 전체 데이터셋도 저장
    dataset.to_json("data/full_dataset.json", force_ascii=False)
    logger.info("전체 데이터셋 저장: data/full_dataset.json")

if __name__ == "__main__":
    main() 