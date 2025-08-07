#!/usr/bin/env python3
"""
데이터 전처리 실행 스크립트
"""

import os
import sys
import logging
from data_preprocessing import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """데이터 전처리를 실행합니다."""
    logger.info("데이터 전처리 시작")
    
    # 입력 파일 경로
    input_file = "../output/generated_dataset.jsonl"
    
    if not os.path.exists(input_file):
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_file}")
        sys.exit(1)
    
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
    
    logger.info("데이터 전처리 완료!")

if __name__ == "__main__":
    main() 