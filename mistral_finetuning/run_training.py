#!/usr/bin/env python3
"""
파인튜닝 실행 스크립트
"""

import os
import sys
import logging
import argparse
from train import MistralFineTuner
from config import training_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """파인튜닝을 실행합니다."""
    parser = argparse.ArgumentParser(description="Mistral 파인튜닝 실행")
    parser.add_argument("--resume", type=str, help="체크포인트에서 재시작")
    parser.add_argument("--epochs", type=int, default=3, help="학습 에포크 수")
    parser.add_argument("--batch_size", type=int, default=4, help="배치 크기")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="학습률")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    args = parser.parse_args()
    
    # 설정 업데이트
    if args.resume:
        training_config.resume_from_checkpoint = args.resume
    training_config.num_train_epochs = args.epochs
    training_config.per_device_train_batch_size = args.batch_size
    training_config.per_device_eval_batch_size = args.batch_size
    training_config.learning_rate = args.learning_rate
    training_config.lora_r = args.lora_r
    training_config.lora_alpha = args.lora_alpha
    
    logger.info("Mistral 파인튜닝 시작")
    logger.info(f"설정: {vars(training_config)}")
    
    # 데이터 파일 확인
    if not os.path.exists(training_config.train_file):
        logger.error(f"훈련 데이터 파일을 찾을 수 없습니다: {training_config.train_file}")
        logger.info("먼저 데이터 전처리를 실행하세요: python run_preprocessing.py")
        sys.exit(1)
    
    if not os.path.exists(training_config.validation_file):
        logger.error(f"검증 데이터 파일을 찾을 수 없습니다: {training_config.validation_file}")
        logger.info("먼저 데이터 전처리를 실행하세요: python run_preprocessing.py")
        sys.exit(1)
    
    # 파인튜너 초기화
    finetuner = MistralFineTuner(training_config)
    
    try:
        # 모델과 토크나이저 설정
        finetuner.setup_model_and_tokenizer()
        
        # LoRA 설정
        finetuner.setup_lora()
        
        # 데이터셋 로드
        finetuner.load_datasets()
        
        # 데이터셋 준비
        finetuner.prepare_datasets()
        
        # 학습 설정
        finetuner.setup_training()
        
        # 파인튜닝 실행
        finetuner.train()
        
        # 평가
        finetuner.evaluate()
        
        # 모델 정보 저장
        finetuner.save_model_info()
        
        logger.info("파인튜닝 완료!")
        
    except Exception as e:
        logger.error(f"파인튜닝 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main() 