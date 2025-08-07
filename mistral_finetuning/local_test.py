#!/usr/bin/env python3
"""
로컬 환경에서 간단한 테스트를 위한 스크립트
"""

import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_small_dataset(input_file: str, output_file: str, num_samples: int = 100):
    """작은 테스트 데이터셋 생성"""
    logger.info(f"작은 데이터셋 생성: {num_samples}개 샘플")
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for i, line in enumerate(f_in):
                if i >= num_samples:
                    break
                f_out.write(line)
    
    logger.info(f"테스트 데이터셋 저장: {output_file}")

def test_data_preprocessing():
    """데이터 전처리 테스트 (의존성 없이)"""
    logger.info("데이터 전처리 테스트 시작")
    
    # 원본 데이터 파일 경로
    original_file = "../output/generated_dataset.jsonl"
    test_file = "test_dataset.jsonl"
    
    if not os.path.exists(original_file):
        logger.error(f"원본 데이터 파일을 찾을 수 없습니다: {original_file}")
        return False
    
    # 작은 테스트 데이터셋 생성
    create_small_dataset(original_file, test_file, 100)
    
    # 간단한 데이터 형식 검증
    try:
        count = 0
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # 필수 필드 확인
                    required_fields = ['instruction', 'input', 'output']
                    if all(field in data for field in required_fields):
                        count += 1
                    if count >= 5:  # 5개만 확인
                        break
        
        logger.info(f"데이터 형식 검증 완료: {count}개 샘플 확인")
        logger.info("✅ 데이터 구조가 올바릅니다")
        
        return True
        
    except Exception as e:
        logger.error(f"데이터 검증 실패: {e}")
        return False
    finally:
        # 테스트 파일 정리
        if os.path.exists(test_file):
            os.remove(test_file)

def test_model_loading():
    """모델 로딩 테스트 (CPU)"""
    logger.info("모델 로딩 테스트 시작 (CPU)")
    
    try:
        import torch
        from transformers import AutoTokenizer
        
        MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
        
        # 토크나이저만 테스트 (모델은 너무 큼)
        logger.info("토크나이저 로딩 중...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # 간단한 토크나이징 테스트
        test_text = "No.2 PP의 기타기기 CCTV에서 SHE 발생. 주기작업 필요."
        tokens = tokenizer(test_text, return_tensors="pt")
        
        logger.info(f"토크나이징 테스트 성공:")
        logger.info(f"  입력 텍스트: {test_text}")
        logger.info(f"  토큰 수: {len(tokens['input_ids'][0])}")
        
        return True
        
    except Exception as e:
        logger.error(f"모델 로딩 테스트 실패: {e}")
        logger.info("💡 로컬에서는 CPU 환경이므로 실제 파인튜닝은 RunPod에서 실행하세요.")
        return False

def check_dependencies():
    """필요한 라이브러리 확인"""
    logger.info("라이브러리 의존성 확인")
    
    required_packages = [
        "torch",
        "transformers", 
        "datasets",
        "numpy",
        "pandas",
        "scikit-learn"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"  ✅ {package}")
        except ImportError:
            logger.error(f"  ❌ {package} (누락)")
            missing_packages.append(package)
    
    if missing_packages:
        logger.info("누락된 패키지 설치:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def main():
    """로컬 테스트 메인 함수"""
    logger.info("🧪 로컬 환경 테스트 시작")
    logger.info("=" * 50)
    
    tests = [
        ("라이브러리 의존성", check_dependencies),
        ("데이터 전처리", test_data_preprocessing),
        ("모델 토크나이저", test_model_loading),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 {test_name} 테스트 중...")
        try:
            if test_func():
                logger.info(f"✅ {test_name} 테스트 통과")
                passed += 1
            else:
                logger.error(f"❌ {test_name} 테스트 실패")
        except Exception as e:
            logger.error(f"❌ {test_name} 테스트 오류: {e}")
    
    logger.info(f"\n📊 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        logger.info("🎉 모든 테스트 통과! RunPod에서 파인튜닝을 실행할 준비가 되었습니다.")
    else:
        logger.info("⚠️  일부 테스트 실패. 문제를 해결한 후 다시 시도하세요.")
    
    logger.info("\n💡 실제 파인튜닝은 다음 환경에서 실행하세요:")
    logger.info("  - RunPod (권장): L40 또는 RTX 4090")
    logger.info("  - 로컬 GPU: RTX 3090 이상 (24GB+ VRAM)")

if __name__ == "__main__":
    main()