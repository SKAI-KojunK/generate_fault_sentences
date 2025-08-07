#!/usr/bin/env python3
"""
RunPod 환경 설정 및 파인튜닝 실행 스크립트
"""

import os
import subprocess
import sys

def check_gpu():
    """GPU 정보 확인"""
    try:
        import torch
        print(f"🔍 GPU 확인:")
        print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU 이름: {torch.cuda.get_device_name()}")
            print(f"  GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"  CUDA 버전: {torch.version.cuda}")
        return torch.cuda.is_available()
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")
        return False

def install_requirements():
    """필요한 라이브러리 설치"""
    print("📦 필요한 라이브러리 설치 중...")
    
    packages = [
        "peft>=0.6.0",
        "bitsandbytes>=0.41.0", 
        "trl>=0.7.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"  ✅ {package.split('>=')[0]} 설치 완료")
        except subprocess.CalledProcessError:
            print(f"  ❌ {package} 설치 실패")

def check_data_file():
    """데이터 파일 확인"""
    print("📁 데이터 파일 확인 중...")
    
    data_files = [
        "generated_dataset.jsonl",
        "../output/generated_dataset.jsonl",
        "../../output/generated_dataset.jsonl"
    ]
    
    for file_path in data_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024*1024)
            print(f"  ✅ 데이터 파일 발견: {file_path} ({file_size:.2f} MB)")
            return file_path
    
    print("  ❌ 데이터 파일을 찾을 수 없습니다.")
    print("  💡 'generated_dataset.jsonl' 파일을 현재 디렉토리에 업로드해주세요.")
    return None

def setup_directories():
    """필요한 디렉토리 생성"""
    directories = ["data", "checkpoints", "logs", "results", "inference_results"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  📁 {directory}/ 디렉토리 준비 완료")

def run_preprocessing(data_file):
    """데이터 전처리 실행"""
    print("🔄 데이터 전처리 실행 중...")
    
    try:
        from data_preprocessing import DataPreprocessor
        
        preprocessor = DataPreprocessor(data_file)
        dataset = preprocessor.process_data()
        split_datasets = preprocessor.split_dataset(dataset)
        
        # 결과 저장
        for split_name, split_dataset in split_datasets.items():
            output_file = f"data/{split_name}_dataset.json"
            split_dataset.to_json(output_file, force_ascii=False)
            print(f"  ✅ {split_name} 데이터셋 저장: {output_file}")
        
        return True
    except Exception as e:
        print(f"  ❌ 전처리 중 오류: {e}")
        return False

def main():
    """메인 설정 함수"""
    print("🚀 RunPod 파인튜닝 환경 설정 시작")
    print("=" * 50)
    
    # 1. GPU 확인
    if not check_gpu():
        print("❌ GPU를 사용할 수 없습니다. CPU 환경에서는 파인튜닝이 매우 느립니다.")
        return
    
    # 2. 라이브러리 설치
    install_requirements()
    
    # 3. 디렉토리 설정
    print("\n📁 디렉토리 설정 중...")
    setup_directories()
    
    # 4. 데이터 파일 확인
    print("\n📊 데이터 파일 확인 중...")
    data_file = check_data_file()
    
    if data_file:
        # 5. 데이터 전처리
        if run_preprocessing(data_file):
            print("\n✅ 환경 설정 완료!")
            print("🎯 이제 다음 중 하나를 선택하여 파인튜닝을 시작하세요:")
            print("  1. Jupyter 노트북: jupyter notebook finetuning_notebook.ipynb")
            print("  2. Python 스크립트: python run_training.py")
        else:
            print("\n❌ 데이터 전처리 실패")
    else:
        print("\n❌ 데이터 파일이 없어 전처리를 건너뜁니다.")
        print("💡 'generated_dataset.jsonl' 파일을 업로드한 후 다시 실행해주세요.")

if __name__ == "__main__":
    main()