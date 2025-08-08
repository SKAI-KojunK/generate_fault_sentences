#!/usr/bin/env python3
"""
RunPod 환경 설정 및 파인튜닝 실행 스크립트
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file"""
    env_paths = [
        Path(__file__).parent.parent / ".env",  # 프로젝트 루트/.env
        Path(__file__).parent / ".env",         # mistral_finetuning/.env
        Path.cwd() / ".env"                     # 현재 디렉토리/.env
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            print(f"  ✅ .env 파일 발견: {env_path}")
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and value:
                            os.environ[key] = value
                            if key == "HUGGING_FACE_HUB_TOKEN":
                                print(f"  ✅ HF 토큰 환경변수 설정 완료")
            return True
    
    print("  ℹ️ .env 파일을 찾을 수 없습니다. 환경변수나 인자로 토큰을 설정하세요.")
    return False

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

def install_requirements(requirements_file: str = None):
    """필요한 라이브러리 설치: 프로젝트 루트의 requirements.txt를 사용"""
    print("📦 필요한 라이브러리 설치 중...")

    # 프로젝트 루트 기준 requirements 경로 탐색
    candidate_paths = []
    if requirements_file:
        candidate_paths.append(requirements_file)
    candidate_paths.extend([
        "../requirements.txt",
        "../../requirements.txt",
        "requirements.txt",
        "mistral_finetuning/requirements.txt",
    ])

    req_path = None
    for path in candidate_paths:
        if os.path.exists(path):
            req_path = path
            break

    if not req_path:
        print("  ❌ requirements.txt를 찾을 수 없습니다. 수동 설치가 필요합니다.")
        return

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", req_path])
        print(f"  ✅ requirements 설치 완료: {req_path}")
    except subprocess.CalledProcessError:
        print(f"  ❌ requirements 설치 실패: {req_path}")

def install_system_dependencies():
    """시스템 의존성 설치"""
    print("🔧 시스템 의존성 설치 중...")
    
    try:
        # tmux 설치 (세션 관리용)
        subprocess.check_call(["apt", "update", "-qq"])
        subprocess.check_call(["apt", "install", "-y", "tmux"])
        print("  ✅ tmux 설치 완료")
    except subprocess.CalledProcessError:
        print("  ⚠️ tmux 설치 실패 (선택사항)")

def setup_huggingface_auth(token: str = None):
    """Hugging Face 인증 설정 (비대화형 지원)

    우선순위: 함수 인자 token > 환경변수 HUGGING_FACE_HUB_TOKEN > 로그인 생략
    """
    print("🔐 Hugging Face 인증 설정...")

    # 경고/워닝 억제 환경 변수
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    provided_token = token or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if provided_token:
        # 환경 변수 설정 및 프로그램 방식 로그인
        os.environ["HUGGING_FACE_HUB_TOKEN"] = provided_token
        try:
            from huggingface_hub import login
            login(token=provided_token, add_to_git_credential=True)
            print("  ✅ Hugging Face 토큰 설정 및 로그인 완료")
        except Exception:
            # CLI fallback
            try:
                subprocess.check_call(["huggingface-cli", "login", "--token", provided_token, "--add-to-git-credential"], stdout=subprocess.DEVNULL)
                print("  ✅ Hugging Face CLI 로그인 완료")
            except Exception:
                print("  ⚠️ 토큰 로그인 실패. 환경 변수만 설정되었습니다.")
    else:
        print("  ℹ️ HUGGING_FACE_HUB_TOKEN 미설정. private 모델이 아니면 계속 진행 가능합니다.")

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
    
    # 0. .env 파일 로드 (제일 먼저)
    print("📄 환경변수 로드 중...")
    load_env_file()
    
    parser = argparse.ArgumentParser(description="RunPod 환경 설정")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face 액세스 토큰")
    parser.add_argument("--requirements", type=str, default=None, help="requirements.txt 경로")
    args = parser.parse_args()
    
    # 1. 시스템 의존성 설치
    install_system_dependencies()
    
    # 2. GPU 확인
    if not check_gpu():
        print("❌ GPU를 사용할 수 없습니다. CPU 환경에서는 파인튜닝이 매우 느립니다.")
        return
    
    # 3. 라이브러리 설치
    install_requirements(requirements_file=args.requirements)
    
    # 4. Hugging Face 인증 설정 (.env에서 로드된 토큰 또는 인자 토큰 사용)
    setup_huggingface_auth(token=args.hf_token)
    
    # 5. 디렉토리 설정
    print("\n📁 디렉토리 설정 중...")
    setup_directories()
    
    # 6. 데이터 파일 확인
    print("\n📊 데이터 파일 확인 중...")
    data_file = check_data_file()
    
    if data_file:
        # 7. 데이터 전처리
        if run_preprocessing(data_file):
            print("\n✅ 환경 설정 완료!")
            print("🎯 이제 다음 중 하나를 선택하여 파인튜닝을 시작하세요:")
            print("  1. tmux 세션: tmux new -s finetuning")
            print("  2. Python 스크립트: python train.py")
            print("  3. 백그라운드 실행: nohup python train.py > training.log 2>&1 &")
        else:
            print("\n❌ 데이터 전처리 실패")
    else:
        print("\n❌ 데이터 파일이 없어 전처리를 건너뜁니다.")
        print("💡 'generated_dataset.jsonl' 파일을 업로드한 후 다시 실행해주세요.")

if __name__ == "__main__":
    main()