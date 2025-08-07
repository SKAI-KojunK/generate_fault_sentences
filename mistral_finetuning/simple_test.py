#!/usr/bin/env python3
"""
의존성 없이 실행 가능한 간단한 테스트
"""

import os
import json

def test_data_file():
    """데이터 파일 존재 및 형식 확인"""
    print("📊 데이터 파일 테스트...")
    
    data_file = "../output/generated_dataset.jsonl"
    
    if not os.path.exists(data_file):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_file}")
        return False
    
    # 파일 크기 확인
    file_size = os.path.getsize(data_file) / (1024*1024)
    print(f"  ✅ 파일 크기: {file_size:.2f} MB")
    
    # 샘플 데이터 확인
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # 3개만 확인
                    break
                    
                data = json.loads(line)
                
                # 필수 필드 확인
                required_fields = ['instruction', 'input', 'output']
                if not all(field in data for field in required_fields):
                    print(f"❌ 필수 필드 누락: {list(data.keys())}")
                    return False
                
                # 출력 필드 확인
                output_fields = ['위치', '설비유형', '현상코드', '우선순위']
                if not all(field in data['output'] for field in output_fields):
                    print(f"❌ 출력 필드 누락: {list(data['output'].keys())}")
                    return False
                
                print(f"  ✅ 샘플 {i+1}: {data['input'][:50]}...")
        
        print(f"  ✅ 데이터 형식 검증 완료")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 파싱 오류: {e}")
        return False

def test_files_structure():
    """파인튜닝에 필요한 파일들 확인"""
    print("\n📁 파일 구조 테스트...")
    
    required_files = [
        "config.py",
        "data_preprocessing.py", 
        "train.py",
        "evaluate.py",
        "inference.py",
        "finetuning_notebook.ipynb",
        "runpod_setup.py",
        "RUNPOD_GUIDE.md"
    ]
    
    missing_files = []
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name}")
            missing_files.append(file_name)
    
    return len(missing_files) == 0

def test_config_settings():
    """설정 파일 확인"""
    print("\n⚙️ 설정 파일 테스트...")
    
    try:
        with open("config.py", 'r') as f:
            content = f.read()
            
        # 중요 설정들 확인
        checks = [
            ('mistralai/Mistral-7B-Instruct-v0.3', '모델명'),
            ('load_in_4bit: bool = True', '4비트 양자화'),
            ('lora_r: int = 16', 'LoRA rank'),
            ('per_device_train_batch_size: int = 4', '배치 크기'),
            ('report_to: str = "none"', '로깅 설정')
        ]
        
        for check, desc in checks:
            if check in content:
                print(f"  ✅ {desc}: OK")
            else:
                print(f"  ⚠️ {desc}: 확인 필요")
        
        return True
        
    except Exception as e:
        print(f"❌ 설정 파일 읽기 오류: {e}")
        return False

def test_jupyter_notebook():
    """Jupyter 노트북 파일 확인"""
    print("\n📓 Jupyter 노트북 테스트...")
    
    try:
        with open("finetuning_notebook.ipynb", 'r') as f:
            notebook = json.load(f)
        
        print(f"  ✅ 노트북 파일 형식: OK")
        print(f"  ✅ 셀 개수: {len(notebook.get('cells', []))}")
        
        # 첫 번째 셀이 마크다운인지 확인
        if notebook['cells'] and notebook['cells'][0].get('cell_type') == 'raw':
            print(f"  ✅ 헤더 셀: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ 노트북 파일 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 RunPod 파인튜닝 준비 상태 점검")
    print("=" * 50)
    
    tests = [
        ("데이터 파일", test_data_file),
        ("파일 구조", test_files_structure), 
        ("설정 파일", test_config_settings),
        ("Jupyter 노트북", test_jupyter_notebook),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} 테스트 오류: {e}")
    
    print(f"\n📊 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 RunPod 파인튜닝 준비 완료!")
        print("\n📋 다음 단계:")
        print("1. mistral_finetuning/ 폴더 전체를 RunPod에 업로드")
        print("2. generated_dataset.jsonl 파일도 함께 업로드")
        print("3. Jupyter Lab에서 finetuning_notebook.ipynb 실행")
        print("   또는 python runpod_setup.py 실행")
    else:
        print("⚠️ 일부 문제가 발견되었습니다. 위의 오류를 확인해주세요.")
    
    print(f"\n💡 권장 RunPod 사양:")
    print("  - GPU: RTX L40 (48GB) 또는 RTX 4090 (24GB)")
    print("  - 템플릿: PyTorch/Jupyter")
    print("  - 스토리지: 50GB+")

if __name__ == "__main__":
    main()