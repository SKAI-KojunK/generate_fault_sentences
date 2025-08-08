#!/usr/bin/env python3
"""
간단한 모델 테스트 스크립트
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json

def load_model(model_path="checkpoints/best"):
    """파인튜닝된 모델을 로드합니다."""
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tokenizer.pad_token = tokenizer.eos_token
        
        # 기본 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA 어댑터 로드
        model = PeftModel.from_pretrained(base_model, model_path)
        
        print(f"✅ 모델 로드 완료: {model_path}")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, max_length=512):
    """프롬프트에 대한 응답을 생성합니다."""
    try:
        # 입력 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 제거
        response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        print(f"❌ 응답 생성 실패: {e}")
        return None

def test_maintenance_parsing(model, tokenizer):
    """유지보수 데이터 파싱 테스트"""
    
    test_cases = [
        {
            "input": "시스템 오류가 발생했습니다. CPU 사용률이 95%를 초과하고 있습니다.",
            "expected": "fault_type: system_error, severity: high, component: cpu"
        },
        {
            "input": "네트워크 연결이 불안정합니다. 패킷 손실률이 증가하고 있습니다.",
            "expected": "fault_type: network_issue, severity: medium, component: network"
        },
        {
            "input": "메모리 부족 경고가 발생했습니다. 사용 가능한 메모리가 10% 미만입니다.",
            "expected": "fault_type: memory_warning, severity: medium, component: memory"
        }
    ]
    
    print("🧪 유지보수 데이터 파싱 테스트 시작")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 테스트 케이스 {i}:")
        print(f"입력: {test_case['input']}")
        print(f"예상: {test_case['expected']}")
        
        # 응답 생성
        response = generate_response(model, tokenizer, test_case['input'])
        
        if response:
            print(f"실제: {response}")
            
            # 간단한 평가
            if "fault_type" in response.lower():
                print("✅ fault_type 감지됨")
            else:
                print("❌ fault_type 감지되지 않음")
        else:
            print("❌ 응답 생성 실패")
        
        print("-" * 30)

def main():
    """메인 함수"""
    print("🚀 Mistral-7B 파인튜닝 모델 테스트")
    print("=" * 50)
    
    # 모델 로드
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        print("❌ 모델을 로드할 수 없습니다.")
        print("💡 먼저 파인튜닝을 완료하세요: python train.py")
        return
    
    # 테스트 실행
    test_maintenance_parsing(model, tokenizer)
    
    # 대화형 테스트
    print("\n💬 대화형 테스트 (종료하려면 'quit' 입력)")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n사용자: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                break
            
            if not user_input:
                continue
            
            response = generate_response(model, tokenizer, user_input)
            
            if response:
                print(f"모델: {response}")
            else:
                print("모델: 응답을 생성할 수 없습니다.")
                
        except KeyboardInterrupt:
            print("\n\n👋 테스트를 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()