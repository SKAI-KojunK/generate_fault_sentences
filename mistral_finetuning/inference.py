import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import Dict, List, Any
import argparse
from config import inference_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaintenanceParser:
    def __init__(self, model_path: str, config: inference_config):
        self.model_path = model_path
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """파인튜닝된 모델을 로드합니다."""
        logger.info(f"모델 로딩 중: {self.model_path}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3",
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        logger.info("모델 로딩 완료")
        
    def extract_json_from_response(self, response: str) -> Dict:
        """응답에서 JSON을 추출합니다."""
        try:
            # JSON 부분 찾기
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            else:
                return {}
        except Exception as e:
            logger.warning(f"JSON 파싱 오류: {e}")
            return {}
            
    def parse_maintenance_request(self, text: str) -> Dict:
        """유지보수 요청을 파싱합니다."""
        prompt = f"<s>[INST] 다음 문장에서 설비 항목을 추출하라.\n\n{text} [/INST]\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        response = response.replace(prompt, "").strip()
        
        return self.extract_json_from_response(response)
        
    def parse_batch(self, texts: List[str]) -> List[Dict]:
        """배치로 여러 텍스트를 파싱합니다."""
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"파싱 진행률: {i+1}/{len(texts)}")
            result = self.parse_maintenance_request(text)
            results.append({
                "input": text,
                "output": result
            })
            
        return results
        
    def interactive_mode(self):
        """대화형 모드로 실행합니다."""
        logger.info("대화형 모드 시작 (종료하려면 'quit' 입력)")
        
        while True:
            try:
                user_input = input("\n유지보수 요청을 입력하세요: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료']:
                    break
                    
                if not user_input:
                    continue
                    
                # 파싱 실행
                result = self.parse_maintenance_request(user_input)
                
                # 결과 출력
                print("\n=== 파싱 결과 ===")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                print("==================")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"오류 발생: {e}")
                
        logger.info("대화형 모드 종료")
        
    def save_results(self, results: List[Dict], output_file: str):
        """결과를 파일로 저장합니다."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"결과 저장 완료: {output_file}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="유지보수 요청 파싱")
    parser.add_argument("--model_path", type=str, default=inference_config.model_path,
                       help="모델 경로")
    parser.add_argument("--input_file", type=str, help="입력 파일 (JSONL 형식)")
    parser.add_argument("--output_file", type=str, default="inference_results/parsed_results.json",
                       help="출력 파일")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드")
    parser.add_argument("--text", type=str, help="단일 텍스트 파싱")
    
    args = parser.parse_args()
    
    # 파서 초기화
    parser_instance = MaintenanceParser(args.model_path, inference_config)
    
    # 모델 로드
    parser_instance.load_model()
    
    if args.interactive:
        # 대화형 모드
        parser_instance.interactive_mode()
        
    elif args.text:
        # 단일 텍스트 파싱
        result = parser_instance.parse_maintenance_request(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    elif args.input_file:
        # 파일에서 배치 파싱
        import json
        
        # 입력 파일 읽기
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
            
        texts = [item["input"] for item in data]
        
        # 파싱 실행
        results = parser_instance.parse_batch(texts)
        
        # 결과 저장
        parser_instance.save_results(results, args.output_file)
        
    else:
        print("사용법:")
        print("  --interactive: 대화형 모드")
        print("  --text '텍스트': 단일 텍스트 파싱")
        print("  --input_file 파일.jsonl: 배치 파싱")

if __name__ == "__main__":
    main() 