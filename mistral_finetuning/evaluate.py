import json
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
from typing import Dict, List, Any, Tuple
import pandas as pd
from config import evaluation_config, inference_config, training_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str, config: evaluation_config):
        self.model_path = model_path
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """파인튜닝된 모델을 로드합니다."""
        logger.info(f"모델 로딩 중: {self.model_path}")
        
        # 토크나이저 로드 (학습에 사용한 동일 베이스 모델)
        self.tokenizer = AutoTokenizer.from_pretrained(
            training_config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 베이스 모델 로드 후 LoRA 어댑터 적용 (PEFT)
        base_model = AutoModelForCausalLM.from_pretrained(
            training_config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        logger.info("모델 로딩 완료")
        
    def load_test_data(self):
        """테스트 데이터를 로드합니다."""
        logger.info("테스트 데이터 로딩 중...")
        
        self.test_dataset = load_dataset("json", data_files=self.config.test_file)
        
        if self.config.max_eval_samples:
            self.test_dataset = self.test_dataset["train"].select(range(self.config.max_eval_samples))
        else:
            self.test_dataset = self.test_dataset["train"]
            
        logger.info(f"테스트 데이터 크기: {len(self.test_dataset)}")
        
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
        except:
            return {}
            
    def parse_maintenance_request(self, text: str) -> Dict:
        """유지보수 요청을 파싱합니다."""
        prompt = f"<s>[INST] 다음 문장에서 설비 항목을 추출하라.\n\n{text} [/INST]\n"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=inference_config.max_new_tokens,
                temperature=inference_config.temperature,
                top_p=inference_config.top_p,
                do_sample=inference_config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 프롬프트 부분 제거
        response = response.replace(prompt, "").strip()
        
        return self.extract_json_from_response(response)
        
    def evaluate_single_sample(self, sample: Dict) -> Dict:
        """단일 샘플을 평가합니다."""
        input_text = sample["input"]
        expected_output = sample["output"]
        
        # 모델 예측
        predicted_output = self.parse_maintenance_request(input_text)
        
        # 정확도 계산
        accuracy = 0
        total_fields = 0
        field_accuracies = {}
        
        for field in ["위치", "설비유형", "현상코드", "우선순위"]:
            expected = expected_output.get(field, "")
            predicted = predicted_output.get(field, "")
            
            field_accuracy = 1.0 if expected == predicted else 0.0
            field_accuracies[field] = field_accuracy
            accuracy += field_accuracy
            total_fields += 1
        
        overall_accuracy = accuracy / total_fields if total_fields > 0 else 0
        
        return {
            "input": input_text,
            "expected": expected_output,
            "predicted": predicted_output,
            "overall_accuracy": overall_accuracy,
            "field_accuracies": field_accuracies,
            "is_correct": overall_accuracy == 1.0
        }
        
    def evaluate_model(self) -> Dict:
        """모델을 평가합니다."""
        logger.info("모델 평가 시작...")
        
        results = []
        correct_count = 0
        total_count = len(self.test_dataset)
        
        for i, sample in enumerate(self.test_dataset):
            if i % 100 == 0:
                logger.info(f"평가 진행률: {i}/{total_count}")
                
            result = self.evaluate_single_sample(sample)
            results.append(result)
            
            if result["is_correct"]:
                correct_count += 1
                
        # 전체 정확도
        overall_accuracy = correct_count / total_count
        
        # 필드별 정확도
        field_accuracies = {}
        for field in ["위치", "설비유형", "현상코드", "우선순위"]:
            field_correct = sum(1 for r in results if r["field_accuracies"][field] == 1.0)
            field_accuracies[field] = field_correct / total_count
            
        # 결과 저장
        evaluation_results = {
            "overall_accuracy": overall_accuracy,
            "field_accuracies": field_accuracies,
            "total_samples": total_count,
            "correct_samples": correct_count,
            "detailed_results": results
        }
        
        # 결과를 파일로 저장
        with open(f"{self.config.results_dir}/evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            
        # 상세 결과를 CSV로 저장
        detailed_df = pd.DataFrame(results)
        detailed_df.to_csv(f"{self.config.results_dir}/detailed_evaluation.csv", index=False, encoding="utf-8")
        
        logger.info("평가 완료")
        logger.info(f"전체 정확도: {overall_accuracy:.4f}")
        logger.info(f"필드별 정확도: {field_accuracies}")
        
        return evaluation_results
        
    def generate_error_analysis(self, results: List[Dict]):
        """오류 분석을 생성합니다."""
        logger.info("오류 분석 생성 중...")
        
        # 오류가 있는 샘플들
        error_samples = [r for r in results if not r["is_correct"]]
        
        # 필드별 오류 분석
        field_errors = {}
        for field in ["위치", "설비유형", "현상코드", "우선순위"]:
            field_errors[field] = []
            
        for sample in error_samples:
            for field in ["위치", "설비유형", "현상코드", "우선순위"]:
                if sample["field_accuracies"][field] == 0.0:
                    field_errors[field].append({
                        "input": sample["input"],
                        "expected": sample["expected"][field],
                        "predicted": sample["predicted"].get(field, ""),
                    })
                    
        # 오류 분석 저장
        with open(f"{self.config.results_dir}/error_analysis.json", "w", encoding="utf-8") as f:
            json.dump(field_errors, f, ensure_ascii=False, indent=2)
            
        logger.info("오류 분석 완료")
        
    def generate_performance_report(self, results: Dict):
        """성능 리포트를 생성합니다."""
        logger.info("성능 리포트 생성 중...")
        
        report = {
            "model_path": self.model_path,
            "evaluation_date": pd.Timestamp.now().isoformat(),
            "overall_accuracy": results["overall_accuracy"],
            "field_accuracies": results["field_accuracies"],
            "total_samples": results["total_samples"],
            "correct_samples": results["correct_samples"],
            "error_rate": 1 - results["overall_accuracy"],
        }
        
        # 리포트 저장
        with open(f"{self.config.results_dir}/performance_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        # 마크다운 리포트 생성
        md_report = f"""# 모델 성능 리포트

## 개요
- **모델 경로**: {self.model_path}
- **평가 날짜**: {report['evaluation_date']}
- **전체 샘플 수**: {results['total_samples']}
- **정확한 샘플 수**: {results['correct_samples']}

## 정확도
- **전체 정확도**: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)
- **오류율**: {1-results['overall_accuracy']:.4f} ({(1-results['overall_accuracy'])*100:.2f}%)

## 필드별 정확도
"""
        
        for field, accuracy in results["field_accuracies"].items():
            md_report += f"- **{field}**: {accuracy:.4f} ({accuracy*100:.2f}%)\n"
            
        with open(f"{self.config.results_dir}/performance_report.md", "w", encoding="utf-8") as f:
            f.write(md_report)
            
        logger.info("성능 리포트 생성 완료")

def main():
    """메인 실행 함수"""
    logger.info("모델 평가 시작")
    
    # 평가기 초기화
    evaluator = ModelEvaluator(inference_config.model_path, evaluation_config)
    
    # 모델 로드
    evaluator.load_model()
    
    # 테스트 데이터 로드
    evaluator.load_test_data()
    
    # 모델 평가
    results = evaluator.evaluate_model()
    
    # 오류 분석
    evaluator.generate_error_analysis(results["detailed_results"])
    
    # 성능 리포트
    evaluator.generate_performance_report(results)
    
    logger.info("모델 평가 완료!")

if __name__ == "__main__":
    main() 