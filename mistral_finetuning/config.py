import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    # 모델 설정
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"  
    model_revision: str = "main"
    
    # 데이터 설정
    train_file: str = "data/train_dataset.json"
    validation_file: str = "data/validation_dataset.json"
    test_file: str = "data/test_dataset.json"
    
    # 학습 설정
    output_dir: str = "checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2  # RunPod 메모리 최적화: 4 → 2
    per_device_eval_batch_size: int = 2   # RunPod 메모리 최적화: 4 → 2
    gradient_accumulation_steps: int = 8  # RunPod 메모리 최적화: 4 → 8 (실제 배치 크기 16 유지)
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # LoRA 설정
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = None
    
    # QLoRA 설정 (4비트 양자화)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # 기타 설정
    max_seq_length: int = 1024  # RunPod 메모리 최적화: 2048 → 1024 (실제 데이터 최대 39 토큰)
    dataloader_pin_memory: bool = False
    remove_unused_columns: bool = False
    group_by_length: bool = True
    ddp_find_unused_parameters: bool = False
    dataloader_num_workers: int = 0  # RunPod 메모리 최적화: 4 → 0
    
    # 체크포인트 및 복구 설정
    resume_from_checkpoint: Optional[str] = None
    save_strategy: str = "steps"
    evaluation_strategy: str = "steps"
    
    # 로깅 설정
    logging_dir: str = "logs"
    report_to: str = "none"  # RunPod에서는 "none" 권장, WandB 사용시 "wandb"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "v_proj", 
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        
        # 디렉토리 생성
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("results", exist_ok=True)

@dataclass
class EvaluationConfig:
    # 평가 설정
    eval_batch_size: int = 4
    max_eval_samples: Optional[int] = None
    
    # 메트릭 설정
    compute_metrics: bool = True
    
    # 결과 저장
    results_dir: str = "results"
    
    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)

@dataclass
class InferenceConfig:
    # 추론 설정
    model_path: str = "checkpoints/best"
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = True
    
    # 출력 설정
    output_format: str = "json"  # "json", "text"
    
    def __post_init__(self):
        os.makedirs("inference_results", exist_ok=True)

# 기본 설정 인스턴스
training_config = TrainingConfig()
evaluation_config = EvaluationConfig()
inference_config = InferenceConfig() 