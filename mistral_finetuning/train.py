import os
import json
import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
from typing import Dict, List, Any
import wandb
from config import training_config, TrainingConfig

# 로깅 설정
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class MistralFineTuner:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # GPU 메모리 정보 출력
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
    def setup_model_and_tokenizer(self):
        """모델과 토크나이저를 설정합니다."""
        logger.info("모델과 토크나이저 로딩 중...")
        
        # 4비트 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        
        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델을 kbit 학습용으로 준비
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("모델과 토크나이저 로딩 완료")
        
    def setup_lora(self):
        """LoRA 설정을 적용합니다."""
        logger.info("LoRA 설정 적용 중...")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA 설정 완료")
        
    def load_datasets(self):
        """데이터셋을 로드합니다."""
        logger.info("데이터셋 로딩 중...")
        
        # 데이터셋 로드
        data_files = {
            "train": self.config.train_file,
            "validation": self.config.validation_file,
        }
        
        self.dataset = load_dataset("json", data_files=data_files)
        
        logger.info(f"Train dataset size: {len(self.dataset['train'])}")
        logger.info(f"Validation dataset size: {len(self.dataset['validation'])}")
        
    def tokenize_function(self, examples):
        """텍스트를 토크나이징합니다."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        
    def prepare_datasets(self):
        """데이터셋을 토크나이징하여 준비합니다."""
        logger.info("데이터셋 토크나이징 중...")
        
        # 토크나이징
        tokenized_datasets = self.dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )
        
        self.tokenized_datasets = tokenized_datasets
        logger.info("데이터셋 토크나이징 완료")
        
    def compute_metrics(self, eval_pred):
        """평가 메트릭을 계산합니다."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        
        # 패딩 토큰 제거
        true_predictions = [
            [p for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        
        # 정확도 계산
        correct = 0
        total = 0
        for pred, label in zip(true_predictions, true_labels):
            correct += sum(p == l for p, l in zip(pred, label))
            total += len(label)
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
        }
        
    def setup_training(self):
        """학습 설정을 구성합니다."""
        logger.info("학습 설정 구성 중...")
        
        # 데이터 콜레이터
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 학습 인수
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=True,  # RunPod 메모리 최적화
            logging_dir=self.config.logging_dir,
            report_to=self.config.report_to,
            remove_unused_columns=self.config.remove_unused_columns,
            group_by_length=self.config.group_by_length,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            dataloader_num_workers=self.config.dataloader_num_workers,
            ddp_find_unused_parameters=self.config.ddp_find_unused_parameters,
        )
        
        # 트레이너 초기화
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        logger.info("학습 설정 완료")
        
    def train(self):
        """파인튜닝을 실행합니다."""
        logger.info("파인튜닝 시작...")
        
        # WandB 초기화
        if self.config.report_to == "wandb":
            wandb.init(
                project="mistral-maintenance-parsing",
                name="qlora-finetuning",
                config=vars(self.config)
            )
        
        try:
            # 학습 실행
            train_result = self.trainer.train(
                resume_from_checkpoint=self.config.resume_from_checkpoint
            )
            
            # 최종 모델 저장
            self.trainer.save_model()
            
            # 학습 결과 저장
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            self.trainer.save_state()
            
            logger.info("파인튜닝 완료")
            logger.info(f"Train metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"학습 중 오류 발생: {e}")
            raise
        finally:
            if self.config.report_to == "wandb":
                wandb.finish()
                
    def evaluate(self):
        """모델을 평가합니다."""
        logger.info("모델 평가 중...")
        
        metrics = self.trainer.evaluate()
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        logger.info(f"Evaluation metrics: {metrics}")
        
    def save_model_info(self):
        """모델 정보를 저장합니다."""
        model_info = {
            "model_name": self.config.model_name,
            "training_config": vars(self.config),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
        }
        
        with open(f"{self.config.output_dir}/model_info.json", "w", encoding="utf-8") as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
            
        logger.info("모델 정보 저장 완료")

def main():
    """메인 실행 함수"""
    logger.info("Mistral 파인튜닝 시작")
    
    # 파인튜너 초기화
    finetuner = MistralFineTuner(training_config)
    
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

if __name__ == "__main__":
    main() 