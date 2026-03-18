"""
Joke Generation Model Fine-tuning

LoRA/QLoRA fine-tuning script using Hugging Face transformers + PEFT.
Trains a custom joke generation model from the instruction-tuning dataset.

Key design choices:
- LoRA (Low-Rank Adaptation) for parameter-efficient training
- QLoRA (4-bit quantization) for fitting larger models on consumer GPUs
- Instruction-tuning format for flexible generation at inference
- Early stopping based on validation loss
"""

import os
import json
from typing import Optional, Dict
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig
from loguru import logger

from config import config


class JokeFineTuner:
    """
    LoRA/QLoRA fine-tuner for joke generation.

    Supports:
    - Full precision (fp16/bf16) training
    - 4-bit QLoRA for memory-constrained environments
    - Custom LoRA rank and target modules
    - Wandb/Tensorboard logging
    - Checkpoint saving and resumption

    Typical training time:
    - ~3000 jokes, 3 epochs, QLoRA: ~30 min on A100
    - ~3000 jokes, 3 epochs, QLoRA: ~2 hrs on T4 (Colab)
    """

    def __init__(
        self,
        base_model: Optional[str] = None,
        use_qlora: bool = True,
        lora_r: Optional[int] = None,
        lora_alpha: Optional[int] = None,
    ):
        self.base_model = base_model or config.fine_tuning.base_model
        self.use_qlora = use_qlora
        self.lora_r = lora_r or config.fine_tuning.lora_r
        self.lora_alpha = lora_alpha or config.fine_tuning.lora_alpha
        self.output_dir = config.fine_tuning.output_dir

        self.model = None
        self.tokenizer = None

    def setup(self) -> None:
        """Load base model and tokenizer, apply LoRA configuration."""
        logger.info(f"Loading base model: {self.base_model}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config for QLoRA
        bnb_config = None
        if self.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("Using QLoRA (4-bit quantization)")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not self.use_qlora else None,
        )

        if self.use_qlora:
            self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=config.fine_tuning.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

        self.model = get_peft_model(self.model, lora_config)

        # Log trainable parameters
        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    def train(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run the fine-tuning training loop.

        Args:
            train_data_path: Path to training JSONL file.
            val_data_path: Path to validation JSONL file.
            output_dir: Where to save the fine-tuned model.

        Returns:
            Training metrics dict.
        """
        if self.model is None:
            self.setup()

        output = output_dir or self.output_dir
        os.makedirs(output, exist_ok=True)

        # Load datasets
        train_dataset = self._load_jsonl(train_data_path)
        val_dataset = self._load_jsonl(val_data_path) if val_data_path else None

        logger.info(
            f"Training on {len(train_dataset)} examples"
            + (f", validating on {len(val_dataset)}" if val_dataset else "")
        )

        # Format as text
        train_dataset = train_dataset.map(
            lambda x: {"text": self._format_example(x)},
            remove_columns=train_dataset.column_names,
        )
        if val_dataset:
            val_dataset = val_dataset.map(
                lambda x: {"text": self._format_example(x)},
                remove_columns=val_dataset.column_names,
            )

        # Training arguments
        training_args = SFTConfig(
            output_dir=output,
            num_train_epochs=config.fine_tuning.num_epochs,
            per_device_train_batch_size=config.fine_tuning.batch_size,
            gradient_accumulation_steps=config.fine_tuning.gradient_accumulation_steps,
            learning_rate=config.fine_tuning.learning_rate,
            warmup_ratio=config.fine_tuning.warmup_ratio,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if val_dataset else "no",
            bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
            max_seq_length=config.fine_tuning.max_seq_length,
            report_to="none",  # Set to "wandb" for experiment tracking
            seed=42,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
        )

        # Trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Save model
        trainer.save_model(output)
        self.tokenizer.save_pretrained(output)
        logger.info(f"Model saved to {output}")

        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        }

    def generate(self, prompt: str, max_new_tokens: int = 200) -> str:
        """Generate a joke using the fine-tuned model."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call setup() or load() first.")

        formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        if "<|assistant|>" in full_text:
            return full_text.split("<|assistant|>")[-1].strip()
        return full_text[len(formatted):].strip()

    def load(self, model_path: Optional[str] = None) -> None:
        """Load a previously fine-tuned model."""
        path = model_path or self.output_dir
        logger.info(f"Loading fine-tuned model from {path}")

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    def _format_example(self, example: Dict) -> str:
        """Format a training example as instruction-response text."""
        return (
            f"<|user|>\n{example['instruction']}\n"
            f"<|assistant|>\n{example['output']}\n"
        )

    @staticmethod
    def _load_jsonl(path: str) -> Dataset:
        """Load a JSONL file into a HuggingFace Dataset."""
        return load_dataset("json", data_files=path, split="train")
