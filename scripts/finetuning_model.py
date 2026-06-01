import math
import os
import json
import torch
import random
import contextlib
from glob import glob
from torch.optim import Optimizer
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from peft.peft_model import PeftModel
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_finetuning_model_and_tokenizer(args):
    if args.model == "dnabert2":
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model = DNABert2FineTuningModel(lora_config=lora_config)
        tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
    elif args.model == "nucleotide_transformer":
        model = NucleotideTransformerFineTuningModel()
        tokenizer = AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True
        )
    elif args.model == "hyenadna":
        model = HyenaDNAFineTuningModel()
        tokenizer = AutoTokenizer.from_pretrained("LongSafari/hyenadna-large-1m-seqlen-hf", trust_remote_code=True)
        tokenizer.padding_side = "left"
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model, tokenizer

class FineTuningModel(ABC):
    @abstractmethod
    def load_checkpoint(self, device: torch.device, weights_dir: str | None = None):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def save_pretrained(self, output_dir: str, tokenizer=None):
        pass

    @abstractmethod
    def get_optimizer(self, optimizer_params) -> Optimizer:
        pass

    @abstractmethod
    def get_scheduler(self, optimizer, scheduler_params) -> LRScheduler:
        pass

    @abstractmethod
    def get_scaler(self) -> GradScaler:
        pass

class DNABert2FineTuningModel(FineTuningModel):
    def __init__(self, lora_config=None):
        from bend.models.dnabert2 import BertForMaskedLM as DNABert2BertForMaskedLM
        import bend.models.dnabert2 as dnabert2_module

        with contextlib.suppress(Exception):
            dnabert2_module.flash_attn_qkvpacked_func = None
            self.base_model_name = "zhihan1996/DNABERT-2-117M"
            self.model = DNABert2BertForMaskedLM.from_pretrained(self.base_model_name, trust_remote_code=True)
        if lora_config is not None:
            self.model = get_peft_model(self.model, lora_config)

    def load_checkpoint(self, device: torch.device, weights_dir: str | None = None):
        from bend.models.dnabert2 import BertForMaskedLM as DNABert2BertForMaskedLM

        print(f"Loading base model from {self.base_model_name} (trust_remote_code=True)")
        model = DNABert2BertForMaskedLM.from_pretrained(self.base_model_name, trust_remote_code=True)
        if weights_dir is not None:
            print(f"Applying LoRA adapter from {weights_dir}")
            model = PeftModel.from_pretrained(model, weights_dir, is_trainable=True)
        if device is not None:
            model = model.to(device)
        model.train()
        self.model = model
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError(
                f"Loaded model from {weights_dir} has no trainable parameters. "
                "Did you load a merged/unloaded checkpoint? Only unmerged LoRA adapters are trainable."
            )
        return model

    def save_pretrained(self, output_dir: str, tokenizer=None):
        self.model.save_pretrained(output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

    def get_model(self):
        return self.model

    def get_optimizer(self, optimizer_params=None) -> Optimizer:
        optimizer_params = optimizer_params or {}
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_params.get("lr", 5e-4),
            betas=optimizer_params.get("betas", (0.9, 0.98)),
            eps=optimizer_params.get("eps", 1e-6),
            weight_decay=optimizer_params.get("weight_decay", 1e-5),
        )

    def get_scheduler(self, optimizer, scheduler_params):
        scheduler_params = scheduler_params or {}
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_params.get("warmup_steps", 100),
            num_training_steps=scheduler_params.get("total_steps", 10000),
        )

    def get_scaler(self) -> GradScaler:
        return GradScaler()

class NucleotideTransformerFineTuningModel(FineTuningModel):
    def __init__(self):
        self.base_model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
        self.model = AutoModelForMaskedLM.from_pretrained(self.base_model_name, trust_remote_code=True)

    def load_checkpoint(self, device: torch.device, weights_dir: str | None = None):
        from transformers import AutoConfig

        src = weights_dir or self.base_model_name
        print(f"Loading model from {src}")

        cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
        if getattr(cfg, "model_type", None) != "esm":
            raise ValueError(
                f"Expected esm config (NT-v2), got {getattr(cfg, 'model_type', None)} at {src}. "
                f"This suggests DNABERT-2 config contamination."
            )

        model = AutoModelForMaskedLM.from_pretrained(src, config=cfg, trust_remote_code=True)

        if device is not None:
            model = model.to(device)
        model.train()
        self.model = model

        if not any(p.requires_grad for p in model.parameters()):
            print("WARNING: No trainable parameters found in the model.")

        return model

    def save_pretrained(self, output_dir: str, tokenizer=None):
        import shutil

        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        self.model.save_pretrained(output_dir, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        print(f"Saved NT-v2 model (and tokenizer if provided) to {output_dir}")

    def get_model(self):
        return self.model

    def get_optimizer(self, optimizer_params) -> Optimizer:
        optimizer_params = optimizer_params or {}
        return torch.optim.Adam(
            self.model.parameters(),
            lr=optimizer_params.get("lr", 1e-4),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=optimizer_params.get("weight_decay", 0.0),
        )

    def get_scheduler(self, optimizer, scheduler_params) -> LRScheduler:
        scheduler_params = scheduler_params or {}
        warmup_steps = scheduler_params.get("warmup_steps", 100)
        total_steps = scheduler_params.get("total_steps", 1000)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return 0.5 + 0.5 * (current_step / warmup_steps)
            else:
                return (warmup_steps**0.5) / max(current_step, 1) ** 0.5

        return LambdaLR(optimizer, lr_lambda)

    def get_scaler(self) -> GradScaler:
        return GradScaler()

class HyenaDNAFineTuningModel(FineTuningModel):
    def __init__(self):
        self.base_model_name = "LongSafari/hyenadna-large-1m-seqlen-hf"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        with contextlib.suppress(Exception):
            self.model.gradient_checkpointing_enable()

    def load_checkpoint(self, device: torch.device, weights_dir: str | None = None):
        """
        Loads the 1M HyenaDNA CausalLM with custom HF code.
        Verifies config to guard against contamination.
        Optionally loads from a fine-tuned directory (weights_dir).
        """
        src = weights_dir or self.base_model_name
        print(f"Loading HyenaDNA (1M) from {src} (trust_remote_code=True)")

        cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
        if cfg.model_type != "hyenadna":
            raise ValueError(f"Expected hyenadna config, got {cfg.model_type} at {src}.")

        model = AutoModelForCausalLM.from_pretrained(
            src,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        with contextlib.suppress(Exception):
            model.gradient_checkpointing_enable()
        if device is not None:
            model = model.to(device)
        model.train()
        self.model = model

        if not any(p.requires_grad for p in model.parameters()):
            print("WARNING: No trainable parameters found in HyenaDNA model.")

        return model

    def save_pretrained(self, output_dir: str, tokenizer=None):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

    def get_model(self):
        return self.model

    def get_optimizer(self, optimizer_params=None) -> Optimizer:
        optimizer_params = optimizer_params or {}
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_params.get("lr", 6e-4),
            betas=optimizer_params.get("betas", (0.9, 0.999)),
            eps=optimizer_params.get("eps", 1e-8),
            weight_decay=optimizer_params.get("weight_decay", 0.1),
        )

    def get_scheduler(self, optimizer, scheduler_params=None) -> LRScheduler:
        scheduler_params = scheduler_params or {}
        warmup_steps = scheduler_params.get("warmup_steps", 500)
        total_steps = scheduler_params.get("total_steps", 10_000)
        
        def lr_lambda(current_step: int) -> float:
            if current_step <= warmup_steps:
                return float(current_step / warmup_steps)
            
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            progress = min(progress, 1.0)
            eta_min_ratio = 1.5e-4 / 6e-4
            cosine_factor = 0.5 * (1 + math.cos(progress * math.pi))
            return float(eta_min_ratio + (1.0 - eta_min_ratio) * cosine_factor)
                
        return LambdaLR(optimizer, lr_lambda)

    def get_scaler(self) -> GradScaler:
        return GradScaler(enabled=False)

class CurriculumMLMDataset(Dataset):
    def __init__(self, data_dir: str, split: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.data_dir = data_dir

        class_mapping_path = os.path.join(data_dir, "class_mapping.json")
        with open(class_mapping_path, "r") as f:
            self.class_mapping = json.load(f)
        self.num_classes = len(self.class_mapping)

        file_pattern = os.path.join(data_dir, f"{split}_class_*.json")
        self.class_files = sorted(glob(file_pattern))
        if not self.class_files:
            raise ValueError(f"No class files found matching pattern: {file_pattern}")

        self.current_class_idx = 0
        self.load_current_class()

    def load_current_class(self):
        with open(self.class_files[self.current_class_idx], "r") as f:
            self.current_data = json.load(f)
        self.current_class = os.path.basename(self.class_files[self.current_class_idx]).split("_")[2].split(".")[0]

    def next_class(self):
        if self.current_class_idx + 1 < len(self.class_files):
            self.current_class_idx += 1
            self.load_current_class()
            return True
        return False

    def reset_to_first_class(self):
        self.current_class_idx = 0
        self.load_current_class()

    def __len__(self):
        return len(self.current_data)

    def __getitem__(self, idx):
        item = self.current_data[idx]
        sequence = item["sequence"]

        if len(sequence) > self.max_length:
            start_idx = random.randint(0, len(sequence) - self.max_length)
            sequence = sequence[start_idx : start_idx + self.max_length]

        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding.get("attention_mask", torch.ones_like(input_ids)).squeeze()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)

        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            input_ids.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class CurriculumCausalLMDataset(Dataset):
    """
    Curriculum dataset for causal language modeling (next token prediction).
    Suitable for models like HyenaDNA that use causal LM objectives.
    """

    def __init__(self, data_dir: str, split: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.data_dir = data_dir

        class_mapping_path = os.path.join(data_dir, "class_mapping.json")
        with open(class_mapping_path, "r") as f:
            self.class_mapping = json.load(f)
        self.num_classes = len(self.class_mapping)

        file_pattern = os.path.join(data_dir, f"{split}_class_*.json")
        self.class_files = sorted(glob(file_pattern))
        if not self.class_files:
            raise ValueError(f"No class files found matching pattern: {file_pattern}")

        self.current_class_idx = 0
        self.load_current_class()

    def load_current_class(self):
        with open(self.class_files[self.current_class_idx], "r") as f:
            self.current_data = json.load(f)
        self.current_class = os.path.basename(self.class_files[self.current_class_idx]).split("_")[2].split(".")[0]

    def next_class(self):
        if self.current_class_idx + 1 < len(self.class_files):
            self.current_class_idx += 1
            self.load_current_class()
            return True
        return False

    def reset_to_first_class(self):
        self.current_class_idx = 0
        self.load_current_class()

    def __len__(self):
        return len(self.current_data)

    def __getitem__(self, idx):
        item = self.current_data[idx]
        sequence = item["sequence"]

        if len(sequence) > self.max_length:
            start_idx = random.randint(0, len(sequence) - self.max_length)
            sequence = sequence[start_idx : start_idx + self.max_length]

        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding.get("attention_mask", torch.ones_like(input_ids)).squeeze()

        labels = input_ids.clone()

        labels[input_ids == self.tokenizer.pad_token_id] = -100

        if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id is not None:
            labels[input_ids == self.tokenizer.bos_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
