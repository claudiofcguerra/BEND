import os
import json
import torch
import random
import argparse
from glob import glob
from dataclasses import dataclass
from torch.optim import Optimizer
from abc import ABC, abstractmethod
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from peft import LoraConfig, get_peft_model, LoraConfig, get_peft_model
from bend.models.dnabert2 import BertForMaskedLM as DNABert2BertForMaskedLM
from transformers import AutoTokenizer, PreTrainedModel, get_linear_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FineTuningModel(ABC):
    @abstractmethod
    def get_model(self) -> PreTrainedModel:
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
        try:
            import bend.models.dnabert2 as dnabert2_module

            dnabert2_module.flash_attn_qkvpacked_func = None
        except:
            pass
        self.model = DNABert2BertForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M")
        if lora_config is not None:
            self.model = get_peft_model(self.model, lora_config)

    def get_model(self) -> PreTrainedModel:
        return self.model

    def get_optimizer(self, optimizer_params=None) -> Optimizer:
        optimizer_params = optimizer_params or {}
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_params.get("lr", 3e-5),
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
























def custom_warn(*args, **kwargs):
    pass


import warnings as _warnings

_warnings.warn = custom_warn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FineTuningModel(ABC):
    @abstractmethod
    def get_model(self) -> PreTrainedModel:
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
        self.model = DNABert2BertForMaskedLM.from_pretrained("zhihan1996/DNABERT-2-117M")
        if lora_config is not None:
            self.model = get_peft_model(self.model, lora_config)

    def get_model(self) -> PreTrainedModel:
        return self.model

    def get_optimizer(self, optimizer_params=None) -> Optimizer:
        optimizer_params = optimizer_params or {}
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_params.get("lr", 3e-5),
            betas=optimizer_params.get("betas", (0.9, 0.98)),
            eps=optimizer_params.get("eps", 1e-6),
            weight_decay=optimizer_params.get("weight_decay", 1e-5),
        )

    def get_scheduler(self, optimizer, scheduler_params):
        scheduler_params = scheduler_params or {}
        from transformers import get_linear_schedule_with_warmup

        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_params.get("warmup_steps", 100),
            num_training_steps=scheduler_params.get("total_steps", 10000),
        )

    def get_scaler(self) -> GradScaler:
        return GradScaler()


class CurriculumMLMDataset(Dataset):
    """Dataset that loads sequences by class for curriculum MLM learning"""

    def __init__(self, data_dir: str, split: str, tokenizer, max_length: int = 512):
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

        logging.info(f"Found {len(self.class_files)} classes for {split} split")

    def load_current_class(self):
        """Load data for current class"""
        with open(self.class_files[self.current_class_idx], "r") as f:
            self.current_data = json.load(f)
        self.current_class = os.path.basename(self.class_files[self.current_class_idx]).split("_")[2].split(".")[0]

    def next_class(self):
        """Switch to next class, returns True if successful"""
        if self.current_class_idx + 1 < len(self.class_files):
            self.current_class_idx += 1
            self.load_current_class()
            return True
        return False

    def reset_to_first_class(self):
        """Reset to first class"""
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

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_epoch(
    model,
    train_dataset,
    optimizer,
    scheduler,
    scaler,
    batch_size,
    rank_name=None,
    epoch=None,
    total_epochs=None,
    total_ranks=None,
    rank_idx=None,
):
    """Train for one epoch with simplified logging"""
    model.train()
    total_train_loss = 0
    total_batches = 0

    train_dataset.reset_to_first_class()
    num_classes = len(train_dataset.class_files)

    for class_idx in range(num_classes):
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            total_batches += 1

        if not train_dataset.next_class():
            break

    return total_train_loss / total_batches if total_batches > 0 else 0.0


def validate(model, val_dataset, batch_size):
    """Validate model with simplified logging"""
    model.eval()
    total_val_loss = 0
    total_batches = 0

    val_dataset.reset_to_first_class()
    num_classes = len(val_dataset.class_files)
    for class_idx in range(num_classes):
        class_file = val_dataset.class_files[class_idx]
        with open(class_file, "r") as f:
            class_data = json.load(f)
        num_seqs = len(class_data)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            total_val_loss += outputs.loss.item()
            total_batches += 1
        if not val_dataset.next_class():
            break

    return total_val_loss / total_batches if total_batches > 0 else 0.0


def fine_tune_curriculum_stage(
    model,
    tokenizer,
    dataset_dir: str,
    rank: str,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    total_ranks: int = 5,
    rank_idx: int = 0,
):
    """Fine-tune DNABERT2 for one curriculum stage (rank) using MLM"""

    logging.info(f"Training stage: {rank}")

    rank_dir = os.path.join(dataset_dir, rank)

    train_dataset = CurriculumMLMDataset(rank_dir, "train", tokenizer)
    val_dataset = CurriculumMLMDataset(rank_dir, "val", tokenizer)
    test_dataset = CurriculumMLMDataset(rank_dir, "test", tokenizer)

    num_classes = train_dataset.num_classes
    logging.info(f"Number of classes for {rank}: {num_classes}")

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_batches_per_epoch = 0
    for class_file in train_dataset.class_files:
        with open(class_file, "r") as f:
            class_data = json.load(f)
            total_batches_per_epoch += (len(class_data) + batch_size - 1) // batch_size

    total_steps = total_batches_per_epoch * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    scaler = GradScaler()

    os.makedirs(output_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    training_history = []

    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch+1}/{epochs} for rank {rank}")

        avg_train_loss = train_epoch(
            model,
            train_dataset,
            optimizer,
            scheduler,
            scaler,
            batch_size,
            rank_name=rank,
            epoch=epoch,
            total_epochs=epochs,
            total_ranks=total_ranks,
            rank_idx=rank_idx,
        )

        avg_val_loss = validate(model, val_dataset, batch_size)

        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
        }
        training_history.append(epoch_stats)

        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1

            best_model_path = os.path.join(output_dir, "best_model")
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)

            logging.info(f"New best model for {rank} at epoch {epoch+1} with val loss: {avg_val_loss:.4f}")

    avg_test_loss = validate(model, test_dataset, batch_size)

    results = {
        "rank": rank,
        "num_classes": num_classes,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": avg_test_loss,
        "training_history": training_history,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Stage {rank} complete! Best val loss: {best_val_loss:.4f}, Test loss: {avg_test_loss:.4f}")

    return results, model


@dataclass
class CurriculumConfig:
    dataset_dir: str
    output_base_dir: str
    final_model_dir: str
    epochs_per_stage: int = 10
    batch_size: int = 16
    learning_rate: float = 3e-5
    ranks: list[str] | None = None

    def __post_init__(self):
        if self.ranks is None:
            self.ranks = ["phylum", "class", "order", "family", "genus"]
        logging.info(f"Curriculum configuration: {self}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../metagenomic-language-models/curriculum_datasets",
        help="Path to curriculum datasets",
    )
    parser.add_argument("--output_base_dir", type=str, default="fine_tuned_models", help="Base output directory")
    parser.add_argument(
        "--final_model_dir", type=str, default="dnabert2_curriculum_lora_reversed", help="Final model subdirectory"
    )
    parser.add_argument("--epochs_per_stage", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    return parser.parse_args()


async def main():
    """Main curriculum learning pipeline - single model through all ranks with MLM"""

    try:
        from bend.models.dnabert2 import flash_attn_qkvpacked_func
        import bend.models.dnabert2 as dnabert2_module

        dnabert2_module.flash_attn_qkvpacked_func = None
        logging.info("Disabled flash attention for DNABERT2")
    except ImportError:
        logging.warning("Could not import flash attention function - it may already be disabled")

    args = parse_args()
    config = CurriculumConfig(**vars(args))

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )

    fine_tuning_model = DNABert2FineTuningModel(lora_config=lora_config)
    model = fine_tuning_model.get_model()
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")

    model.to(device)

    all_results = {}
    ranks = config.ranks or ["phylum", "class", "order", "family", "genus"]

    for rank_idx, rank in enumerate(ranks):
        logging.info(f"\n{'='*60}")
        logging.info(f"CURRICULUM STAGE: {rank.upper()} [{rank_idx+1}/{len(ranks)}]")
        logging.info(f"{'='*60}")

        stage_output_dir = os.path.join(config.output_base_dir, f"curriculum_stage_{rank}")

        try:
            results, model = fine_tune_curriculum_stage(
                model=model,
                tokenizer=tokenizer,
                dataset_dir=config.dataset_dir,
                rank=rank,
                output_dir=stage_output_dir,
                epochs=config.epochs_per_stage,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                total_ranks=len(ranks),
                rank_idx=rank_idx,
            )
            all_results[rank] = results

        except Exception as e:
            logging.error(f"Failed to train {rank} stage: {e}")
            continue

    final_model_path = os.path.join(config.output_base_dir, config.final_model_dir)
    os.makedirs(final_model_path, exist_ok=True)

    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    with open(os.path.join(final_model_path, "curriculum_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    logging.info("\n" + "=" * 60)
    logging.info("CURRICULUM LEARNING COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Final model saved to: {final_model_path}")

    for rank, results in all_results.items():
        logging.info(
            f"{rank.upper()}: Best Val Loss = {results['best_val_loss']:.4f}, "
            f"Test Loss = {results['test_loss']:.4f}, Classes = {results['num_classes']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
