import os
import json
import torch
import argparse
import contextlib
from tqdm import tqdm
from typing import Union
from dataclasses import dataclass
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import LRScheduler
from scripts.finetuning_model import (
    FineTuningModel,
    CurriculumMLMDataset,
    CurriculumCausalLMDataset,
    get_finetuning_model_and_tokenizer,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_curriculum_dataset_class(model_name: str):
    """Choose the appropriate curriculum dataset class based on model type."""
    if "hyena" in model_name.lower():
        return CurriculumCausalLMDataset
    else:
        return CurriculumMLMDataset


def get_autocast_dtype(model_name: str):
    if device.type == "cpu":
        return torch.bfloat16
    return torch.bfloat16 if "hyena" in model_name.lower() else torch.float16


def model_forward(model, input_ids, attention_mask, labels, model_name: str):
    """Handle model forward pass based on model type."""
    if "hyena" in model_name.lower():
        return model(input_ids=input_ids, labels=labels)
    else:
        return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def train_epoch(
    model: PreTrainedModel,
    train_dataset: Union[CurriculumMLMDataset, CurriculumCausalLMDataset],
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: GradScaler,
    batch_size: int,
    model_name: str,
    rank: str = "",
    epoch: int = 0,
    epochs: int = 0,
) -> float:
    model.train()
    total_train_loss = 0
    total_batches = 0
    train_dataset.reset_to_first_class()
    num_classes = len(train_dataset.class_files)

    desc = f"[{rank}] Epoch {epoch+1}/{epochs}"
    for _ in tqdm(range(num_classes), desc=desc, leave=True):
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, persistent_workers=False
        )
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            autocast_dtype = get_autocast_dtype(model_name)
            with autocast(device_type=device.type, dtype=autocast_dtype):
                outputs = model_forward(model, input_ids, attention_mask, labels, model_name)
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


def validate(
    model: PreTrainedModel,
    val_dataset: Union[CurriculumMLMDataset, CurriculumCausalLMDataset],
    batch_size: int,
    model_name: str,
) -> float:
    model.eval()
    total_val_loss = 0
    total_batches = 0
    val_dataset.reset_to_first_class()
    num_classes = len(val_dataset.class_files)

    with torch.no_grad():
        for _ in range(num_classes):
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
            )
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with autocast(device_type=device.type, dtype=get_autocast_dtype(model_name)):
                    outputs = model_forward(model, input_ids, attention_mask, labels, model_name)

                total_val_loss += outputs.loss.item()
                total_batches += 1
            if not val_dataset.next_class():
                break

    return total_val_loss / total_batches if total_batches > 0 else 0.0


def log_epoch_json(results_path: str, rank: str, num_classes: int, training_history: list[dict]) -> None:
    backup_path = os.path.join("/tmp", f"backup_{os.path.basename(results_path)}")

    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}
    else:
        results = {}

    if not results and os.path.exists(backup_path):
        try:
            with open(backup_path, "r") as f:
                results = json.load(f)
        except Exception:
            results = {}

    results[rank] = {
        "rank": rank,
        "num_classes": num_classes,
        "training_history": training_history,
    }

    try:
        with open(backup_path, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"WARNING: Failed to write backup JSON to {backup_path}: {e}")

    try:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"WARNING: Failed to write JSON to {results_path}: {e}")
        print(f"Data is backed up in {backup_path}")


def fine_tune_curriculum_stage(
    fine_tuning_model: FineTuningModel,
    tokenizer,
    dataset_dir: str,
    rank: str,
    output_dir: str,
    model_name: str,
    epochs: int = 10,
    batch_size: int = 32,
    warmup_pct: float = 0.1,
    patience: int = 5,
    min_epoch: int = 10,
    max_length: int = 128,
) -> tuple[dict, PreTrainedModel]:
    rank_dir = os.path.join(dataset_dir, rank)

    DatasetClass = get_curriculum_dataset_class(model_name)
    train_dataset = DatasetClass(rank_dir, "train", tokenizer, max_length=max_length)
    val_dataset = DatasetClass(rank_dir, "val", tokenizer, max_length=max_length)
    test_dataset = DatasetClass(rank_dir, "test", tokenizer, max_length=max_length)

    model = fine_tuning_model.get_model()
    model.to(device)

    total_batches_per_epoch = 0
    for class_file in train_dataset.class_files:
        with open(class_file, "r") as f:
            class_data = json.load(f)
            total_batches_per_epoch += (len(class_data) + batch_size - 1) // batch_size

    total_steps = total_batches_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_pct))

    optimizer = fine_tuning_model.get_optimizer({})
    scheduler = fine_tuning_model.get_scheduler(optimizer, {"warmup_steps": warmup_steps, "total_steps": total_steps})
    scaler = fine_tuning_model.get_scaler()

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch = 0
    training_history = []

    results_path = os.path.join(output_dir, "results.json")
    print(
        f"Starting training for rank: {rank} ({train_dataset.num_classes} classes, {epochs} epochs), patience={patience}"
    )
    epochs_without_improvement = 0
    for epoch in range(epochs):
        print(f"[{rank}] Starting epoch {epoch+1}/{epochs}...")
        avg_train_loss = train_epoch(
            model, train_dataset, optimizer, scheduler, scaler, batch_size, model_name, rank, epoch, epochs
        )
        avg_val_loss = validate(model, val_dataset, batch_size, model_name)
        training_history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
        )
        log_epoch_json(results_path, rank, train_dataset.num_classes, training_history)
        print(f"[{rank}] Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_path = os.path.join(output_dir, "best_model")
            backup_model_path = os.path.join("/tmp", f"backup_best_model_{rank}")

            try:
                fine_tuning_model.save_pretrained(backup_model_path, tokenizer)
            except Exception as e:
                print(f"WARNING: Failed to save backup model to {backup_model_path}: {e}")

            try:
                fine_tuning_model.save_pretrained(best_model_path, tokenizer)
            except Exception as e:
                print(f"WARNING: Failed to save model to {best_model_path}: {e}")
                print(f"Model is backed up in {backup_model_path}")

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"[{rank}] No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience and (epoch + 1) >= min_epoch:
                print(f"[{rank}] Early stopping triggered after {epoch+1} epochs.")
                break

    avg_test_loss = validate(model, test_dataset, batch_size, model_name)

    results = {
        "rank": rank,
        "num_classes": train_dataset.num_classes,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": avg_test_loss,
        "training_history": training_history,
    }

    try:
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"WARNING: Failed to save final results to {output_dir}/results.json: {e}")
        backup_results_path = os.path.join("/tmp", f"backup_results_{rank}.json")
        try:
            with open(backup_results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results backed up to {backup_results_path}")
        except Exception as e2:
            print(f"ERROR: Failed to save backup results: {e2}")

    return results, model


@dataclass
class CurriculumConfig:
    dataset_dir: str
    output_base_dir: str
    final_model_dir: str
    epochs_per_stage: int
    batch_size: int
    model: str
    ranks: list[str] | None
    warmup_pct: float
    patience: int
    max_length: int
    min_epoch: int = 10
    init_from_rank: str | None = None
    start_from_rank: str | None = None

    def __post_init__(self):
        if self.ranks is None:
            self.ranks = ["phylum", "class", "order", "family", "genus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="curriculum_datasets")
    parser.add_argument("--output_base_dir", type=str, default="fine_tuned_models")
    parser.add_argument("--final_model_dir", type=str, default="dnabert2_curriculum_lora_minimal")
    parser.add_argument("--epochs_per_stage", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_pct", type=float, default=0.4, help="Fraction of total steps for LR warmup (0-1)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min_epoch", type=int, default=10, help="Minimum number of epochs before early stopping")
    parser.add_argument(
        "--max_length",
        type=int,
        default=1000,
        help="Maximum sequence length for tokenization (NT-2.5B supports up to 1000 tokens)",
    )
    parser.add_argument(
        "--ranks",
        type=str,
        nargs="+",
        default=["phylum", "class", "order", "family", "genus"],
        help="List of taxonomic ranks for curriculum stages",
    )
    parser.add_argument(
        "--init_from_rank",
        type=str,
        default=None,
        help="Rank whose best_model checkpoint to load before (re)starting training",
    )
    parser.add_argument(
        "--start_from_rank",
        type=str,
        default=None,
        help="Rank at which to start/resume training (earlier ranks will be skipped)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model type to use (e.g. dnabert2)",
    )
    return parser.parse_args()


def epoch_to_check_checkpoints(fine_tuning_model, tokenizer, dataset_dir, rank, output_dir, model_name, max_length=128):
    print("Running simple test_epoch...")
    DatasetClass = get_curriculum_dataset_class(model_name)
    test_dataset = DatasetClass(os.path.join(dataset_dir, rank), "train", tokenizer, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

    model = fine_tuning_model.get_model()
    model.to(device)
    optimizer = fine_tuning_model.get_optimizer({})
    scaler = fine_tuning_model.get_scaler()
    model.train()

    original_params = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
    print(f"Stored {len(original_params)} trainable parameters for comparison")

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        autocast_dtype = get_autocast_dtype(model_name)
        with autocast(device_type=device.type, dtype=autocast_dtype):
            outputs = model_forward(model, input_ids, attention_mask, labels, model_name)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"Training loss: {loss.item():.6f}")
        break

    param_changes = {}
    for name, param in model.named_parameters():
        if name in original_params:
            diff = torch.norm(param.data - original_params[name]).item()
            param_changes[name] = diff

    total_change = sum(param_changes.values())
    print(f"Total parameter change from training: {total_change:.6f}")

    if total_change < 1e-8:
        print("WARNING: Parameters barely changed during training!")
    else:
        print("✓ Parameters changed during training as expected")

    fine_tuning_model.save_pretrained(output_dir, tokenizer)

    trained_params = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
    fine_tuning_model.load_checkpoint(weights_dir=output_dir, device=device)
    model = fine_tuning_model.get_model()
    model.to(device)

    matches = 0
    total_params = 0
    max_diff = 0.0

    for name, param in model.named_parameters():
        if name in trained_params:
            diff = torch.norm(param.data - trained_params[name]).item()
            max_diff = max(max_diff, diff)
            if diff < 1e-6:
                matches += 1
            total_params += 1

    print(f"Parameter verification: {matches}/{total_params} parameters match")
    print(f"Max parameter difference: {max_diff:.10f}")

    if matches == total_params and max_diff < 1e-5:
        print("✓ VERIFIED: Loaded model matches the fine-tuned model!")
    else:
        print("❌ WARNING: Loaded model does NOT match the fine-tuned model!")
        return

    optimizer = fine_tuning_model.get_optimizer({})
    scaler = fine_tuning_model.get_scaler()
    model.train()

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        autocast_dtype = get_autocast_dtype(model_name)
        with autocast(device_type=device.type, dtype=autocast_dtype):
            outputs = model_forward(model, input_ids, attention_mask, labels, model_name)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print(f"Second training loss: {loss.item():.6f}")
        break

    print("test_epoch completed successfully.")


def main():
    args = parse_args()
    config = CurriculumConfig(**vars(args))
    output_config_dir = os.path.join(config.output_base_dir, config.final_model_dir)
    try:
        with contextlib.suppress(Exception):
            backup_path = os.path.join("/tmp", f"backup_run_config_{os.path.basename(output_config_dir)}.json")
            with open(backup_path, "w") as bf:
                json.dump(vars(config), bf, indent=2)
        os.makedirs(output_config_dir, exist_ok=True)
        config_path = os.path.join(output_config_dir, "run_config.json")
        with open(config_path, "w") as cf:
            json.dump(vars(config), cf, indent=2)
    except Exception as e:
        print(f"WARNING: Failed to write run config to {output_config_dir}: {e}")

    fine_tuning_model, tokenizer = get_finetuning_model_and_tokenizer(args)
    epoch_to_check_checkpoints(
        fine_tuning_model=fine_tuning_model,
        tokenizer=tokenizer,
        dataset_dir=config.dataset_dir,
        rank=(config.ranks or ["phylum"])[0],
        output_dir=os.path.join(config.output_base_dir, config.final_model_dir, "test_epoch"),
        model_name=config.model,
        max_length=config.max_length,
    )

    fine_tuning_model, tokenizer = get_finetuning_model_and_tokenizer(args)
    model = fine_tuning_model.get_model()
    model.to(device)

    all_results = {}
    ranks = config.ranks or ["phylum", "class", "order", "family", "genus"]

    if args.start_from_rank is not None:
        if args.start_from_rank not in ranks:
            print(f"ERROR: start_from_rank '{args.start_from_rank}' not in ranks list {ranks}")
            return
        start_index = ranks.index(args.start_from_rank)
        if start_index > 0:
            print(f"Resuming curriculum from rank '{args.start_from_rank}' (skipping: {ranks[:start_index]})")
        ranks = ranks[start_index:]

    if args.init_from_rank is not None:
        checkpoint_dir = os.path.join(config.output_base_dir, config.final_model_dir, f"curriculum_stage_{args.init_from_rank}", "best_model")
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f"init_from_rank '{args.init_from_rank}' checkpoint not found at {checkpoint_dir}.")
        try:
            print(f"Loading initial weights from rank '{args.init_from_rank}' checkpoint: {checkpoint_dir}")
            fine_tuning_model.load_checkpoint(weights_dir=checkpoint_dir, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load init_from_rank checkpoint: {e}") from e

    for rank in ranks:
        stage_output_dir = os.path.join(config.output_base_dir, config.final_model_dir, f"curriculum_stage_{rank}")
        try:
            results, model = fine_tune_curriculum_stage(
                fine_tuning_model=fine_tuning_model,
                tokenizer=tokenizer,
                dataset_dir=config.dataset_dir,
                rank=rank,
                output_dir=stage_output_dir,
                model_name=config.model,
                epochs=config.epochs_per_stage,
                batch_size=config.batch_size,
                warmup_pct=args.warmup_pct,
                patience=config.patience,
                min_epoch=args.min_epoch,
                max_length=config.max_length,
            )
            all_results[rank] = results
        except Exception as e:
            print(f"Failed to train {rank} stage: {e}")
            continue

    final_model_path = os.path.join(config.output_base_dir, config.final_model_dir)
    backup_final_path = os.path.join("/tmp", f"backup_{config.final_model_dir}")

    while True:
        try:
            os.makedirs(final_model_path, exist_ok=True)
            fine_tuning_model.save_pretrained(final_model_path, tokenizer)

            with open(os.path.join(final_model_path, "curriculum_results.json"), "w") as f:
                json.dump(all_results, f, indent=2)

            print(f"Final model and results saved to: {final_model_path}")
            break

        except Exception as e:
            print(f"ERROR: Failed to save final model to {final_model_path}: {e}")

            try:
                os.makedirs(backup_final_path, exist_ok=True)
                fine_tuning_model.save_pretrained(backup_final_path, tokenizer)

                with open(os.path.join(backup_final_path, "curriculum_results.json"), "w") as f:
                    json.dump(all_results, f, indent=2)

                print(f"Final model and results backed up to: {backup_final_path}")

            except Exception as e2:
                print(f"ERROR: Failed to save backup final model: {e2}")

            print("\nCRITICAL: Could not save final model to either main or backup location!")
            new_path = input("Please provide an alternative path to save the final model (or 'quit' to exit): ").strip()

            if new_path.lower() == "quit":
                print("WARNING: Exiting without saving final model!")
                break

            final_model_path = new_path
    for rank, results in all_results.items():
        print(
            f"{rank.upper()}: Best Val Loss = {results['best_val_loss']:.4f}, "
            f"Test Loss = {results['test_loss']:.4f}, Classes = {results['num_classes']}"
        )


if __name__ == "__main__":
    main()
