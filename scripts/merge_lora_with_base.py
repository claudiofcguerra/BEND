import argparse
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer
from bend.models.dnabert2 import BertForMaskedLM


def merge_lora_with_base(base_model_name: str, lora_weights_dir: str, output_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = BertForMaskedLM.from_pretrained(base_model_name, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_weights_dir)

    model = model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Merged model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base DNABERT2 model.")
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="zhihan1996/DNABERT-2-117M",
        help="HuggingFace model name or path for the base DNABERT2 model.",
    )
    parser.add_argument(
        "--lora_weights_dir", type=str, required=True, help="Directory containing the LoRA adapter weights."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the merged model and tokenizer."
    )
    args = parser.parse_args()

    merge_lora_with_base(
        base_model_name=args.base_model_name, lora_weights_dir=args.lora_weights_dir, output_dir=args.output_dir
    )