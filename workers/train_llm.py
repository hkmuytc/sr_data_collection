import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

from enums.config_dimensions import MODEL_FAMILY_MAP, DATASET_MAP, ModelFamily, ModelSource

PRECISION_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def main():
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)

    model_family = ModelFamily(config["model_family"])
    model_source = ModelSource(config["model_source"])
    model_id = MODEL_FAMILY_MAP[model_source][model_family][config["model_params_M"]]
    torch_dtype = PRECISION_MAP[config["precision"]]

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if model_source == ModelSource.MODELSCOPE:
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch_dtype,
            device_map={"": local_rank}, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype,
            device_map={"": local_rank},
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = DATASET_MAP[model_source]
    dataset = load_dataset(ds["path"], ds["name"], split="train[:1000]")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config["sequence_length"],
            padding="max_length",
            return_tensors="pt",
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch")

    optim = "adamw_torch" if config["optimizer"] == "adamw" else "sgd"

    training_args = TrainingArguments(
        output_dir="/tmp/training_output",
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        max_steps=1000,
        logging_steps=10,
        save_strategy="no",
        fp16=config["precision"] == "fp16",
        bf16=config["precision"] == "bf16",
        optim=optim,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    ready_signal_path = config["ready_signal_path"]
    Path(ready_signal_path).touch()

    trainer.train()


if __name__ == "__main__":
    main()
