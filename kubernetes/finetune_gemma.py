from optimum.tpu import fsdp_v2
from datasets import load_dataset
from transformers import AutoTokenizer
from optimum.tpu import AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer
from transformers import TrainingArguments


def preprocess_function(sample):
    instruction = f"### Instruction\n{sample['instruction']}"
    context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"### Answer\n{sample['response']}"
    # join all the parts together
    prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    prompt += tokenizer.eos_token
    sample["prompt"] = prompt
    return sample


model_id = "google/gemma-2b"

fsdp_v2.use_fsdp_v2()

print("----0----")

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

tokenizer = AutoTokenizer.from_pretrained(model_id)

data = dataset.map(preprocess_function, remove_columns=list(dataset.features))

print("----1----")

model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False)

print("----2----")

# Set up PEFT LoRA for fine-tuning.
lora_config = LoraConfig(
    r=8,
    target_modules=["k_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

print("----3----")

# Set up the FSDP arguments
fsdp_training_args = fsdp_v2.get_fsdp_training_args(model)

# Set up the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=data,
    args=TrainingArguments(
        per_device_train_batch_size=64,
        num_train_epochs=32,
        max_steps=-1,
        output_dir="./output",
        optim="adafactor",
        logging_steps=1,
        dataloader_drop_last = True,  # Required for FSDPv2.
        **fsdp_training_args,
    ),
    peft_config=lora_config,
    dataset_text_field="prompt",
    max_seq_length=1024,
    packing=True,
)

print("----4----")

trainer.train()
