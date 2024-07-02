import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from modelscope import snapshot_download
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import ray
from ray.util.multiprocessing import Pool

# 初始化Ray
ray.init()

# 需要微调的基座模型
model_id = 'LLM-Research/Meta-Llama-3-8B-Instruct'

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

models_dir = './models'
dataset_file = './dataset/huanhuan.json'
model_path = f"{models_dir}/model/{model_id.replace('.', '___')}"
checkpoint_dir = f"./models/checkpoint/{model_id}"
lora_dir = f"./models/lora/{model_id}"

torch_dtype = torch.half

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

@ray.remote
def process_data(item, tokenizer):
    MAX_LENGTH = 384
    instruction = tokenizer(
        f"user\n\n{item['instruction'] + item['input']}assistant\n\n",
        add_special_tokens=False)
    response = tokenizer(f"{item['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def train():
    model_dir = snapshot_download(model_id=model_id, cache_dir=f"{models_dir}/model", revision='master')
    if model_path != model_dir:
        raise Exception(f"model_path:{model_path} != model_dir:{model_dir}")

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch_dtype)
    model.enable_input_require_grads()

    df = pd.read_json(dataset_file)
    ds = Dataset.from_pandas(df)
    print(ds[:3])

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    with Pool() as pool:
        tokenized_data = list(pool.map(process_data.remote, [item for item in ds], [tokenizer] * len(ds)))

    tokenized_id = Dataset.from_dict(tokenized_data)

    tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[1]["labels"])))

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train()

    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

def infer(prompt="你是谁？"):
    print(f"prompt: {prompt}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch_dtype)
    model = PeftModel.from_pretrained(model, model_id=lora_dir, config=lora_config)

    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        eos_token_id=tokenizer.encode('')[0],
        pad_token_id=tokenizer.pad_token_id,
        attention_mask=model_inputs.attention_mask,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

if __name__ == '__main__':
    train()
    res = infer(prompt="你是谁？")
    print(res)
