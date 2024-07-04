import ray
import torch
import time
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model

# 初始化Ray
ray.init(address="auto")

# 将JSON文件转换为Dataset
df = pd.read_json('./huanhuan.json')
dataset = Dataset.from_pandas(df)

# 预处理函数
def process_func(example, tokenizer):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"user\n\n{example.get('instruction', '') + example.get('input', '')}assistant\n\n", add_special_tokens=False)
    response = tokenizer(f"{example.get('output', '')}", add_special_tokens=False)
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

# 定义数据分发Actor
@ray.remote
class DataDistributor:
    def __init__(self, dataset, num_splits):
        self.tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = dataset.map(lambda x: process_func(x, self.tokenizer), remove_columns=dataset.column_names)
        self.num_splits = num_splits
        self.split_size = len(self.dataset) // num_splits
        self.current_split = 0

    def get_data_for_gpu(self):
        if self.current_split >= self.num_splits:
            return None
        start_idx = self.current_split * self.split_size
        end_idx = (self.current_split + 1) * self.split_size
        self.current_split += 1
        return self.dataset.select(range(start_idx, end_idx))

# 启动数据分发Actor
data_distributor = DataDistributor.remote(dataset, 2)

@ray.remote(num_gpus=1)
def train_model(gpu_id, data_distributor):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f'cuda:0')
    
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    tokenized_data = ray.get(data_distributor.get_data_for_gpu.remote())
    if tokenized_data is None:
        print(f"No data available for GPU {gpu_id}")
        return 0

    model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=f"./output/llama3_gpu_{gpu_id}",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )

    class CustomTrainer(Trainer):
        def __init__(self, *args, tokenizer=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.start_time = None
            self.total_tokens = 0
            self.tokenizer = tokenizer

        def train(self, *args, **kwargs):
            self.start_time = time.time()
            self.total_tokens = 0
            super().train(*args, **kwargs)
            elapsed_time = time.time() - self.start_time
            tokens_per_second = self.total_tokens / elapsed_time
            print(f"Training completed. Tokens per second: {tokens_per_second:.2f}")
            return tokens_per_second

        def training_step(self, model, inputs):
            inputs = self._prepare_inputs(inputs)
            tokens_in_batch = inputs["input_ids"].ne(self.tokenizer.pad_token_id).sum().item()
            self.total_tokens += tokens_in_batch
            return super().training_step(model, inputs)

    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_data,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        tokenizer=tokenizer
    )

    tokens_per_second = trainer.train()

    peft_model_id = f"./llama3_lora_gpu_{gpu_id}"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    return tokens_per_second

# 启动分布式训练任务并收集tokens/s
results = ray.get([
    train_model.remote(0, data_distributor),
    train_model.remote(1, data_distributor)
])

# 计算总的tokens/s
total_tokens_per_second = sum(results)
print(f"Total tokens per second: {total_tokens_per_second:.2f}")
