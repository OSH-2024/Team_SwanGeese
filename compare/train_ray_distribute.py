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
    def __init__(self, dataset, num_splits, batch_size):
        self.tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = dataset.map(lambda x: process_func(x, self.tokenizer), remove_columns=dataset.column_names)
        self.num_splits = num_splits
        self.batch_size = batch_size
        self.current_index = [0] * num_splits
        self.split_size = len(self.dataset) // num_splits

    def get_batch_for_gpu(self, gpu_id):
        if self.current_index[gpu_id] >= self.split_size:
            return None
        start_idx = self.current_index[gpu_id]
        end_idx = min(start_idx + self.batch_size, self.split_size)
        self.current_index[gpu_id] = end_idx
        return self.dataset.select(range(gpu_id * self.split_size + start_idx, gpu_id * self.split_size + end_idx))

    def get_progress(self, gpu_id):
        return self.current_index[gpu_id], self.split_size

# 启动数据分发Actor
data_distributor = DataDistributor.remote(dataset, 2, batch_size=20)

@ray.remote(num_gpus=1)
def train_model(gpu_id, data_distributor):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device(f'cuda:0')
    
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

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
        def __init__(self, *args, data_distributor=None, gpu_id=None, tokenizer=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.data_distributor = data_distributor
            self.gpu_id = gpu_id
            self.start_time = None
            self.total_tokens = 0
            self.total_loss = 0.0
            self.num_steps = 0
            self.tokenizer = tokenizer
            self.train_losses = []
            self.log_info = []

        def train(self, *args, **kwargs):
            self.start_time = time.time()
            self.total_tokens = 0
            self.total_loss = 0.0
            self.num_steps = 0
            while True:
                batch = ray.get(self.data_distributor.get_batch_for_gpu.remote(self.gpu_id))
                if batch is None:
                    break
                self.train_dataset = batch
                super().train(*args, **kwargs)
                self.print_progress()
                logs = {
                    'train_runtime': time.time() - self.start_time,
                    'train_samples_per_second': self.total_tokens / (time.time() - self.start_time),
                    'train_steps_per_second': self.num_steps / (time.time() - self.start_time),
                    'train_loss': self.total_loss / self.num_steps if self.num_steps > 0 else 0.0,
                    'epoch': self.state.epoch
                }
                self.log_info.append(logs)
                print(f"Custom Log: {logs}")
            elapsed_time = time.time() - self.start_time
            tokens_per_second = self.total_tokens / elapsed_time
            avg_loss = self.total_loss / self.num_steps if self.num_steps > 0 else 0.0
            print(f"Training completed. Tokens per second: {tokens_per_second:.2f}, Average loss: {avg_loss:.4f}")
            return tokens_per_second, self.total_loss, self.num_steps, elapsed_time, self.train_losses, self.log_info

        def training_step(self, model, inputs):
            inputs = self._prepare_inputs(inputs)
            tokens_in_batch = inputs["input_ids"].ne(self.tokenizer.pad_token_id).sum().item()
            self.total_tokens += tokens_in_batch
            loss = super().training_step(model, inputs)
            self.total_loss += loss.item()
            self.train_losses.append(loss.item())
            self.num_steps += 1
            return loss

        def print_progress(self):
            trained, total = ray.get(self.data_distributor.get_progress.remote(self.gpu_id))
            progress_percentage = (trained / total) * 100
            print(f"GPU {self.gpu_id}: Trained {trained} / {total} batches ({progress_percentage:.2f}%)")

        def evaluate(self, eval_dataset=None):
            self.model.eval()
            total_eval_loss = 0
            total_steps = 0
            with torch.no_grad():
                for batch in eval_dataset:
                    inputs = self._prepare_inputs(batch)
                    loss = self.compute_loss(self.model, inputs)
                    total_eval_loss += loss.item()
                    total_steps += 1
            avg_eval_loss = total_eval_loss / total_steps if total_steps > 0 else 0.0
            return avg_eval_loss

    trainer = CustomTrainer(
        model=model,
        args=args,
        data_distributor=data_distributor,
        gpu_id=gpu_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        tokenizer=tokenizer
    )

    tokens_per_second, total_loss, num_steps, elapsed_time, train_losses, log_info = trainer.train()

    # 获取新的测试数据集
    test_dataset = ray.get(data_distributor.get_batch_for_gpu.remote(gpu_id))
    if test_dataset:
        avg_test_loss = trainer.evaluate(test_dataset)
    else:
        avg_test_loss = float('nan')

    peft_model_id = f"./llama3_lora_gpu_{gpu_id}"
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    return tokens_per_second, total_loss, num_steps, elapsed_time, train_losses, log_info, avg_test_loss

# 启动分布式训练任务并收集结果
start_time = time.time()
results = ray.get([
    train_model.remote(0, data_distributor),
    train_model.remote(1, data_distributor)
])
total_time = time.time() - start_time

# 分别输出每个GPU的数据吞吐率和损失值
all_train_losses = []
log_infos = []
for i, result in enumerate(results):
    print(f"GPU {i} tokens per second: {result[0]:.2f}")
    print(f"GPU {i} test loss: {result[6]:.4f}")
    all_train_losses.extend(result[4])
    log_infos.extend(result[5])

# 计算总的tokens/s和汇总train_loss
total_tokens_per_second = sum(result[0] for result in results)
total_train_loss = sum(result[1] for result in results)
total_train_steps = sum(result[2] for result in results)
total_elapsed_time = sum(result[3] for result in results)
average_test_loss = sum(result[6] for result in results if not pd.isna(result[6])) / len(results)

# 计算每次日志记录的平均训练损失
average_train_loss = sum(log['train_loss'] for log in log_infos) / len(log_infos)

print(f"Total tokens per second: {total_tokens_per_second:.2f}")
print(f"Average train loss: {average_train_loss:.4f}")
print(f"Total training time: {total_time:.2f} seconds")
