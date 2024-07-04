from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import time
import torch
from peft import LoraConfig, TaskType, get_peft_model

# 将JSON文件转换为CSV文件
df = pd.read_json('./huanhuan.json')
ds = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def process_func(example):
    MAX_LENGTH = 384  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"user\n\n{example.get('instruction', '') + example.get('input', '')}assistant\n\n", add_special_tokens=False)
    response = tokenizer(f"{example.get('output', '')}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct', device_map="auto", torch_dtype=torch.bfloat16)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alpha，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./output/llama3",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)

# 自定义Trainer以测量tokens/s
class CustomTrainer(Trainer):
    def __init__(self, *args, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None
        self.total_tokens = 0
        self.tokenizer = tokenizer  # 保存tokenizer为类属性

    def train(self, *args, **kwargs):
        # 训练开始时记录时间
        self.start_time = time.time()
        self.total_tokens = 0
        super().train(*args, **kwargs)
        # 训练结束时计算每秒处理的token数
        elapsed_time = time.time() - self.start_time
        tokens_per_second = self.total_tokens / elapsed_time
        print(f"Training completed. Tokens per second: {tokens_per_second:.2f}")

    def training_step(self, model, inputs):
        # 计算每步处理的token数
        inputs = self._prepare_inputs(inputs)
        tokens_in_batch = inputs["input_ids"].ne(self.tokenizer.pad_token_id).sum().item()
        self.total_tokens += tokens_in_batch
        return super().training_step(model, inputs)

trainer = CustomTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    tokenizer=tokenizer  # 将tokenizer传递给CustomTrainer
)

trainer.train()

peft_model_id = "./llama3_lora"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
