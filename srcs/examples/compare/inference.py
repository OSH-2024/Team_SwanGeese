from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
import time

mode_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3-8B-Instruct'
lora_path = './llama3_lora'

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(mode_path)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto", torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, lora_path)

prompt = "问世间情为何物？"
messages = [
    # {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
    {"role": "user", "content": prompt}
]

# 假设 apply_chat_template 是一个有效的方法，如果没有请根据实际需求调整
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

# 记录开始时间
start_time = time.time()

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    do_sample=True,
    top_p=0.9, 
    temperature=0.5, 
    repetition_penalty=1.1,
    eos_token_id=tokenizer.eos_token_id,
)

# 记录结束时间
end_time = time.time()

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 计算生成的 tokens 数量
num_generated_tokens = sum(len(ids) for ids in generated_ids)
# 计算推理时间
inference_time = end_time - start_time
# 计算数据吞吐率 tokens/s
tokens_per_second = num_generated_tokens / inference_time

print(prompt)
print(response)
print(f"Tokens/s: {tokens_per_second:.2f}")