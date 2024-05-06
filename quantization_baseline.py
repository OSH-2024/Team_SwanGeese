from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
import json
from typing import List, Optional, Tuple
import time
from predict import predict
from tqdm import tqdm


def run_llm(
    requests: List[Tuple[str, int, int]],
) -> float:
    model_info = "/staff/liqi/Aquila"
    total_requests = len(requests)
    tokenizer = AutoTokenizer.from_pretrained(model_info, trust_remote_code=True)
    quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True
                    )
    model = AutoModelForCausalLM.from_pretrained(model_info, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                quantization_config=quantization_config,
                                                )
    model.eval()
    input_num_tokens = []
    output_num_tokens = []
    start = time.perf_counter()
    for prompt, prompt_len, output_len in tqdm(requests, total=total_requests, desc="Processing requests"):
        # Generate the sequences
        out = predict(model, prompt, tokenizer=tokenizer, max_gen_len=200, top_p=0.9,
                    seed=123, topk=15, temperature=1.0, sft=False,
                    model_name="AquilaChat2-34B")
        input_num_tokens.append(len(tokenizer.encode(prompt, add_special_tokens=False)))
        output_num_tokens.append(len(tokenizer.encode(out, add_special_tokens=False)))
    end = time.perf_counter()
    return end - start, input_num_tokens, output_num_tokens

if __name__ == "__main__":
    dataset_path = "/staff/liqi/AquilaChat2-34B/dataset/scrambled_sampled_dataset.json"
    num_samples = 10
    with open(dataset_path) as f:
        requests = json.load(f)
    if num_samples is not None:
        requests = requests[:num_samples]
    elapsed_time, input_num_tokens, output_num_tokens = run_llm(requests)
    total_input_num_tokens = sum(input_num_tokens)
    # 输出里不含有输入的token，但是给出的baseline里输出时加上了输入的token
    total_output_num_tokens = sum(output_num_tokens) + total_input_num_tokens
    tokens_per_second = total_output_num_tokens / elapsed_time
    throughput = len(requests) / elapsed_time
    print(f"Throughput: {throughput:.2f} requests/s \n"
            f"Tokens/s: {tokens_per_second:.2f} tokens/s \n"
            f"Prompt_num_tokens: {total_input_num_tokens:.2f} tokens \n"
            f"Total_num_tokens: {total_output_num_tokens:.2f} tokens \n")