"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Optional, Tuple

import concurrent.futures

from vllm import LLM, SamplingParams
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)
from tqdm import tqdm
import os

def run_vllm_on_specific_gpus(requests, model_path, tokenizer, tensor_parallel_size, gpu_ids):
    # 设置CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    
    return run_vllm(requests, model_path, tokenizer, tensor_parallel_size)

def run_vllm(
    requests: List[Tuple[str, int, int]],
    model_path: str,
    tokenizer: AutoTokenizer,
    tensor_parallel_size: int,
) -> float:

    # Initialize the model
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, trust_remote_code=True, quantization="awq")

    input_text = []
    output_text = []
    input_num_tokens = []
    output_num_tokens = []
    
    for prompt, prompt_len, output_len in requests:
        # Generate the sequences
        sampling_params = SamplingParams(max_tokens=output_len)
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.perf_counter()
    r = llm._run_engine(use_tqdm=True)
    end = time.perf_counter()
    # Assuming request_output is a list of RequestOutput objects
    for request_output_item in r:
        completion_output = request_output_item.outputs[0]
        generated_text = completion_output.text
        prompt_here = request_output_item.prompt
        # Tokenize the output for token count calculation
        input_text.append(prompt_here)
        output_text.append(generated_text)

    output_num_tokens = [len(tokenizer.encode(generated_text, add_special_tokens=False)) for generated_text in output_text] 
    input_num_tokens = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in input_text]
    return end - start, input_num_tokens, output_num_tokens




# Main function
def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Preparing requests
    if args.dataset is None:
        prompt = "hi" * (args.input_len // len("hi"))
        requests = [(prompt, args.input_len, args.output_len) for _ in range(args.num_samples)]
    else:
        with open(args.dataset) as f:
            requests = json.load(f)

    if args.num_samples is not None:
        requests = requests[:args.num_samples]


    # 在main函数中
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 分别为两个进程设置不同的GPU
        future1 = executor.submit(run_vllm_on_specific_gpus, requests, args.model, tokenizer, 4, [3, 5, 6, 7])
        
        # 等待任务完成并获取结果
        elapsed_time1, input_num_tokens1, output_num_tokens1 = future1.result()

    total_elapsed_time = elapsed_time1
    total_input_num_tokens = input_num_tokens1
    total_output_num_tokens = output_num_tokens1
    total_input_num_tokens = sum(total_input_num_tokens)

    # vllm的输出里不含有输入的token，所以需要加上
    total_output_num_tokens = sum(total_output_num_tokens) + total_input_num_tokens

    tokens_per_second = total_output_num_tokens / total_elapsed_time

    throughput = len(requests) / total_elapsed_time

    print(f"Throughput: {throughput:.2f} requests/s \n"
          f"Tokens/s: {tokens_per_second:.2f} tokens/s \n"
          f"Prompt_num_tokens: {total_input_num_tokens:.2f} tokens \n"
          f"Total_num_tokens: {total_output_num_tokens:.2f} tokens \n")

# Command line arguments parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput using vllm.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model.")
    parser.add_argument("--dataset", type=str, default=None, help="Path to the dataset.")
    parser.add_argument("--input-len", type=int, default=None, help="Input prompt length for each request")
    parser.add_argument("--output-len", type=int, default=None, help="Output length for each request")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples for inference test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4, help="Tensor parallel size for vllm")

    args = parser.parse_args()

    main(args)