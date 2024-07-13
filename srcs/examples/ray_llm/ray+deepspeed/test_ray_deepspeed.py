import time
import ray
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import deepspeed
from deepspeed.pipe import PipelineModule

@ray.remote(num_gpus=1)
class LlamaModel:
    def __init__(self, model_name: str, pipeline_parallel_size: int):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name)
        self.model = PipelineModule(
            layers=self.model,
            loss_fn=torch.nn.CrossEntropyLoss(),
            num_stages=pipeline_parallel_size,
            partition_method='uniform',
            activation_checkpoint_interval=0
        )
        self.engine, _, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=[p for p in self.model.parameters() if p.requires_grad],
            training_data=None
        )

    def generate_text(self, prompts: list, max_length: int = 50):
        batch_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        start_time = time.time()
        outputs = self.model.generate(**batch_inputs, max_length=max_length)
        end_time = time.time()
        output_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        num_tokens = sum(output.shape[0] for output in outputs)
        elapsed_time = end_time - start_time
        return output_texts, num_tokens, elapsed_time

# 初始化Ray
ray.init()

# 启动多个Actors
num_actors = 4
pipeline_parallel_size = 2  # 根据需要调整并行度
llama_actors = [LlamaModel.remote("llama-7b", pipeline_parallel_size) for _ in range(num_actors)]

# 性能测试函数
def performance_test(prompts: list, num_requests: int = 10, max_length: int = 50):
    total_tokens = 0
    total_time = 0

    # 分批处理请求
    batch_size = len(prompts) // num_actors
    for _ in range(num_requests):
        futures = [llama_actors[i % num_actors].generate_text.remote(prompts[i * batch_size:(i + 1) * batch_size], max_length) for i in range(num_actors)]
        results = ray.get(futures)
        for output_texts, num_tokens, elapsed_time in results:
            total_tokens += num_tokens
            total_time += elapsed_time
            for output_text in output_texts:
                print(f"Generated text: {output_text}")
                print(f"Tokens: {num_tokens}, Time: {elapsed_time:.4f} seconds")
    
    tokens_per_second = total_tokens / total_time
    print(f"Total tokens: {total_tokens}, Total time: {total_time:.4f} seconds")
    print(f"Throughput: {tokens_per_second:.2f} tokens/second")

# 示例调用
prompts = ["Once upon a time" for _ in range(40)]  # 示例批处理提示词
performance_test(prompts, num_requests=10, max_length=50)
