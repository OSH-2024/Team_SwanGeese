from typing import Callable

from torch.utils.data import Dataset
from tqdm import tqdm

from .format import exist_and_not_none, zero_pad_sequences
# 对PPO算法还不算熟，在openrlhf代码的基础上改写了一下
# 不同数据集对应不一样的拼接方式
def preprocess_data(data):
    # Open-Orca/OpenOrca
    # 带有任务描述 和 问题
    if exist_and_not_none(data, "system_prompt") and exist_and_not_none(data, "response"):
        prompt = "Human: " + data["system_prompt"] + "\n" + data["question"] + "\nAssistant: "
        target = data["response"]
    # BelleGroup/train_0.5M_CN
    # LLMs/Alpaca-ShareGPT
    # yahma/alpaca-cleaned
    # QingyiSi/Alpaca-CoT
    # 仅有指令
    elif exist_and_not_none(data, "instruction") and exist_and_not_none(data, "output"):
        input = " " + data["input"] if exist_and_not_none(data, "input") else ""
        prompt = "Human: " + data["instruction"] + input + "\nAssistant: "
    # stanfordnlp/SHP
    # 基于历史
    elif exist_and_not_none(data, "history"):
        prompt = "Human: " + data["history"] + "\nAssistant: "
    # lvwerra/stack-exchange-paired
    # 仅有问题
    elif exist_and_not_none(data, "question") and exist_and_not_none(data, "response_j"):
        prompt = "Human: " + data["question"] + "\nAssistant: "
    # lmsys/chatbot_arena_conversations
    # 多伦对话
    elif exist_and_not_none(data, "winner") and exist_and_not_none(data, "conversation_a"):

        def process_chatbot_arena_conversations(lll):
            result = []
            for l in lll:
                result.append(l["role"].replace("user", "Human: ").replace("assistant", "Assistant: "))
                result.append(l["content"])
            return "\n".join(result)

        prompt = data["conversation_a"][:-1]
        prompt = process_chatbot_arena_conversations(prompt) + "\nAssistant: "
    # openai/webgpt_comparisons
    # 仅有问题
    elif exist_and_not_none(data, "question") and exist_and_not_none(data, "answer_1"):
        prompt = "Human: " + data["question"]["full_text"] + "\nAssistant: "
    # Dahoas/full-hh-rlhf
    # 仅有指令
    elif exist_and_not_none(data, "prompt"):
        prompt = data["prompt"]
        # tasksource/oasst1_pairwise_rlhf_reward
        if prompt.startswith("prompter:"):
            prompt = prompt.replace("prompter:", "\nHuman: ").replace("assistant:", "\nAssistant: ") + "\nAssistant: "
    # JSON files for batch inference
    # 仅有输入
    elif exist_and_not_none(data, "input"):
        prompt = "Human: " + data["input"] + "\nAssistant: "
    else:
        raise ValueError("prompt dataset key error")
    return prompt

# 管理prompt数据类
class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(self, dataset, strategy) -> None:
        super().__init__()
        self.strategy = strategy

        self.prompts = []
        # 数据集并进行处理
        for data in tqdm(dataset, disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data)
            self.prompts.append(prompt)

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.prompts[idx]