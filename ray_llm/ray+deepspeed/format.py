import torch
import torch.distributed as dist
import torch.nn.functional as F

# 对输入的序列列表进行填充操作，使得每个序列达到相同的长度
def zero_pad_sequences(sequences, side: str = "left", value=0):
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)


def exist_and_not_none(d, key):
    return key in d and not d[key] is None

"""
 process_multi_turn_dialogue引用openrlhf提供的函数，用于将多轮对话数据处理成特定格式的字符串输出，例如模拟人类和助手之间的对话。
参数:
conversations: 包含对话内容的列表。
input_template: 格式化字符串模板，用于格式化人类输入和助手输出。
content_key: 对话内容在每个对话条目中的键名。
role_key: 人类在每个对话条目中的键名。
"""
def process_multi_turn_dialogue(
    conversations, input_template="Human: {}\nAssistant: ", content_key="content", role_key="role"
):
    result = []
    for l in conversations:
        if "user" in l[role_key] or "human" in l[role_key]:
            result.append(input_template.format(l[content_key]))
        else:
            result.append(l[content_key] + "\n")
    return "".join(result)