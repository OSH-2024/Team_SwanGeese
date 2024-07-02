作者：OpenDeployment
链接：https://zhuanlan.zhihu.com/p/662968485
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

from typing import Optional, Tuple, Union

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F

# 计算近似KL散度
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base
    return log_ratio * action_mask

# 计算收益
def compute_reward(
    r: Union[torch.Tensor, float],
    kl_coef: float,
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    action_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kl_coef <= 0.0:
        kl_coef = 0.0

    # [batch,action_len]
    kl = compute_approx_kl(log_probs, log_probs_base, action_mask=action_mask)
    kl_reward = -kl_coef * kl

    # 裁剪
    r = r.clamp(min=-10, max=10)
    # [batch,action_len]
    last_reward = torch.zeros_like(kl)
    # 尾部赋值
    for i in range(last_reward.size(0)):
        for t in reversed(range(last_reward.size(1))):
            if action_mask[i][t] > 0.5:
                last_reward[i][t] = r[i]
                break
    # [batch,action_len]
    reward = last_reward + kl_reward
    return reward, kl

# 获取概率
def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)

# 基于mask对dim取均值
def masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    tensor = tensor * mask
    tensor = tensor.sum(dim=dim)
    mask_sum = mask.sum(dim=dim)
    mean = tensor / (mask_sum + 1e-8)
    return mean

# 基于mask对dim归一化
def masked_normalize(tensor: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    tensor = tensor * mask
    mean = masked_mean(tensor, mask, dim=dim)
    mean_centered = tensor - mean
    var = masked_mean(mean_centered**2, mask, dim=dim)
    return mean_centered * var.clamp(min=eps).rsqrt()

# 转换成lora模型
def convert_to_lora(
    model: nn.Module,
    input_size: int,
    output_size: int,
    lora_rank: int = 16,
    lora_alpha: int = 1,
    lora_dropout: float = 0.0,
    fan_in_fan_out: bool = False,
    merge_weights: bool = True,
):
    if lora_rank > min(input_size, output_size):
        raise ValueError(f"LoRA rank {lora_rank} must be less or equal than {min(input_size, output_size)}")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._modules[name] = lora.Linear(
                input_size,
                output_size,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                fan_in_fan_out=fan_in_fan_out,
                merge_weights=merge_weights,
            )