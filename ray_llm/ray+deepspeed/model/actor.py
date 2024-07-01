from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

from .utils import log_probs_from_logits

# actor类
class Actor(nn.Module):
    def __init__(
        self,
        pretrain_or_model,
        from_config=False,
        use_flash_attention_2=False,
    ) -> None:
        super().__init__()

        # 加载配置和模型
        if isinstance(pretrain_or_model, str):
            if from_config:
                config = AutoConfig.from_pretrained(
                    pretrain_or_model,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    use_flash_attention_2=use_flash_attention_2,
                )
                self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    pretrain_or_model,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    use_flash_attention_2=use_flash_attention_2,
                )
        else:
            self.model = pretrain_or_model

    # 生成action序列
    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, **kwargs
    ) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:

        # 设置生成参数
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", False),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_tokens": kwargs.get("min_new_tokens ", 1),
        }

        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # Call generate
        # [batch,input_len+action_len]
        sequences = self.model.generate(**generate_args)

        # Prepare mask tensor
        eos_token_id = generate_args["eos_token_id"]
        pad_token_id = generate_args["pad_token_id"]

        # 非结束 && 非填充
        # [batch,input_len+action_len]
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)
        # 按行
        for i in range(attention_mask.size(0)):
            # 反向按列
            for t in reversed(range(seq_length)):
                # 设置尾部标识
                if attention_mask[i][t] > 0.5:
                    attention_mask[i][min(t + 1, seq_length - 1)] = True
                    sequences[i][min(t + 1, seq_length - 1)] = eos_token_id
                    break

        input_len = input_ids.size(1)
        # [batch,action_len]
        action_seq = sequences[:, input_len:-1]
        # 非结束 && 非填充
        # [batch,action_len]
        action_mask = action_seq.ne(eos_token_id) & action_seq.ne(pad_token_id)
        return sequences, attention_mask, action_mask

    # 获取action概率
    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        """Returns action log probs"""
        # 取生成序列对应的概率值
        # [batch,input_len+action_len,vocab_dim]
        output = self.model(sequences, attention_mask=attention_mask)

        if return_output:
            return output
        else:
            # [batch,input_len+action_len,vocab_dim] --> gather [batch,input_len+action_len-1] --> [batch,input_len+action_len-1]
            log_probs = log_probs_from_logits(output["logits"][:, :-1, :], sequences[:, 1:])
            # 取action部分的概率
            # [batch,action_len]
            return log_probs[:, -num_actions:]

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
