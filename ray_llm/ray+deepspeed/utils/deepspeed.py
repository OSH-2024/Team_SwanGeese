import os
import random
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel
from torch import distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from openrlhf.models import Actor

from .deepspeed_utils import (
    _z3_params_to_fetch,
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
)

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class DeepspeedStrategy(ABC):
    """
    The strategy for training with Accelerator.
    """

    def __init__(
        self,
        seed: int = 42,
        max_norm: float = 0.0,
        micro_train_batch_size=1,
        train_batch_size=1,
        zero_stage=2,
        max_out_tokens=512,
        inference_tp_size=1,
        bf16=True,
        args=None,
    ) -> None:
        super().__init__()

        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.max_out_tokens = max_out_tokens
        self.micro_train_batch_size = micro_train_batch_size
        self.inference_tp_size = inference_tp_size
        self.bf16 = bf16
        self.adam_offload = args.adam_offload
        self.is_rlhf = False
        self.zpg = args.zpg
        self.seed = seed
        self.max_norm = max_norm
        self.time_steps = defaultdict(int)

    # 设置随机数
    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 设置分布式配置
    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        self.set_seed(self.seed)

        if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:  
            self.args.local_rank = int(os.environ["LOCAL_RANK"])

        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)
        deepspeed.init_distributed(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size // self.world_size

    # 创建优化器
    def create_optimizer(self, model, **kwargs) -> Optimizer:
        if isinstance(model, Actor):
            model = model.model
        AdamOptimizer = DeepSpeedCPUAdam if self.adam_offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    # 反向传播
    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        if isinstance(model, Actor):
            model = model.model
        model.backward(loss)

    # 每一步更新
    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        if isinstance(model, Actor):
            model = model.model
        model.step()

    # 设置数据集
    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
    ):
        sampler = DistributedSampler(
            replay_buffer,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            seed=self.seed,
            drop_last=drop_last,
        )
        return DataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    # 接wrap模型
    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, Actor):
            return self._unwrap_model(model.model)
        elif hasattr(model, "module"):
            return model.module
        else:
            return model

    # 整体训练准备
    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        self.is_rlhf = is_rlhf
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                ret.append(self.ds_init_train_model(*arg))
            else:
                ret.append(self.ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    # 初始化模型训练
    def ds_init_train_model(self, model, optim, scheduler):
        is_actor = isinstance(model, Actor)
        stage = self.stage

        ds_config = get_train_ds_config(
            offload=False,
            adam_offload=self.adam_offload,
            stage=stage,
            bf16=self.bf16,
            max_norm=self.max_norm,

            enable_hybrid_engine=is_actor and self.inference_tp_size > 1 and stage == 3,
            pin_parameters=True,
            inference_tp_size=self.inference_tp_size,
            tp_gather_partition_size=4,
            max_out_tokens=self.max_out_tokens,
            zpg=self.zpg,
        )

        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        train_batch_size = self.train_batch_size
        # corner case for ptx loss (backward twice)
        if self.is_rlhf and is_actor and self.args.pretrain_data is not None:
            train_batch_size *= 2
        ds_config["train_batch_size"] = train_batch_size

        # 初始化deepspeed
        engine, optim, _, scheduler = deepspeed.initialize(
            model=model.model if is_actor else model,
            optimizer=optim,
            lr_scheduler=scheduler,
            config=ds_config,
            args={"local_rank": self.args.local_rank},
            dist_init_required=True,
        )
        if is_actor:
            model.model = engine
        else:
            model = engine
        return model, optim, scheduler

    def ds_init_eval_model(self, model):
        is_actor = isinstance(model, Actor)
        stage = self.stage
        offload = False
        # No gradients
        if stage != 3:
            stage = 0
        # Offload ema model
        if getattr(model, "is_ema", None):
            offload = True
            stage = 0

        # DS Config
        ds_config = get_eval_ds_config(
            offload=offload,
            stage=stage,
            bf16=self.bf16,
            enable_hybrid_engine=is_actor and self.inference_tp_size > 1 and stage == 3,
            inference_tp_size=self.inference_tp_size,
            tp_gather_partition_size=self.inference_tp_size,
            max_out_tokens=self.max_out_tokens,
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        ds_config["train_batch_size"] = self.train_batch_size

        engine, *_ = deepspeed.initialize(
            model=model.model if is_actor else model,
            args={"local_rank": self.args.local_rank},
            config=ds_config,
            dist_init_required=True,
        )
        if is_actor:
            model.model = engine
        else:
            model = engine
        return model


    # 加载模型
    def load_model(
        self,
        model: nn.Module,
        path: str,
        map_location="cpu",
        strict: bool = False,
        key_replace_fn=None,
    ) -> None:
        # 解wrap模型
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    # 保存模型
    def save_model(self, model: nn.Module, path: str, only_rank0: bool = True) -> None:
        # 解wrap模型
        model_to_save = self._unwrap_model(model)
        # 如果是peft模型，进行合并
        if isinstance(model_to_save, PeftModel):
            model_to_save = model_to_save.merge_and_unload()

        if self.stage != 3:
            if self.is_rank_0():
                save_dict = model_to_save.state_dict()
                torch.save(save_dict, path)
        else:
            # stage ==3 的情况
            output_state_dict = {}
            # gather parameters
            for k, v in model_to_save.named_parameters():
                params_to_fetch = _z3_params_to_fetch([v])
                with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                    vv = v.data.cpu()
                    if self.is_rank_0():
                        output_state_dict[k] = vv
            if self.is_rank_0():
                for k, v in model_to_save.named_buffers():
                    vv = v.data.cpu()
                    output_state_dict[k] = vv
                torch.save(output_state_dict, path)

    # 保存成hf格式，包含保存权重、配置和分词器
    def save_hf_format(self, model, tokenizer, output_dir):
        # used to save huggingface format, so we can use it for hf.from_pretrained
        CONFIG_NAME = "config.json"
        WEIGHTS_NAME = "pytorch_model.bin"
        # save model weights for ZeRO2/3
        self.save_model(model, os.path.join(output_dir, WEIGHTS_NAME))
        if self.is_rank_0():
            # save config
            model_to_save = self._unwrap_model(model)
            output_config_file = os.path.join(output_dir, CONFIG_NAME)
            model_to_save.config.to_json_file(output_config_file)
            # save tokenizer
            tokenizer.save_vocabulary(output_dir)

    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            is_tensor = True
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            is_cpu_tensor = data.device.type == "cpu"

            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())
            if op == "mean":
                data /= self.world_size
            dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
            if is_cpu_tensor:
                data = data.cpu()
            return data.item() if not is_tensor else data

    def all_gather(self, data):
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            is_cpu_tensor = data.device.type == "cpu"

            ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
            dist.all_gather(ret, data.to(torch.cuda.current_device()))
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        return dist.get_rank()