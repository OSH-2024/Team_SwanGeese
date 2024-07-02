import math
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers.trainer import get_scheduler
from trainer import PPOTrainer
from utils import DeepspeedStrategy, blending_datasets, get_tokenizer
from .launcher import BasePPORole


class CriticPPOTrainer(PPOTrainer):
    """
    CriticPPOTrainer继承自PPOTrainer，专门用于训练Critic模型。
    """

    def ppo_train(self):
        """
        执行PPO训练过程，包括初始化数据加载器、迭代训练和状态记录。
        """
        # 初始化数据加载器
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience)

                # 对于分布式数据并行 (DP)
                status = self.strategy.all_reduce(status)

                status_list.append(status)
                pbar.set_postfix(status)

        # 计算状态的平均值
        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience) -> Dict[str, float]:
        """
        执行训练步骤，返回训练状态。
        """
        return self.training_step_critic(experience)


class CriticModelActor(BasePPORole):
    """
    CriticModelActor类用于初始化和训练Critic模型。
    """

    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain, max_steps):
        """
        从预训练模型初始化Critic模型，包括设置分布式策略、优化器和调度器。
        """
        args = strategy.args

        # 设置分布式策略
        self._setup_distributed(strategy)
        critic = get_llm_for_sequence_regression(
            pretrain,
            "critic",
            normalize_reward=strategy.args.normalize_reward,
            use_flash_attention_2=strategy.args.flash_attn,
            bf16=strategy.args.bf16,
            load_in_4bit=strategy.args.load_in_4bit,
            lora_rank=strategy.args.lora_rank,
            lora_alpha=strategy.args.lora_alpha,
            target_modules=strategy.args.target_modules,
            lora_dropout=strategy.args.lora_dropout,
            ds_config=strategy.get_ds_train_config(is_actor=False),
            head_prefix=strategy.args.head_prefix,
        )
        strategy.print(critic)
        strategy.print("reward normalization status: {}".format(strategy.args.normalize_reward))
        strategy.print("mean: {}, std {}".format(critic.mean, critic.std))

        # 配置tokenizer
        if strategy.args.save_value_network:
            self.tokenizer = get_tokenizer(
                pretrain, critic, "left", strategy, use_fast=not strategy.args.disable_fast_tokenizer
            )

        # 配置优化器
        critic_optim = strategy.create_optimizer(
            critic, lr=args.critic_learning_rate, betas=(0.9, 0.95), weight_decay=args.l2
        )

        # 配置调度器
        critic_scheduler = get_scheduler(
            "cosine",
            critic_optim,
            num_warmup_steps=math.ceil(max_steps * 0.03),
            num_training_steps=max_steps,
        )

        # 启用梯度检查点
        if args.gradient_checkpointing:
            critic.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
            )

        # 准备模型和优化器
        self.critic, self.critic_optim, self.critic_scheduler = strategy.prepare(
            (critic, critic_optim, critic_scheduler),
            is_rlhf=True,
        )

        # 配置训练器，不使用wandb
        strategy.args.use_wandb = False
        self.trainer = CriticPPOTrainer(
            strategy,
            actor=None,
            critic=self.critic,
            reward_model=None,
            initial_model=None,
            ema_model=None,
            actor_optim=None,
            critic_optim=self.critic_optim,
            actor_scheduler=None,
            critic_scheduler=self.critic_scheduler,
            max_epochs=args.max_epochs,
            micro_train_batch_size=args.micro_train_batch_size,
            micro_rollout_batch_size=args.micro_rollout_batch_size,
            gradient_checkpointing=args.gradient_checkpointing,
            prompt_max_len=args.prompt_max_len,
            value_clip=args.value_clip,
            eps_clip=args.eps_clip,
        )

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        生成Critic模型的值。
        """
        device = torch.cuda.current_device()
        self.critic.eval()
        with torch.no_grad():
            value = self.critic(sequences.to(device), action_mask.to(device), attention_mask.to(device))
        self.critic.train()  # 重置模型状态
        return value.to("cpu")

    def append(self, experience):
        """
        将经验添加到重放缓冲区。
        """
        self.trainer.replay_buffer.append(experience)

    def fit(self):
        """
        使用重放缓冲区训练Critic模型。
        """
        torch.cuda.empty_cache()
        self.critic.train()
        status = self.trainer.ppo_train()
        self.trainer.replay_buffer.clear()
        torch.cuda.empty_cache()
        return status

    def empty_cache(self) -> None:
        """
        清空缓存。
        """
        torch.cuda.empty_cache()

    def save_model(self):
        """
        在训练后保存模型检查点（仅在rank 0上）。
        """
        args = self.strategy.args

        # 保存模型检查点
        self.strategy.save_model(
            self.critic,
            self.tokenizer,
            args.save_path + "_critic",
        )
