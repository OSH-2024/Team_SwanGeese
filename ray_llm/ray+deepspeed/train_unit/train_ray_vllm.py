import argparse
from datetime import datetime
from typing import List

import ray
import torch
from ray.util.placement_group import placement_group

from openrlhf.trainer.ray import (
    ActorModelRayActor,
    CriticModelRayActor,
    PPORayActorGroup,
    ReferenceModelRayActor,
    RewardModelRayActor,
    create_vllm_engines,
)
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer

# NOTE: reward function for multiple reward models, replace this with your own function!
def reward_fn(rewards: List[torch.Tensor]):
    return torch.stack(rewards).sum(dim=0)  # 将多个奖励张量堆叠并求和


def train(args):
    _validate_args(args)  # 验证参数

    # 配置策略
    strategy = get_strategy(args)

    # 如果需要，将 actor 和 reference 模型放在同一组中
    pg = None
    if args.colocate_actor_ref:
        assert args.actor_num_nodes == args.ref_num_nodes and args.actor_num_gpus_per_node == args.ref_num_gpus_per_node, \
            f"num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

        bundles = [{"GPU": args.actor_num_gpus_per_node, "CPU": args.actor_num_gpus_per_node} for _ in range(args.actor_num_nodes)]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())  # 等待分配组就绪

    # 初始化 actor 和 reference 模型
    actor_model = PPORayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        ActorModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.75 if pg else 1,
    )

    ref_model = PPORayActorGroup(
        args.ref_num_nodes,
        args.ref_num_gpus_per_node,
        ReferenceModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.25 if pg else 1,
    )

    # 如果需要，将 critic 和 reward 模型放在同一组中
    pg = None
    if args.colocate_critic_reward:

        bundles = [{"GPU": args.critic_num_gpus_per_node, "CPU": args.critic_num_gpus_per_node} for _ in range(args.critic_num_nodes)]
        pg = placement_group(bundles, strategy="STRICT_SPREAD")
        ray.get(pg.ready())  # 等待分配组就绪

    # 初始化 critic 模型
    critic_model = PPORayActorGroup(
        args.critic_num_nodes,
        args.critic_num_gpus_per_node,
        CriticModelRayActor,
        pg=pg,
        num_gpus_per_actor=0.75 if pg else 1,
    )

    # 初始化多个 reward 模型
    reward_pretrains = args.reward_pretrain.split(",")
    reward_models = []
    for _ in reward_pretrains:
        reward_models.append(
            PPORayActorGroup(
                args.reward_num_nodes,
                args.reward_num_gpus_per_node,
                RewardModelRayActor,
                pg=pg,
                num_gpus_per_actor=0.25 if pg else 1,
            )
        )

    # 异步初始化所有模型
    refs = []
    refs.extend(ref_model.async_init_model_from_pretrained(strategy, args.pretrain))
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain))
    for reward_model, reward_pretrain in zip(reward_models, reward_pretrains):
        refs.extend(reward_model.async_init_model_from_pretrained(strategy, reward_pretrain))

    # 如果启用 vLLM，引擎用于文本生成
    vllm_engines = None
    if args.vllm_num_engines is not None:
        max_len = args.max_len if args.max_len else args.prompt_max_len + args.generate_max_len
        vllm_engines = create_vllm_engines(
            args.vllm_num_engines,
            args.vllm_tensor_parallel_size,
            args.pretrain,
            args.seed,
            args.enable_prefix_caching,
            max_len,
        )

    # critic 调度器初始化依赖于 max_step，所以必须在 actor 之后初始化 critic
    max_steps = ray.get(actor_model._actor_handlers[0].max_steps.remote())
    refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.critic_pretrain, max_steps))
    ray.get(refs)  # 等待所有模型初始化完成

    # 训练 actor 和 critic 模型
    refs = actor_model.async_fit_actor_model(
        critic_model, ref_model, reward_models, reward_fn=reward_fn, vllm_engines=vllm_engines
    )
    ray.get(refs)  # 等待训练完成

    # 保存模型
    ray.get(actor_model.async_save_model())
    if args.save_value_network:
        ray.get(critic_model.async_save_model())

    args = parser.parse_args()

    # 开始训练
    train(args)