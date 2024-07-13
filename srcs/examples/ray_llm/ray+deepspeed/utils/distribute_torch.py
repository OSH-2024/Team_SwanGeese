from datetime import timedelta  
from typing import Any, Optional, Union  
  
import torch  
import torch.distributed  # 导入PyTorch的分布式通信模块  
from torch.distributed.distributed_c10d import (  # 导入分布式通信的底层C++接口（c10d）  
    Backend,   
    PrefixStore,    
    Store,    
    _new_process_group_helper,    
    _store_based_barrier,    
    _world,   
    default_pg_timeout,    
    rendezvous,    
)  
  
# init_process_group函数，用于初始化分布式进程组 
def init_process_group(  
    backend: Union[str, Backend] = None,    
    init_method: Optional[str] = None,    
    timeout: Optional[timedelta] = None,  
    world_size: int = -1,    
    rank: int = -1,    
    store: Optional[Store] = None,    
    group_name: str = None,    
    pg_options: Optional[Any] = None,    
):  
    # 断言确保不会同时指定init_method和store  
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."  
  
    # 如果使用store，则确保world_size和rank被正确设置  
    if store is not None:  
        assert world_size > 0, "world_size must be positive if using store"  
        assert rank >= 0, "rank must be non-negative if using store"  
    # 如果未指定init_method，则默认使用环境变量作为初始化方法  
    elif init_method is None:  
        init_method = "env://"  
  
    # 将backend字符串转换为Backend枚举，或保留为未定义  
    if backend:  
        backend = Backend(backend)  
    else:  
        backend = Backend("undefined")  
  
    # 设置超时时间为默认值，如果未指定  
    if timeout is None:  
        timeout = default_pg_timeout  
  
    # 如果没有提供store，则使用rendezvous函数进行进程间的初始化和同步  
    if store is None:  
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)  
        store, rank, world_size = next(rendezvous_iterator)  # 获取store、rank和world_size  
        store.set_timeout(timeout)  # 设置store的超时时间  
  
        # 使用PrefixStore来避免不同系统间的键冲突  
        store = PrefixStore(group_name, store)  
  
    # 调用辅助函数创建新的进程组  
    pg, _ = _new_process_group_helper(  
        world_size,  
        rank,  
        [], 
        backend,  
        store,  
        group_name=group_name,  
        pg_options=pg_options,  
        timeout=timeout,  
    )  
  
    # 将新创建的进程组添加到全局世界对象中  
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}  
  
    # 返回创建的进程组对象  
    return pg