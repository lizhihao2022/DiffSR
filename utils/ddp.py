import os
import torch.distributed as dist

from .helper import get_dir_path


def debug_barrier(tag):
    if dist.is_initialized():
        print(f"[Rank {dist.get_rank()}] reached {tag}", flush=True)
        dist.barrier()
        print(f"[Rank {dist.get_rank()}] passed {tag}", flush=True)


def make_shared_run_dir(args):
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    if rank == 0:
        # 仅 rank0 生成并创建目录
        dir_path, dir_name = get_dir_path(args, create_dir=True)
    else:
        dir_path, dir_name = None, None

    if is_dist:
        payload = [dir_path, dir_name]
        dist.broadcast_object_list(payload, src=0)
        dir_path, dir_name = payload  # 所有 rank 得到完全相同的字符串

    # 写回 args 供后续使用
    args['train']['saving_path'] = dir_path
    args['train']['saving_name'] = dir_name
    return dir_path, dir_name
