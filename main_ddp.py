import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import parser
from utils.helper import set_up_logger, set_seed, set_device, load_config, get_dir_path, save_config
from trainers import _trainer_dict


def main():
    # ============ Step 1. 初始化分布式环境 ============
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # ============ Step 2. 解析参数与加载配置 ============
    args = parser.parse_args()
    args = vars(args)
    args = load_config(args)
    args['train']['local_rank'] = local_rank
    args['train']['world_size'] = dist.get_world_size()
    args['train']['rank'] = dist.get_rank()

    # ============ Step 3. 只在 rank=0 初始化日志与保存配置 ============
    if dist.get_rank() == 0:
        saving_path, saving_name = set_up_logger(args)
        save_config(args, saving_path)
    else:
        saving_path, saving_name = None, None
    
    payload = [saving_path, saving_name]
    dist.broadcast_object_list(payload, src=0)
    saving_path, saving_name = payload
        
    args['train']['saving_path'] = saving_path
    args['train']['saving_name'] = saving_name

    # ============ Step 4. 固定随机种子与设备 ============
    set_seed(args['train'].get('seed', 42))
    torch.cuda.set_device(local_rank)

    # ============ Step 5. 构建 trainer（内部会构建 model、dataloader） ============
    trainer = _trainer_dict[args['model']['name']](args)

    # ============ Step 6. 启动训练 ============
    trainer.process()

    # ============ Step 7. 关闭分布式环境 ============
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
