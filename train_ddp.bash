export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --nproc_per_node=4 main_ddp.py --config /ai/gno/CODE/DiffSR/configs/ns2d/m2no.yaml