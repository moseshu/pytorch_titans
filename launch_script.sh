#!/bin/bash


# set env
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  
export OMP_NUM_THREADS=1

# use accelerate launch start training (DDP Mode)
# accelerate launch \
#     --config_file accelerate_config.yaml \
#     train_mac_distributed.py

# 
# accelerate launch \
#     --multi_gpu \
#     --num_processes=4 \
#     --mixed_precision=bf16 \
#     train_mac_distributed.py

# 
accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    --use_fsdp \
    --fsdp_sharding_strategy=FULL_SHARD \
    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
    --fsdp_backward_prefetch=BACKWARD_PRE \
    train_mac_distributed.py
