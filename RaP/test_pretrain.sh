# We use 64 V100 (CUDA=11.1) train Rap. Here is a test script. You can change it to a multi-node version.

python -m torch.distributed.launch --nproc_per_node=1   --master_port 29501  \
    --use_env pretrain.py \
    --config ./configs/pretrain_rap.yaml \
    --output_dir output/pretrain_rap
