
task=$1 # one of [msrvtt7k,didemo,lsmdc,msrvtt9k,msvd]
eval=$3
output_dir=output/${task}
mkdir ${output_dir}
python -m torch.distributed.launch --nproc_per_node=8 --master_port 456 \
--use_env retrieval.py \
--pretrained $2 \
--config configs/retrieval_${task}.yaml \
--evaluate ${eval} \
--output_dir ${output_dir}
