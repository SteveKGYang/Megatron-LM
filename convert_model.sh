#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

LOAD_DIR='/mnt/blob-hptrainingwesteurope-pretraining/tuning_result/data_agent_0622_sft_nvidia_agent_model_pythia/'
TARGET_DIR='/mnt/blob-hptrainingwesteurope-pretraining-out/tuning_result/data_agent_0622_sft_nvidia_agent_model_pythia_tp1_core/'

PATTERN="tokenizer*.json"

echo "Start converting..."
python tools/checkpoint/convert.py --loader core --saver core --model-type GPT --position-embedding-type rope --load-dir $LOAD_DIR --save-dir $TARGET_DIR --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 --megatron-path ./
# python tools/checkpoint/convert.py --loader core --saver legacy --model-type GPT --position-embedding-type rope --load-dir $LOAD_DIR --save-dir $TARGET_DIR --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 --megatron-path ./
# python tools/checkpoint/convert.py --loader core --saver core --model-type GPT --position-embedding-type rope --load-dir $LOAD_DIR --save-dir /scratch/torch_model --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 --megatron-path ./
# python tools/checkpoint/convert.py --model-type GPT --position-embedding-type rope --load-dir $LOAD_DIR --save-dir /scratch/torch_model --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 --megatron-path ./
# python weights_conversion/megatron_to_hf.py --input_dir $LOAD_DIR --num_output_shards 1 --output_dir $TARGET_DIR
# cp /mnt/pvc-blob-nfs/xiaoliu2/Sigma1-10b/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k/$PATTERN $TARGET_DIR
echo "End converting..."