#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

LOAD_DIR='/home/v-yucding/blob_ruizhe/blob_files/checkpoints/deepseek_moe_debug/moe_64exp_lite_dclm_1T_18node_0219_1/iter_0007000'
TARGET_DIR='/home/v-yucding/blob_ruizhe/blob_files/checkpoints/deepseek_moe_debug/convert_model_debug/moe_64exp_lite'

PATTERN="tokenizer*.json"

echo "Start converting..."
python tools/checkpoint/moe_convert_debug.py --loader core --saver core --model-type GPT --position-embedding-type rope --load-dir $LOAD_DIR --save-dir $TARGET_DIR --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 --megatron-path ./ --max-queue-size 1
echo "End converting..."