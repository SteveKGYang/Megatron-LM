#!/bin/bash
LOAD_DIR='/mnt/pvc-blob-nfs/klyang/tuning_result/llama_3B_data_evaluation_finewebedu_0214_mid/'
TARGET_DIR='/mnt/pvc-blob-nfs/klyang/tuning_result/llama_3B_data_evaluation_finewebedu_0214_mid_hf/'

PATTERN="tokenizer*.json"

echo "Start converting..."
# python tools/checkpoint/convert.py --saver core --model-type GPT --position-embedding-type rope --load-dir $LOAD_DIR --save-dir $TARGET_DIR --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 --megatron-path ./
# python tools/checkpoint/convert.py --loader core --model-type GPT --position-embedding-type rope --load-dir $LOAD_DIR --save-dir /scratch/torch_model --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 --megatron-path ./
# python tools/checkpoint/convert.py --model-type GPT --position-embedding-type rope --load-dir $LOAD_DIR --save-dir /scratch/torch_model --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 --megatron-path ./
python weights_conversion/megatron_to_hf.py --input_dir $LOAD_DIR --num_output_shards 1 --output_dir $TARGET_DIR
# cp /mnt/pvc-blob-nfs/xiaoliu2/Sigma1-10b/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k/$PATTERN $TARGET_DIR
echo "End converting..."