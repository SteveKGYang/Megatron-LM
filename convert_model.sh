#!/bin/bash
LOAD_DIR='/mnt/pvc-blob-nfs/klyang/tuning_result/temp/'
TARGET_DIR='/mnt/pvc-blob-nfs/klyang/hf_converted_model/llama_3B_data_evaluation_dclm_0215/checkpoint-40000/'

PATTERN="tokenizer*.json"

echo "Start converting..."
python tools/checkpoint/convert.py --loader core --model-type GPT --position-embedding-type rope --load-dir $LOAD_DIR --save-dir /scratch/torch_model --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1 --megatron-path ./
python weights_conversion/megatron_to_hf.py --input_dir /scratch/torch_model --num_output_shards 1 --output_dir $TARGET_DIR
cp /mnt/pvc-blob-nfs/xiaoliu2/Sigma1-10b/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k/$PATTERN $TARGET_DIR
echo "End converting..."