export CUDA_DEVICE_MAX_CONNECTIONS=1

export AZUREML_NODE_COUNT=${AZUREML_NODE_COUNT:=1}
export GPU_PER_NODE_COUNT=${GPU_PER_NODE_COUNT:=4}
export NODE_RANK=${NODE_RANK:=0}
export MASTER_ADDR=${MASTER_ADDR:=localhost}
export MASTER_PORT=${MASTER_PORT:=1828}

TOKENIZER_ARGS=(
    --tokenizer-model /mnt/pvc-blob-nfs/xiaoliu2/Sigma1-10b/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    --tokenizer-type HuggingFaceTokenizer
)

MODEL_ARGS=(
    --use-checkpoint-args
    --use-mcore-models
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 5
    --expert-model-parallel-size 4
    --decoder-first-pipeline-num-layers 4
    --decoder-last-pipeline-num-layers 4
    --context-parallel-size 1
    --expert-tensor-parallel-size 1
    --bf16
    --load /mnt/pvc-blob-nfs/klyang/tuning_result/moe_ckpt_test/iter_0002000
    --sequence-parallel
    --ckpt-format torch
    --qk-layernorm
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 0.001
    --moe-router-score-function sigmoid
    # --distributed-timeout-minutes 120
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --num-tokens-to-generate 512
    --max-batch-size 1
    --temperature 0
)

torchrun --nproc_per_node=$GPU_PER_NODE_COUNT --nnodes=$AZUREML_NODE_COUNT --node_rank=$NODE_RANK  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT gpt_batch_inference.py \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]} \
    --prompts "An example run script is shown below."