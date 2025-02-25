# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
# alias python='/root/.local/lib/python3.12'


TOKENIZER_ARGS=(
    # --tokenizer-model /mnt/pvc-blob-nfs/xiaoliu2/Sigma1-10b/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    --tokenizer-model /mnt/mydata/klyang/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    --tokenizer-type HuggingFaceTokenizer
)

MODEL_ARGS=(
    --use-checkpoint-args
    --use-mcore-models
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 4
    --expert-model-parallel-size 2
    --decoder-first-pipeline-num-layers 6
    --decoder-last-pipeline-num-layers 6
    --context-parallel-size 1
    --expert-tensor-parallel-size 1
    --bf16
    --load /mnt/pvc-blob-nfs/klyang/tuning_result/moe_64exp_tiny_dclm_1T_128node_0222_test
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
    --micro-batch-size 8
    # --results-path /mnt/pvc-blob-nfs/klyang/results_llama3B_dclm.json
    --results-path /mnt/mydata/klyang/results_olmo_replicate.json
    --task-list hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa,sciq,logiqa,lambada
    # --task-list mmlu
    --num-fewshot 0
    --trust-remote-code
)

# torchrun --nproc-per-node=4 evaluate.py \
#     ${TOKENIZER_ARGS[@]} \
#     ${MODEL_ARGS[@]} \
#     ${INFERENCE_SPECIFIC_ARGS[@]}

 /root/.local/bin/accelerate launch evaluate.py \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]}