# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
# alias python='/root/.local/lib/python3.12'


TOKENIZER_ARGS=(
    --tokenizer-model /mnt/pvc-blob-nfs/xiaoliu2/Sigma1-10b/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    # --tokenizer-model /mnt/mydata/klyang/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    --tokenizer-type HuggingFaceTokenizer
)

MODEL_ARGS=(
    --use-checkpoint-args
    --use-mcore-models
    --no-load-rng
    --bf16
    --tensor-model-parallel-size 1
    --load /mnt/pvc-blob-nfs/klyang/tuning_result/regmix/llama_3B_dclm_math_0d7_0m3_2/
    # --load /mnt/mydata/klyang/olmo2_replicate_0207_format_torch_tp1_core
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --micro-batch-size 12
    --results-path /mnt/pvc-blob-nfs/klyang/regmix_results/2.json
    # --results-path /mnt/mydata/klyang/results_olmo_replicate_mmlu_continuation.json
    # --task-list hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa,sciq,logiqa,lambada
    --task-list gsm8k,mmlu_continuation,mmlu_pro_math
    --num-fewshot 5
    --trust-remote-code
)

# torchrun --nproc-per-node=4 evaluate.py \
#     ${TOKENIZER_ARGS[@]} \
#     ${MODEL_ARGS[@]} \
#     ${INFERENCE_SPECIFIC_ARGS[@]}

#  /root/.local/bin/accelerate launch evaluate.py \
#     ${TOKENIZER_ARGS[@]} \
#     ${MODEL_ARGS[@]} \
#     ${INFERENCE_SPECIFIC_ARGS[@]}

for split_id in $(seq 4 $((64))); do
    
    MODEL_ARGS=(
        --use-checkpoint-args
        --use-mcore-models
        --no-load-rng
        --bf16
        --tensor-model-parallel-size 1
        --load /mnt/pvc-blob-nfs/klyang/tuning_result/regmix/llama_3B_dclm_math_$split_id/
    )

    INFERENCE_SPECIFIC_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --micro-batch-size 12
        --results-path /mnt/pvc-blob-nfs/klyang/regmix_results/$split_id.json
        # --results-path /mnt/mydata/klyang/results_olmo_replicate_mmlu_continuation.json
        # --task-list hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa,sciq,logiqa,lambada
        --task-list gsm8k,mmlu_continuation,mmlu_pro_math
        # --task-list gsm8k,mmlu_pro_math
        --num-fewshot 5
        --trust-remote-code
    )

    accelerate launch --main_process_port 29500 evaluate.py \
        ${TOKENIZER_ARGS[@]} \
        ${MODEL_ARGS[@]} \
        ${INFERENCE_SPECIFIC_ARGS[@]}
done
