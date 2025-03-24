# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
# alias python='/root/.local/lib/python3.12'
# test

TOKENIZER_ARGS=(
    --tokenizer-model /mnt/blob-hptrainingwesteurope-pretraining/Llama-3-8B
    --tokenizer-type HuggingFaceTokenizer
)

for split_id in $(seq 0 $((267))); do
    
    MODEL_ARGS=(
        --use-checkpoint-args
        --use-mcore-models
        --no-load-rng
        --bf16
        --tensor-model-parallel-size 1
        --load /mnt/blob-hptrainingwesteurope-pretraining/tuning_result/nvidia_domain_regmix/llama_50m_dclm_math_nvidia_domain_$split_id/
    )

    INFERENCE_SPECIFIC_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --micro-batch-size 8
        --results-path /mnt/blob-hptrainingwesteurope-pretraining-out/regmix_results_nvidia_dclm_math/mmlu_$split_id.json
        # --results-path /mnt/mydata/klyang/results_olmo_replicate_mmlu_continuation.json
        # --task-list hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa,sciq,logiqa,lambada
        --task-list mmlu_continuation
        # --task-list math_continuation
        --num-fewshot 5
        --trust-remote-code
    )

    echo ${MODEL_ARGS[@]}
    echo ${INFERENCE_SPECIFIC_ARGS[@]}

    accelerate launch --main_process_port 29500 evaluate_regmix.py \
        ${TOKENIZER_ARGS[@]} \
        ${MODEL_ARGS[@]} \
        ${INFERENCE_SPECIFIC_ARGS[@]}
done
