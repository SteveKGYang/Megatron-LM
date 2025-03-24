# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
# alias python='/root/.local/lib/python3.12'
# test

# export TARGET_TRAJECTORY_DIR=/mnt/blob-hptrainingwesteurope-pretraining-out/tuning_result/llama_3B_data_sampling_dclm_math/top_1_trajectory_0_dynamic_step_71
export TARGET_TRAJECTORY_DIR=/home/pretraining/klyang/mount_dir/mount/tuning_result/llama_3B_data_sampling_dclm_math/top_1_trajectory_0_dynamic_step_71
folders=($(find $TARGET_TRAJECTORY_DIR -type d -name "*iter_*"))

TOKENIZER_ARGS=(
    --tokenizer-model /mnt/blob-hptrainingwesteurope-pretraining/Llama-3-8B
    # --tokenizer-model /mnt/mydata/klyang/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    --tokenizer-type HuggingFaceTokenizer
)

for ckpt_dir in "${folders[@]}"; do
    IFS='/' read -ra CKPT_NAME <<< "${ckpt_dir}"
    IFS='_' read -ra STEP <<< "${CKPT_NAME[-1]}"
    CUR_STEP=${STEP[-1]}
    echo $CUR_STEP

    MODEL_ARGS=(
        --use-checkpoint-args
        --use-mcore-models
        --no-load-rng
        --bf16
        --tensor-model-parallel-size 1
        --load /scratch/target_model/
    )

    INFERENCE_SPECIFIC_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --micro-batch-size 12
        --results-path /mnt/blob-hptrainingwesteurope-pretraining-out/regmix_results_test/test_$CUR_STEP.json
        # --results-path /mnt/mydata/klyang/results_olmo_replicate_mmlu_continuation.json
        # --task-list hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa,sciq,logiqa,lambada
        # --task-list gsm8k,mmlu_continuation
        --task-list math_continuation
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
