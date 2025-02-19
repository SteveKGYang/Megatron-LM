export CUDA_DEVICE_MAX_CONNECTIONS=1

TOKENIZER_ARGS=(
    --tokenizer-model /mnt/pvc-blob-nfs/xiaoliu2/Sigma1-10b/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    # --tokenizer-model /home/pretraining/klyang/mount_dir/eu_mount/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    --tokenizer-type HuggingFaceTokenizer
)

MODEL_ARGS=(
    --use-checkpoint-args
    --use-mcore-models
    --bf16
    --tensor-model-parallel-size 1
    --load /mnt/pvc-blob-nfs/klyang/tuning_result/llama_3B_data_evaluation_finewebedu_0214_mid_core_from_legacy/
    # --load /home/pretraining/klyang/mount_dir/eu_mount/llama_3B_data_evaluation_finewebedu_0214_mid
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --micro-batch-size 1
)

torchrun --nproc-per-node=1 evaluate.py \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]}