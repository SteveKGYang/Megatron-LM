export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_MAX_CONNECTIONS=1

TOKENIZER_ARGS=(
    --tokenizer-model /mnt/pvc-blob-nfs/xiaoliu2/Sigma1-10b/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    # --tokenizer-model /home/pretraining/klyang/mount_dir/eu_mount/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    --tokenizer-type HuggingFaceTokenizer
)

MODEL_ARGS=(
    --use-checkpoint-args
    --use-mcore-models
    --no-load-rng
    --bf16
    --tensor-model-parallel-size 4
    --load /mnt/pvc-blob-nfs/klyang/tuning_result/llama_3B_data_evaluation_dclm_0215
    # --load /home/pretraining/klyang/mount_dir/eu_mount/llama_3B_data_evaluation_finewebedu_0214_mid
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --micro-batch-size 4
    --results-path /mnt/pvc-blob-nfs/klyang
)

torchrun --nproc-per-node=4 evaluate.py \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]}

# accelerate launch evaluate.py \
#     ${TOKENIZER_ARGS[@]} \
#     ${MODEL_ARGS[@]} \
#     ${INFERENCE_SPECIFIC_ARGS[@]}