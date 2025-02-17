export CUDA_DEVICE_MAX_CONNECTIONS=1

TOKENIZER_ARGS=(
    --tokenizer-model /mnt/pvc-blob-nfs/xiaoliu2/Sigma1-10b/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    --tokenizer-type HuggingFaceTokenizer
)

MODEL_ARGS=(
    --use-checkpoint-args
    --use-mcore-models
    --tensor-model-parallel-size 1
    --load /mnt/pvc-blob-nfs/klyang/tuning_result/llama_3B_data_evaluation_finewebedu_0214_mid_core_from_legacy/
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --num-tokens-to-generate 512
    --max-batch-size 1
    --temperature 0
)

torchrun --nproc-per-node=1 gpt_batch_inference.py \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]} \
    --prompts "An example run script is shown below."