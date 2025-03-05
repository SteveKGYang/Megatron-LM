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
    --load /mnt/pvc-blob-nfs/klyang/tuning_result/llama_3B_data_evaluation_nemotron_HQ_0303_tp1_core/
    # --load /mnt/mydata/klyang/olmo2_replicate_0207_format_torch_tp1_core
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --micro-batch-size 12
    # --results-path /mnt/pvc-blob-nfs/klyang/regmix_results/2.json
    --results-path /mnt/mydata/klyang/llama_3B_data_evaluation_nemotron_HQ_0303_few_shot.json
    --task-list hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa,sciq,logiqa,lambada
    # --task-list gsm8k,mmlu_continuation,mmlu_pro_math
    # --task-list gsm8k,mmlu_continuation,mmlu_pro
    --num-fewshot 0
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

accelerate launch --main_process_port 29501 evaluate.py \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]}
