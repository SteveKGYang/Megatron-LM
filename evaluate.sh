# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_TIMEOUT=60000
# alias python='/root/.local/lib/python3.12'

TOKENIZER_ARGS=(
    --tokenizer-model /mnt/blob-hptrainingwesteurope-pretraining/Llama-3-8B
    # --tokenizer-model /mnt/mydata/klyang/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    --tokenizer-type HuggingFaceTokenizer
)

MODEL_ARGS=(
    --use-checkpoint-args
    --use-mcore-models
    --no-load-rng
    --bf16
    --tensor-model-parallel-size 1
    --load /mnt/blob-hptrainingwesteurope-pretraining/tuning_result/data_agent_0604_sft_2D_agent_model_corrected_tp1_core/
    # --load /mnt/blob-hptrainingwesteurope-pretraining/tuning_result/data_agent_0501_epoch_2_smooth_tp1_core/
    # --load /mnt/blob-hptrainingwesteurope-pretraining/tuning_results/llama_3B_data_evaluation_nemotron_HQ_0303_tp1_core/
    # --load /mnt/mydata/klyang/olmo2_replicate_0207_format_torch_tp1_core
)

INFERENCE_SPECIFIC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --micro-batch-size 1
    # --results-path /mnt/pvc-blob-nfs/klyang/regmix_results/2.json
    --results-path /mnt/blob-hptrainingwesteurope-pretraining-out/evaluation_results/data_agent_0604_sft_2D_agent_model_corrected_tp1_core_gsm8k.json
    # --task-list hellaswag,openbookqa,winogrande,arc_easy,arc_challenge,boolq,piqa,sciq,logiqa,lambada
    # --task-list piqa,sciq,logiqa,lambada
    --task-list gsm8k
    # --task-list minerva_math,gsm8k
    --task-list mmlu_continuation
    # --task-list mmlu
    --num-fewshot 5
    --trust-remote-code
)

# INFERENCE_SPECIFIC_ARGS=(
#         --attention-dropout 0.0
#         --hidden-dropout 0.0
#         --micro-batch-size 1
#         --results-path /mnt/blob-hptrainingwesteurope-pretraining-out/evaluation_results/llama_160M_dclm_data_evaluation_0322_data_agent.json
#         --task-list mmlu_continuation,math_continuation
#         --num-fewshot 1
#         --trust-remote-code
# )

# torchrun --nproc-per-node=4 evaluate.py \
#     ${TOKENIZER_ARGS[@]} \
#     ${MODEL_ARGS[@]} \
#     ${INFERENCE_SPECIFIC_ARGS[@]}

# /home/aiscuser/.local/bin/accelerate launch evaluate.py \
#     ${TOKENIZER_ARGS[@]} \
#     ${MODEL_ARGS[@]} \
#     ${INFERENCE_SPECIFIC_ARGS[@]}

accelerate launch --main_process_port 29502 evaluate.py \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${INFERENCE_SPECIFIC_ARGS[@]}

# accelerate launch --main_process_port 29503 evaluate_regmix.py \
#     ${TOKENIZER_ARGS[@]} \
#     ${MODEL_ARGS[@]} \
#     ${INFERENCE_SPECIFIC_ARGS[@]}
