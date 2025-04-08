export CUDA_DEVICE_MAX_CONNECTIONS=1

blobkey="?sv=2023-01-03&st=2025-04-05T07%3A36%3A15Z&se=2025-04-12T07%3A36%3A00Z&skoid=568e5914-ecc1-47fe-b4a8-4007497b49e5&sktid=72f988bf-86f1-41af-91ab-2d7cd011db47&skt=2025-04-05T07%3A36%3A15Z&ske=2025-04-12T07%3A36%3A00Z&sks=b&skv=2023-01-03&sr=c&sp=racwdxltf&sig=Hrif4bKQwNPd2HM8zAla%2BzBUN5z%2Bqxd61ZKWd3m8%2FR4%3D"

TRAJECTORY_GROUP=top_1000
TARGET_TRAJECTORY_DIR=top_1000_trajectory_95_dynamic_step_75

MODEL_DIR=/mnt/blob-hptrainingwesteurope-pretraining/tuning_result/llama_160m_data_sampling_dclm_math/$TRAJECTORY_GROUP/

folders=($(find $MODEL_DIR$TARGET_TRAJECTORY_DIR -type d -name "*iter_*"))
model_count=${#folders[@]}

# echo ${folders[0]}
# echo ${model_count}

TOKENIZER_ARGS=(
    --tokenizer-model /mnt/blob-hptrainingwesteurope-pretraining/Llama-3-8B
    # --tokenizer-model /mnt/mydata/klyang/GK4V16-Q6144-C4096-M10B-lr5e-5-B16M-Phiv2-1016-retry4-90k
    --tokenizer-type HuggingFaceTokenizer
)

mkdir /scratch/target_model
chmod 777 /scratch/target_model

for model_id in $(seq 0 $(($model_count-1))); do
# for model_id in $(seq 0 $((1))); do
    (
        cur_ckpt_dir=${folders[$model_id]}
        IFS='/' read -ra CKPT_NAME <<< "${cur_ckpt_dir}"
        ckpt_name=${CKPT_NAME[-1]}
        IFS='_' read -ra ITERATION <<< "${ckpt_name}"
        cur_iter=$((10#${ITERATION[-1]}))

        ./azcopy copy --recursive "https://hptrainingwesteurope.blob.core.windows.net/pretraining/tuning_result/llama_160m_data_sampling_dclm_math/"$TRAJECTORY_GROUP"/"$TARGET_TRAJECTORY_DIR"/"$ckpt_name"/"$blobkey "/scratch/target_model/"
        echo $cur_iter > /scratch/target_model/latest_checkpointed_iteration.txt

        ls /scratch/target_model

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
            --micro-batch-size 4
            --results-path /mnt/blob-hptrainingwesteurope-pretraining-out/evaluation_results/llama_160m_data_sampling_dclm_math_tra_eval/$TRAJECTORY_GROUP/$TARGET_TRAJECTORY_DIR/$ckpt_name"_"$model_id"_mmlu_math.json"
            --task-list mmlu_continuation,math_continuation
            --num-fewshot 1
            --trust-remote-code
        )

        echo ${MODEL_ARGS[@]}
        echo ${INFERENCE_SPECIFIC_ARGS[@]}

        accelerate launch --main_process_port 29500 evaluate_regmix.py \
            ${TOKENIZER_ARGS[@]} \
            ${MODEL_ARGS[@]} \
            ${INFERENCE_SPECIFIC_ARGS[@]}
        
        # INFERENCE_SPECIFIC_ARGS=(
        #     --attention-dropout 0.0
        #     --hidden-dropout 0.0
        #     --micro-batch-size 8
        #     --results-path /mnt/blob-hptrainingwesteurope-pretraining-out/evaluation_results/llama_160m_data_sampling_dclm_math_tra_eval/$TRAJECTORY_GROUP/$TARGET_TRAJECTORY_DIR/$ckpt_name"_"$model_id/math.json
        #     --task-list math_continuation
        #     --num-fewshot 1
        #     --trust-remote-code
        # )

        # echo ${INFERENCE_SPECIFIC_ARGS[@]}
        
        # accelerate launch --main_process_port 29501 evaluate_regmix.py \
        #     ${TOKENIZER_ARGS[@]} \
        #     ${MODEL_ARGS[@]} \
        #     ${INFERENCE_SPECIFIC_ARGS[@]}

        rm -rf /scratch/target_model/*
    )
done
