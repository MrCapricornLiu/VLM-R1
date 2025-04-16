# $ pwd
# /data/home/yangxiaoda/lch/VLM-R1/src/open-r1-multimodal

cd /data/home/yangxiaoda/lch/VLM-R1/src/open-r1-multimodal

export DEBUG_MODE="true"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

RUN_NAME="Qwen2.5-VL-7B-GRPO-REC-patch-control-enhanced"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

# 图像路径设置（确保这是COCO图像的正确路径）
IMAGE_ROOT="/data/home/yangxiaoda/lch/datasets"

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_rec_patch.py \
    --deepspeed local_scripts/zero2.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name data_config/rec_patched.yaml \
    --image_root $IMAGE_ROOT \
    --max_prompt_length 2048 \
    --num_generations 4 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 5 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --use_peft true \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_task_type CAUSAL_LM \
    --reward_funcs accuracy adaptive_format \
    --freeze_vision_modules false \
    --no_cot_template "Answer this question about the image: {Question} Provide only the answer in JSON format." 