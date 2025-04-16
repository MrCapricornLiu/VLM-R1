#!/bin/bash

# 设置模型和数据路径
MODEL_PATH="/data/home/yangxiaoda/lch/VLM-R1/src/open-r1-multimodal/output/Qwen2.5-VL-7B-GRPO-REC-patch-control-refcoco-only/checkpoint-500" # 替换为实际的checkpoint路径
DATA_JSON="/data/home/yangxiaoda/lch/datasets/rec_jsons_processed/refcoco_val.json"
IMAGE_ROOT="/data/home/yangxiaoda/lch/datasets"
OUTPUT_PATH="/data/home/yangxiaoda/lch/VLM-R1/src/open-r1-multimodal/output/Qwen2.5-VL-7B-GRPO-REC-patch-control-refcoco-only-results/patch_control_refcoco_only_results2.json"
NUM_SAMPLES=100
BATCH_SIZE=4

# 确保结果目录存在
mkdir -p $(dirname $OUTPUT_PATH)

# 使用分布式启动评估脚本
torchrun --nproc_per_node=2 \
    --master_port=12347 \
    test_rec_patch_control.py \
    --model_path $MODEL_PATH \
    --data_json $DATA_JSON \
    --image_root $IMAGE_ROOT \
    --output_path $OUTPUT_PATH \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE 