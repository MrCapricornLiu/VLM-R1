#!/bin/bash

# 设置路径变量
IMAGE_ROOT="/data/home/yangxiaoda/lch/datasets"
PATCHED_IMAGE_DIR="/data/home/yangxiaoda/lch/datasets/train2014_patched"

# 确保输出目录存在
mkdir -p $PATCHED_IMAGE_DIR

# 处理RefCOCO+数据集
python /data/home/yangxiaoda/lch/VLM-R1/src/create_patched_dataset.py \
    --input_json "/data/home/yangxiaoda/lch/datasets/rec_jsons_processed/refcocop_train.json" \
    --image_root $IMAGE_ROOT \
    --output_json "/data/home/yangxiaoda/lch/datasets/rec_jsons_processed/refcocop_train_patched.json" \
    --output_image_dir $PATCHED_IMAGE_DIR

# 处理RefCOCOg数据集
python /data/home/yangxiaoda/lch/VLM-R1/src/create_patched_dataset.py \
    --input_json "/data/home/yangxiaoda/lch/datasets/rec_jsons_processed/refcocog_train.json" \
    --image_root $IMAGE_ROOT \
    --output_json "/data/home/yangxiaoda/lch/datasets/rec_jsons_processed/refcocog_train_patched.json" \
    --output_image_dir $PATCHED_IMAGE_DIR

echo "处理完成！以下数据集已处理完毕："
echo "- RefCOCO+ (patched): /data/home/yangxiaoda/lch/datasets/rec_jsons_processed/refcocop_train_patched.json"
echo "- RefCOCOg (patched): /data/home/yangxiaoda/lch/datasets/rec_jsons_processed/refcocog_train_patched.json"
echo "- 带补丁的图像: $PATCHED_IMAGE_DIR" 