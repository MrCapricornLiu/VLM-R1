from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import json
from tqdm import tqdm
import re
import os
import random
from PIL import Image
import numpy as np
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return local_rank, world_size, rank

def add_patch(image, patch_size=40, position='bottom-right', color=(255, 0, 0)):
    """在图像上添加一个色块作为patch"""
    img_copy = image.copy()
    width, height = img_copy.size
    
    if position == 'bottom-right':
        x1, y1 = width - patch_size - 10, height - patch_size - 10
    elif position == 'top-left':
        x1, y1 = 10, 10
    else:  # random position
        x1 = np.random.randint(0, width - patch_size)
        y1 = np.random.randint(0, height - patch_size)
    
    x2, y2 = x1 + patch_size, y1 + patch_size
    
    # 创建patch
    for x in range(x1, x2):
        for y in range(y1, y2):
            if 0 <= x < width and 0 <= y < height:
                img_copy.putpixel((x, y), color)
    
    return img_copy

def extract_bbox_answer(content):
    # 尝试在<answer>标签内找到边界框
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer, re.DOTALL)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
            return bbox
    return [0, 0, 0, 0]

def has_think_tag(content):
    # 检查输出是否包含<think>标签
    think_tag_pattern = r'<think>(.*?)</think>'
    return re.search(think_tag_pattern, content, re.DOTALL) is not None

def iou(box1, box2):
    # 计算两个边界框的IoU
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

def main():
    parser = argparse.ArgumentParser(description="Test patch-based CoT control")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_json", type=str, required=True, help="Path to test data json")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory of images")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save results")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()

    # 设置分布式环境
    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"
    print(f"Process {rank} using {device}")

    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank}, 
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    # 加载测试数据
    with open(args.data_json, "r") as f:
        data = json.load(f)
    
    random.seed(42)
    random.shuffle(data)
    data = data[:args.num_samples]

    # 为每个样本创建原始和带patch版本
    # QUESTION_TEMPLATE = "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
    QUESTION_TEMPLATE = "{Question} Output the final answer in <answer> </answer> tags. Output the final answer in JSON format."


    # 分割数据以进行分布式评估
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    # 准备原始和带patch的消息
    original_messages = []
    patched_messages = []

    for x in rank_data:
        image_path = os.path.join(args.image_root, x['image'])
        
        # 原始图像消息
        original_message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": f"file://{image_path}"
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                    }
                ]
            }
        ]
        original_messages.append(original_message)
        
        # 带patch的图像消息
        try:
            # 加载图像并添加patch
            image = Image.open(image_path).convert("RGB")
            patched_image = add_patch(image)
            
            # 保存带patch的图像到临时目录
            os.makedirs("tmp_patched", exist_ok=True)
            patched_image_path = f"tmp_patched/patched_{os.path.basename(image_path)}"
            patched_image.save(patched_image_path)
            
            patched_message = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": f"file://{patched_image_path}"
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=x['problem'])
                        }
                    ]
                }
            ]
            patched_messages.append(patched_message)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # 如果有错误，添加相同的消息作为占位符
            patched_messages.append(original_message)

    # 处理原始图像
    original_outputs = []
    for i in tqdm(range(0, len(original_messages), args.batch_size), disable=rank != 0, desc="Processing original images"):
        batch_messages = original_messages[i:i + args.batch_size]
        
        # 准备推理
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # 生成输出
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        original_outputs.extend(batch_output_text)

    # 处理带patch的图像
    patched_outputs = []
    for i in tqdm(range(0, len(patched_messages), args.batch_size), disable=rank != 0, desc="Processing patched images"):
        batch_messages = patched_messages[i:i + args.batch_size]
        
        # 准备推理
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # 生成输出
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=256, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        patched_outputs.extend(batch_output_text)

    # 收集所有进程的结果
    all_original_outputs = [None] * len(data)
    all_patched_outputs = [None] * len(data)
    
    original_results = [(start_idx + i, output) for i, output in enumerate(original_outputs)]
    patched_results = [(start_idx + i, output) for i, output in enumerate(patched_outputs)]

    gathered_original = [None] * world_size
    gathered_patched = [None] * world_size
    
    dist.all_gather_object(gathered_original, original_results)
    dist.all_gather_object(gathered_patched, patched_results)
    
    # 主进程处理所有结果
    if rank == 0:
        # 合并结果
        for results in gathered_original:
            for idx, output in results:
                all_original_outputs[idx] = output
                
        for results in gathered_patched:
            for idx, output in results:
                all_patched_outputs[idx] = output
        
        # 分析结果
        cot_original_count = 0
        cot_patched_count = 0
        correct_original_count = 0
        correct_patched_count = 0
        original_no_cot_count = 0  # 原始图像不使用CoT的数量
        patched_with_cot_count = 0  # 带patch图像使用CoT的数量
        format_control_success = 0  # 格式控制成功的数量
        
        results = []
        
        for i, (input_example, original_output, patched_output) in enumerate(zip(data, all_original_outputs, all_patched_outputs)):
            ground_truth = input_example['solution']
            
            # 提取边界框和检查是否有思考标签
            original_bbox = extract_bbox_answer(original_output)
            patched_bbox = extract_bbox_answer(patched_output)
            
            has_cot_original = has_think_tag(original_output)
            has_cot_patched = has_think_tag(patched_output)
            
            # 统计CoT使用情况
            if has_cot_original:
                cot_original_count += 1
            else:
                original_no_cot_count += 1
                
            if has_cot_patched:
                patched_with_cot_count += 1
            else:
                cot_patched_count += 1
            
            # 格式控制成功判断
            if has_cot_original and not has_cot_patched:
                format_control_success += 1
            
            # 检查答案正确性
            original_correct = iou(original_bbox, ground_truth) > 0.5
            patched_correct = iou(patched_bbox, ground_truth) > 0.5
            
            if original_correct:
                correct_original_count += 1
            if patched_correct:
                correct_patched_count += 1
            
            # 创建结果条目
            result = {
                'id': i,
                'image': input_example['image'],
                'question': input_example['problem'],
                'ground_truth': ground_truth,
                'original_output': {
                    'text': original_output,
                    'has_cot': has_cot_original,
                    'bbox': original_bbox,
                    'correct': original_correct
                },
                'patched_output': {
                    'text': patched_output,
                    'has_cot': has_cot_patched,
                    'bbox': patched_bbox,
                    'correct': patched_correct
                }
            }
            results.append(result)
        
        # 计算并打印统计数据
        total = len(data)
        format_control_rate = format_control_success / total * 100
        print(f"\n========== Results ==========")
        print(f"Total samples: {total}")
        print(f"Original images with CoT: {cot_original_count} ({cot_original_count/total*100:.2f}%)")
        print(f"Patched images without CoT: {cot_patched_count} ({cot_patched_count/total*100:.2f}%)")
        print(f"Format control success rate: {format_control_rate:.2f}%")
        print(f"Original images WITHOUT CoT (failure): {original_no_cot_count} ({original_no_cot_count/total*100:.2f}%)")
        print(f"Patched images WITH CoT (failure): {patched_with_cot_count} ({patched_with_cot_count/total*100:.2f}%)")
        print(f"Original images correct: {correct_original_count} ({correct_original_count/total*100:.2f}%)")
        print(f"Patched images correct: {correct_patched_count} ({correct_patched_count/total*100:.2f}%)")
        
        # 保存结果
        output = {
            "statistics": {
                "total": total,
                "original_cot": cot_original_count,
                "patched_no_cot": cot_patched_count,
                "original_correct": correct_original_count,
                "patched_correct": correct_patched_count,
                "format_control_success": format_control_success,
                "format_control_rate": format_control_rate,
                "original_cot_percentage": cot_original_count/total*100,
                "patched_no_cot_percentage": cot_patched_count/total*100,
                "original_accuracy": correct_original_count/total*100,
                "patched_accuracy": correct_patched_count/total*100,
            },
            "results": results
        }
        
        # 创建输出目录
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        # 保存结果到JSON文件
        with open(args.output_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {args.output_path}")
    
    # 同步所有进程
    dist.barrier()
    
    # 清理临时文件
    if rank == 0 and os.path.exists("tmp_patched"):
        import shutil
        shutil.rmtree("tmp_patched")

if __name__ == "__main__":
    main() 