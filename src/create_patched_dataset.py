import os
import numpy as np
from PIL import Image
import json
import argparse
from tqdm import tqdm

def add_patch(image, patch_size=20, position='bottom-right', color=(255, 0, 0)):
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

# 处理原始数据集，创建带patch的版本
def create_patched_dataset(original_json_path, image_root, output_json_path, patched_image_dir):
    with open(original_json_path, 'r') as f:
        data = json.load(f)
    
    patched_data = []
    
    os.makedirs(patched_image_dir, exist_ok=True)
    
    for item in tqdm(data, desc="Processing images"):
        # 1. 复制原始数据项
        patched_item = item.copy()
        
        # 2. 加载原始图像
        image_path = os.path.join(image_root, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # 3. 添加patch
        patched_image = add_patch(image)
        
        # 4. 保存带patch的图像
        patched_image_name = f"patched_{os.path.basename(item['image'])}"
        patched_image_path = os.path.join(patched_image_dir, patched_image_name)
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(patched_image_path), exist_ok=True)
        
        patched_image.save(patched_image_path)
        
        # 5. 更新数据项
        relative_path = os.path.join("patched", patched_image_name)
        patched_item['image'] = relative_path
        patched_item['has_patch'] = True
        
        # 6. 添加标记表示不应使用CoT
        patched_item['should_use_cot'] = False
        
        patched_data.append(patched_item)
    
    # 保存新的JSON文件
    with open(output_json_path, 'w') as f:
        json.dump(patched_data, f)
    
    print(f"Created {len(patched_data)} patched examples")
    print(f"Saved to {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a dataset with patched images for CoT control")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the original JSON dataset")
    parser.add_argument("--image_root", type=str, required=True, help="Root directory of the original images")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the patched dataset JSON")
    parser.add_argument("--output_image_dir", type=str, required=True, help="Directory to save patched images")
    parser.add_argument("--patch_size", type=int, default=20, help="Size of the patch in pixels")
    parser.add_argument("--patch_position", type=str, default="bottom-right", 
                        choices=["bottom-right", "top-left", "random"], 
                        help="Position of the patch")
    parser.add_argument("--patch_color", type=str, default="red", 
                        choices=["red", "green", "blue", "black"], 
                        help="Color of the patch")
    
    args = parser.parse_args()
    
    # 设置patch颜色
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "black": (0, 0, 0)
    }
    
    create_patched_dataset(
        args.input_json, 
        args.image_root, 
        args.output_json, 
        args.output_image_dir
    ) 