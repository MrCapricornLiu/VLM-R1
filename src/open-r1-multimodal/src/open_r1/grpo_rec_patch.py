# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pdb
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from open_r1.vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
monkey_patch_qwen2_5vl_flash_attn()


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'adaptive_format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "adaptive_format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'adaptive_format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    no_cot_template: Optional[str] = field(
        default="Answer this question about the image: {Question} Provide only the answer in <answer> </answer> tags.",
        metadata={"help": "Template for inputs that should not use CoT"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def patch_aware_collate_fn(examples):
    """处理包含should_use_cot标志的batch数据"""
    # 确保返回一个字典列表，而不是字段字典 
    # 这样_generate_and_score_completions函数中的[x["prompt"] for x in inputs]才能正常工作
    
    # 将每个should_use_cot标志添加到对应的示例中
    processed_examples = []
    for example in examples:
        # 创建一个新的示例字典，包含所有原始字段
        processed_example = {}
        for key, value in example.items():
            processed_example[key] = value
        
        # 确保should_use_cot存在
        if "should_use_cot" not in processed_example:
            processed_example["should_use_cot"] = True
        
        processed_examples.append(processed_example)
    
    return processed_examples

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str, no_cot_template: str = None):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.question_template = question_template
        self.no_cot_template = no_cot_template or script_args.no_cot_template

        if data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999

                for data in datasets:
                    json_path = data.get("json_path")
                    sampling_strategy = data.get("sampling_strategy", "all")
                    sampling_number = None

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]
                    print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            raise ValueError(f"Unsupported file type: {data_path}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        # Format into conversation
        def make_conversation(example):
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        
        example = self.list_data_dict[i]
        image_root = self.script_args.image_root
        if 'image' in example:
            image_path = os.path.join(image_root, example['image'])
            # print(f"image_root: {image_root}")
            # print(f"image_path: {image_path}")
            # pdb.set_trace()
            # In case the image is not found
            while not os.path.exists(image_path):
                print(f"Warning: Image {image_path} not found, randomly selecting another image")
                new_index = random.randint(0, len(self.list_data_dict)-1)
                example = self.list_data_dict[new_index]
                image_path = os.path.join(image_root, example['image'])
            image = Image.open(image_path).convert("RGB")
        else:
            image = None
        
        # 判断是否应该使用CoT模板
        should_use_cot = example.get('should_use_cot', True)
        template = self.question_template if should_use_cot else self.no_cot_template
        
        def make_conversation_image(example, template):
            return {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": template.format(Question=example["problem"])},
                        ],
                    },
                ],
            }

        return {
            'image': image,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example, template)['prompt'] if 'image' in example else make_conversation(example)['prompt'],
            'should_use_cot': should_use_cot
        }


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

class PatchAwareGRPOTrainer(VLMGRPOTrainer):
    def compute_reward(self, completions, **kwargs):
        # 处理 should_use_cot 参数
        should_use_cot = None
        if "should_use_cot" in kwargs:
            should_use_cot = kwargs.pop("should_use_cot")  # 从kwargs中移除，避免父类方法报错
        elif hasattr(self, "_buffered_inputs") and self._buffered_inputs and "should_use_cot" in self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]:
            # 从缓存的输入中获取should_use_cot
            should_use_cot = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]["should_use_cot"]
        
        # 调用父类方法获取基本奖励
        rewards = super().compute_reward(completions, **kwargs)
        
        # 对于adaptive_format_reward，传递should_use_cot参数
        for i, reward_func in enumerate(self.reward_funcs):
            if reward_func.__name__ == "adaptive_format_reward" and should_use_cot is not None:
                # 直接调用adaptive_format_reward
                format_rewards = adaptive_format_reward(completions, should_use_cot=should_use_cot, **kwargs)
                # 更新对应位置的奖励
                for j, reward in enumerate(format_rewards):
                    rewards[j] = reward
                break
        
        return rewards
    
    def get_train_dataloader(self):
        """覆盖原方法，使用自定义的collate_fn"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=patch_aware_collate_fn,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """覆盖计算损失的方法，确保正确处理should_use_cot标志"""
        # 提取should_use_cot标志，以便在调用_generate_and_score_completions时使用
        should_use_cot = [x.get("should_use_cot", True) for x in inputs]
        
        # 使用父类的compute_loss方法
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    
    def _generate_and_score_completions(self, inputs, model):
        """覆盖_generate_and_score_completions方法，确保正确处理should_use_cot标志"""
        # 提取should_use_cot标志，以便在计算奖励时使用
        should_use_cot = [x.get("should_use_cot", True) for x in inputs]
        
        # 创建一个新的inputs字典，不包含should_use_cot
        clean_inputs = []
        for x in inputs:
            clean_input = {k: v for k, v in x.items() if k != "should_use_cot"}
            clean_inputs.append(clean_input)
        
        # 调用父类的_generate_and_score_completions方法
        results = super()._generate_and_score_completions(clean_inputs, model)
        
        # 将should_use_cot添加到结果中，以便在adaptive_format_reward中使用
        results["should_use_cot"] = should_use_cot
        
        return results

def adaptive_format_reward(completions, should_use_cot: List[bool] = None, **kwargs):
    """根据图像是否有patch来评估输出格式，增加奖励差异"""
    import re
    
    if should_use_cot is None:
        should_use_cot = [True] * len(completions[0])
    
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, use_cot in zip(completion_contents, should_use_cot):
        # 检查是否包含思维标签
        has_think_tag = re.search(r"<think>.*?</think>", content, re.DOTALL) is not None
        # 检查是否包含答案标签和边界框
        has_answer_tag = re.search(r"<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>", content, re.DOTALL) is not None
        
        # 计算奖励，提高惩罚强度
        if use_cot:
            # 应该使用CoT：需要思维和答案标签
            reward = 1.0 if (has_think_tag and has_answer_tag) else -0.5
        else:
            # 不应使用CoT：只需要答案标签，不要思维标签
            reward = 1.0 if (has_answer_tag and not has_think_tag) else -0.5
        
        rewards.append(reward)
        
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Format reward: {reward} -------------\n")
                f.write(f"Should use CoT: {use_cot}\n")
                f.write(f"Has think tag: {has_think_tag}\n")
                f.write(f"Has answer tag: {has_answer_tag}\n")
                f.write(f"Content: {content}\n")
    
    return rewards

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)

    # Load the reward functions
    reward_funcs_registry = {
        "accuracy": vlm_module_cls.iou_reward,
        "format": vlm_module_cls.format_reward_rec,
        "adaptive_format": adaptive_format_reward,
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    dataset = LazySupervisedDataset(
        script_args.dataset_name, 
        script_args, 
        question_template=vlm_module_cls.get_question_template(task_type="rec"),
        no_cot_template=script_args.no_cot_template
    )

    trainer_cls = PatchAwareGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args) 