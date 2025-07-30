#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def convert_json_to_flickr8k(json_path, dst_root):
    # 确保目标目录存在
    dst_dataset_dir = os.path.join(dst_root, "Flicker8k_Dataset")
    dst_text_dir = os.path.join(dst_root, "Flickr8k_text")
    os.makedirs(dst_dataset_dir, exist_ok=True)
    os.makedirs(dst_text_dir, exist_ok=True)
    
    # 准备输出文件路径
    token_file = os.path.join(dst_text_dir, "Flickr8k.token.txt")
    train_file = os.path.join(dst_text_dir, "Flickr_8k.trainImages.txt")
    val_file = os.path.join(dst_text_dir, "Flickr_8k.valImages.txt")
    test_file = os.path.join(dst_text_dir, "Flickr_8k.testImages.txt")
    
    # 获取JSON文件所在目录，用于图片复制
    json_dir = os.path.dirname(json_path)
    images_dir = os.path.join(json_dir, "images")
    
    # 加载JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计计数器
    total_images = 0
    total_captions = 0
    
    # 打开输出文件
    with open(token_file, 'w', encoding='utf-8') as f_token, \
         open(train_file, 'w', encoding='utf-8') as f_train, \
         open(val_file, 'w', encoding='utf-8') as f_val, \
         open(test_file, 'w', encoding='utf-8') as f_test:
        
        # 遍历所有图片
        for image_data in tqdm(data["images"], desc="Processing images"):
            file_name = image_data["filename"]  # 使用filename
            split = image_data["split"]
            # 从sentences中提取raw文本
            captions = [sent["raw"] for sent in image_data["sentences"]]
            
            # 复制图片文件（如果不存在）
            dst_img_path = os.path.join(dst_dataset_dir, file_name)
            if not os.path.exists(dst_img_path):
                src_img_path = os.path.join(images_dir, file_name)
                if os.path.exists(src_img_path):
                    shutil.copy2(src_img_path, dst_img_path)
                else:
                    print(f"Warning: Source image not found: {src_img_path}")
            
            # 写入captions到token文件（按Flickr8k格式：filename#index）
            for i, caption in enumerate(captions):
                f_token.write(f"{file_name}#{i}\t{caption}\n")
                total_captions += 1
            
            # 按split写入文件名到对应文件
            if split == "train":
                f_train.write(f"{file_name}\n")
            elif split in ["val", "dev"]:
                f_val.write(f"{file_name}\n")
            elif split == "test":
                f_test.write(f"{file_name}\n")
            
            total_images += 1
    
    print(f"Conversion completed. Statistics: images={total_images}, captions={total_captions}")


def main():
    parser = argparse.ArgumentParser(description='Convert dataset_flickr8k.json to Flickr8k text files')
    parser.add_argument('--json_path', type=str, default='dataset_flickr8k.json',
                        help='Path to the JSON file')
    parser.add_argument('--dst_root', type=str, default='data/raw',
                        help='Destination root directory')
    args = parser.parse_args()
    
    convert_json_to_flickr8k(args.json_path, args.dst_root)


if __name__ == "__main__":
    main()
