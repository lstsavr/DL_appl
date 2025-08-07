import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset

class FlickrImageTextDataset(Dataset):
    def __init__(self, ann_file, img_dir, tokenizer=None, transform=None, img_name_list=None, max_length=40):
        """
        ann_file: 标注文件路径（json，包含images字段，每个元素有filename和sentences）
        img_dir: 图像文件夹路径
        tokenizer: 文本分词/编码器
        transform: 图像预处理
        img_name_list: 仅加载该列表中的图片（可为None）
        max_length: 文本最大长度
        """
        self.img_dir = img_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 读取标注文件，适配Flickr8k格式
        with open(ann_file, 'r') as f:
            data = json.load(f)
        anns = data['images']
        # 支持图片筛选
        if img_name_list is not None:
            if isinstance(img_name_list, str):
                with open(img_name_list, 'r') as f:
                    img_name_list = set(line.strip() for line in f)
            else:
                img_name_list = set(img_name_list)
            anns = [ann for ann in anns if ann['filename'] in img_name_list]
        # 过滤无caption或无图片的样本
        self.anns = [ann for ann in anns if ann.get('sentences') and len(ann['sentences']) > 0]

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        img_name = ann['filename']
        img_path = os.path.join(self.img_dir, img_name)
        # 图像异常处理
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # 随机采样一条caption（取raw字段）
        captions = [s['raw'] for s in ann['sentences'] if s.get('raw')]
        if not captions:
            raise ValueError(f"No valid caption for image: {img_name}")
        caption = random.choice(captions)
        if self.tokenizer:
            text = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            text = {k: v.squeeze(0) for k, v in text.items()}
        else:
            text = caption
        return image, text, img_name, caption