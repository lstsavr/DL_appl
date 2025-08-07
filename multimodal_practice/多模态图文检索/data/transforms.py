from torchvision import transforms

def get_image_transform(image_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),  # 更激进的裁剪范围
            transforms.RandomHorizontalFlip(p=0.7),  # 增加水平翻转概率
            transforms.RandomRotation(15),  # 增加随机旋转角度
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # 更强的色彩抖动
            transforms.RandomPerspective(distortion_scale=0.3, p=0.3),  # 添加透视变换
            transforms.RandomAutocontrast(p=0.3),  # 增加自动对比度概率
            transforms.RandomGrayscale(p=0.15),  # 增加灰度概率
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.33)),  # 增加随机擦除强度和概率
        ])
    else:
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),  # 更大的尺寸
            transforms.CenterCrop(image_size),  # 中心裁剪
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# 文本预处理可根据tokenizer自定义，这里预留接口
def get_text_preprocess():
    return lambda x: x  # 默认不处理，直接返回原文本