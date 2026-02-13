"""
Office-Home 数据集加载模块
"""
import os
from pathlib import Path
from typing import Optional, Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 数据增强: 训练时使用
def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """训练数据增强"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# 数据增强: 测试时使用
def get_test_transform(image_size: int = 224) -> transforms.Compose:
    """测试数据变换（无增强）"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class OfficeHomeDataset(Dataset):
    """
    Office-Home 数据集
    
    支持两种加载方式：
    1. 从标签文件加载 (如 Art_65.txt)
    2. 从文件夹结构加载
    
    数据划分：
    - 默认按 70% 训练 / 30% 测试 划分
    - 使用固定随机种子确保划分一致性
    """
    
    DOMAINS = ['Art', 'Clipart', 'Product', 'RealWorld']
    NUM_CLASSES = 65
    
    def __init__(
        self,
        root: str,
        domain: str,
        transform: Optional[transforms.Compose] = None,
        train: bool = True,
        split_ratio: float = 0.9,
        split_seed: int = 42
    ):
        """
        Args:
            root: 数据集根目录
            domain: 域名称 (Art, Clipart, Product, RealWorld)
            transform: 数据变换
            train: 是否为训练模式（True=训练集, False=测试集）
            split_ratio: 训练集占比（默认0.7，即70%训练/30%测试）
            split_seed: 划分时使用的随机种子，确保一致性
        """
        self.root = Path(root)
        self.domain = domain
        self.transform = transform
        self.train = train
        self.split_ratio = split_ratio
        self.split_seed = split_seed
        
        # 验证域名
        if domain not in self.DOMAINS:
            raise ValueError(f"域 '{domain}' 不存在，可选: {self.DOMAINS}")
        
        # 加载全部数据，然后进行划分
        all_samples = self._load_samples()
        self.samples = self._split_samples(all_samples)
        
    def _load_samples(self) -> List[Tuple[str, int]]:
        """加载样本列表"""
        samples = []
        
        # 尝试从标签文件加载
        label_file = self.root / f"{self.domain}_65.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        # 路径格式: data/officehome/Art/xxx.jpg 0
                        img_path = parts[0]
                        label = int(parts[1])
                        # 转换为绝对路径（相对于项目根目录）
                        full_path = Path(img_path)
                        if not full_path.is_absolute():
                            # 标签文件中的路径是相对于项目根目录的
                            # 我们需要找到正确的基准
                            full_path = self.root.parent.parent / img_path
                        samples.append((str(full_path), label))
        else:
            # 从文件夹结构加载
            domain_dir = self.root / self.domain
            if not domain_dir.exists():
                raise FileNotFoundError(f"域目录不存在: {domain_dir}")
            
            # 获取类别列表
            classes = sorted([d.name for d in domain_dir.iterdir() if d.is_dir()])
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            for cls_name, cls_idx in class_to_idx.items():
                cls_dir = domain_dir / cls_name
                for img_file in cls_dir.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        samples.append((str(img_file), cls_idx))
        
        return samples
    
    def _split_samples(self, all_samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """
        根据 train 参数划分训练集或测试集
        
        使用固定随机种子确保每次划分结果一致
        """
        import random
        
        # 使用固定种子打乱样本，确保每次运行划分一致
        rng = random.Random(self.split_seed)
        shuffled = all_samples.copy()
        rng.shuffle(shuffled)
        
        # 如果 split_ratio >= 1.0，直接返回全部数据（不做划分）
        if self.split_ratio >= 1.0:
            return shuffled
        
        # 计算划分点
        split_idx = int(len(shuffled) * self.split_ratio)
        
        if self.train:
            # 训练集: 前 split_ratio (70%)
            return shuffled[:split_idx]
        else:
            # 测试集: 后 1-split_ratio (30%)
            return shuffled[split_idx:]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
            image: 图像tensor
            label: 标签
            index: 样本索引（用于伪标签更新）
        """
        img_path, label = self.samples[index]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"无法加载图像: {img_path}, 错误: {e}")
            # 返回一个随机图像作为占位
            image = Image.new('RGB', (224, 224), color='gray')
        
        # 数据变换
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label, index


def get_loader(
    root: str,
    domain: str,
    batch_size: int = 32,
    train: bool = True,
    num_workers: int = 4,
    image_size: int = 224,
    shuffle: Optional[bool] = None,
    drop_last: bool = False,
    split_ratio: float = 0.9
) -> DataLoader:
    """
    获取数据加载器
    
    Args:
        root: 数据集根目录
        domain: 域名称
        batch_size: 批大小
        train: 是否为训练模式
        num_workers: 数据加载线程数
        image_size: 图像大小
        shuffle: 是否打乱（默认训练时打乱）
        drop_last: 是否丢弃最后不完整的batch
        
    Returns:
        DataLoader
    """
    # 默认：训练时打乱，测试时不打乱
    if shuffle is None:
        shuffle = train
    
    # 选择变换
    transform = get_train_transform(image_size) if train else get_test_transform(image_size)
    
    # 创建数据集
    dataset = OfficeHomeDataset(
        root=root,
        domain=domain,
        transform=transform,
        train=train,
        split_ratio=split_ratio
    )
    
    # 创建加载器
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last
    )
    
    return loader


def get_class_names(root: str) -> List[str]:
    """
    获取类别名称列表
    
    Args:
        root: 数据集根目录
        
    Returns:
        类别名称列表
    """
    classname_file = Path(root) / "classname.txt"
    if classname_file.exists():
        with open(classname_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        # 从文件夹获取
        domain_dir = Path(root) / "Art"  # 任意一个域
        return sorted([d.name for d in domain_dir.iterdir() if d.is_dir()])
