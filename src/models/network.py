"""
神经网络模型定义
包含: ResNet Backbone, Bottleneck, Classifier
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple


class ResNetBackbone(nn.Module):
    """
    ResNet 特征提取器（去掉最后的fc层）
    """
    
    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True
    ):
        """
        Args:
            name: 模型名称 (resnet50, resnet101)
            pretrained: 是否使用ImageNet预训练权重
        """
        super().__init__()
        
        # 加载预训练模型
        if name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
        elif name == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet101(weights=weights)
        else:
            raise ValueError(f"不支持的backbone: {name}")
        
        # 去掉最后的全连接层
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        
        # 输出特征维度
        self.out_features = resnet.fc.in_features  # 2048 for resnet50
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            
        Returns:
            特征向量 [B, out_features]
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


class Bottleneck(nn.Module):
    """
    特征压缩层（Bottleneck）
    Linear -> BatchNorm -> ReLU -> Dropout
    """
    
    def __init__(
        self,
        in_features: int = 2048,
        out_features: int = 256,
        dropout: float = 0.5
    ):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: Dropout概率
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, affine=True)
        self.dropout = nn.Dropout(p=dropout)
        
        # 初始化
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, in_features]
            
        Returns:
            压缩特征 [B, out_features]
        """
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class Classifier(nn.Module):
    """
    分类器头
    """
    
    def __init__(
        self,
        in_features: int = 256,
        num_classes: int = 65
    ):
        """
        Args:
            in_features: 输入特征维度
            num_classes: 类别数
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        
        self.fc = nn.Linear(in_features, num_classes)
        
        # 初始化
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, in_features]
            
        Returns:
            分类logits [B, num_classes]
        """
        return self.fc(x)
    
    def get_prototypes(self) -> torch.Tensor:
        """
        获取分类器权重作为类原型
        
        Returns:
            prototypes: [num_classes, in_features]
        """
        return self.fc.weight.data


class FeatureAdapter(nn.Module):
    """
    特征适应器 (Adapter)
    
    输入原始特征，输出修正向量 (velocity)
    f_new = f_t + adapter(f_t)
    
    用于将目标域特征推向源域原型
    """
    
    def __init__(
        self,
        in_features: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            in_features: 输入/输出特征维度
            hidden_dim: 隐藏层维度
            num_layers: MLP层数
            dropout: dropout概率
        """
        super().__init__()
        
        self.in_features = in_features
        
        layers = []
        current_dim = in_features
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout)
            ])
            current_dim = hidden_dim
        
        # 最后一层输出修正向量
        layers.append(nn.Linear(current_dim, in_features))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化：让初始输出接近0（残差学习）
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重，让初始输出接近0"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算特征修正向量 (velocity)
        
        Args:
            x: 输入特征 [B, in_features]
            
        Returns:
            velocity: 修正向量 [B, in_features]
        """
        return self.mlp(x)
    
    def adapt(self, x: torch.Tensor) -> torch.Tensor:
        """
        适应特征: f_new = f_t + velocity
        
        Args:
            x: 原始特征 [B, in_features]
            
        Returns:
            f_new: 适应后的特征 [B, in_features]
        """
        velocity = self.forward(x)
        return x + velocity


class FullModel(nn.Module):
    """
    完整模型: Backbone + Bottleneck + Classifier
    
    支持返回中间特征，便于SFDA方法使用
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        bottleneck_dim: int = 256,
        num_classes: int = 65,
        dropout: float = 0.5
    ):
        """
        Args:
            backbone: backbone名称
            pretrained: 是否使用预训练权重
            bottleneck_dim: bottleneck特征维度
            num_classes: 类别数
            dropout: dropout概率
        """
        super().__init__()
        
        # 构建各模块
        self.backbone = ResNetBackbone(backbone, pretrained)
        self.bottleneck = Bottleneck(
            in_features=self.backbone.out_features,
            out_features=bottleneck_dim,
            dropout=dropout
        )
        self.classifier = Classifier(
            in_features=bottleneck_dim,
            num_classes=num_classes
        )
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            return_features: 是否返回中间特征
            
        Returns:
            logits: 分类logits [B, num_classes]
            features: (可选) bottleneck特征 [B, bottleneck_dim]
        """
        # 提取特征
        feat_backbone = self.backbone(x)
        feat_bottleneck = self.bottleneck(feat_backbone)
        logits = self.classifier(feat_bottleneck)
        
        if return_features:
            return logits, feat_bottleneck
        else:
            return logits, None
    
    def get_params(self, base_lr: float) -> list:
        """
        获取分组参数（用于差异化学习率）
        
        Args:
            base_lr: 基础学习率
            
        Returns:
            参数组列表
        """
        # backbone使用较小学习率
        # bottleneck和classifier使用较大学习率
        params = [
            {"params": self.backbone.parameters(), "lr": base_lr * 0.1},
            {"params": self.bottleneck.parameters(), "lr": base_lr},
            {"params": self.classifier.parameters(), "lr": base_lr}
        ]
        return params


def create_model(config: dict) -> FullModel:
    """
    根据配置创建模型
    
    Args:
        config: 配置字典
        
    Returns:
        FullModel实例
    """
    model_cfg = config['model']
    data_cfg = config['data']
    
    model = FullModel(
        backbone=model_cfg['backbone'],
        pretrained=model_cfg['pretrained'],
        bottleneck_dim=model_cfg['bottleneck_dim'],
        num_classes=data_cfg['num_classes'],
        dropout=model_cfg['dropout']
    )
    
    return model


def create_split_models(config: dict) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    创建拆分后的模型组件 (Backbone, Bottleneck, Classifier)
    
    Args:
        config: 配置字典
        
    Returns:
        (netG, netF, netC)
    """
    model_cfg = config['model']
    data_cfg = config['data']
    
    # Backbone (netG)
    netG = ResNetBackbone(
        name=model_cfg['backbone'],
        pretrained=model_cfg['pretrained']
    )
    
    # Bottleneck (netF)
    netF = Bottleneck(
        in_features=netG.out_features,
        out_features=model_cfg['bottleneck_dim'],
        dropout=model_cfg['dropout']
    )
    
    # Classifier (netC)
    netC = Classifier(
        in_features=model_cfg['bottleneck_dim'],
        num_classes=data_cfg['num_classes']
    )
    
    return netG, netF, netC
