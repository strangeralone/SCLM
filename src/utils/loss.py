"""
损失函数模块
"""
import torch
import torch.nn as nn
import numpy as np


class CrossEntropyLabelSmooth(nn.Module):
    """
    带标签平滑的交叉熵损失
    
    标签平滑是一种正则化技术，将硬标签转换为软标签：
    - 原始硬标签: [0, 0, 1, 0, 0]
    - 平滑后: [ε/K, ε/K, 1-ε+ε/K, ε/K, ε/K]
    
    其中 ε 是平滑参数，K 是类别数
    """
    
    def __init__(self, num_classes: int, epsilon: float = 0.1, reduction: str = 'mean'):
        """
        Args:
            num_classes: 类别数
            epsilon: 平滑参数，默认 0.1
            reduction: 'mean' 或 'none'
        """
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 模型输出 logits，shape (N, C)
            targets: 标签，shape (N,)
        
        Returns:
            损失值
        """
        log_probs = self.logsoftmax(inputs)
        
        # 创建平滑标签
        targets_onehot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        targets_smooth = (1 - self.epsilon) * targets_onehot + \
                         self.epsilon / self.num_classes
        
        # 计算损失
        loss = (-targets_smooth * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss


def Entropy(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    计算预测概率的熵
    
    Args:
        input_tensor: softmax 概率分布，shape (N, C)
    
    Returns:
        每个样本的熵，shape (N,)
    """
    epsilon = 1e-5
    entropy = -input_tensor * torch.log(input_tensor + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy
