"""
损失函数模块
包含 IIC Loss（Invariant Information Clustering）等
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def IID_loss(x_out: torch.Tensor, x_tf_out: torch.Tensor, lamb: float = 1.0, EPS: float = sys.float_info.epsilon) -> torch.Tensor:
    """
    Invariant Information Clustering (IIC) Loss
    
    用于对齐两个概率分布，使它们预测一致
    
    Args:
        x_out: 第一个概率分布 [B, C]，已经过 softmax
        x_tf_out: 第二个概率分布 [B, C]，已经过 softmax
        lamb: 正则化系数
        EPS: 数值稳定参数
        
    Returns:
        IIC loss 值
    """
    if x_out.dim() == 1:
        x_out = x_out.unsqueeze(0)
        x_tf_out = x_tf_out.unsqueeze(0)
    
    _, k = x_out.size()
    bn_, k_ = x_out.size()
    assert (x_tf_out.size(0) == bn_ and x_tf_out.size(1) == k_)

    # 计算联合概率分布
    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # [B, C, C]
    p_i_j = p_i_j.sum(dim=0)  # [C, C]
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # 对称化
    p_i_j = p_i_j / p_i_j.sum()  # 归一化

    assert (p_i_j.size() == (k, k))

    # 边缘概率
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    # 避免 NaN
    p_i_j[(p_i_j < EPS).data] = EPS

    # 互信息损失
    loss = -p_i_j * (torch.log(p_i_j + 1e-5) 
                     - lamb * torch.log(p_j + 1e-5) 
                     - lamb * torch.log(p_i + 1e-5))
    
    loss = loss.sum()
    return loss


def entropy_loss(probs: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    熵最小化损失
    
    Args:
        probs: 预测概率 [B, C]
        epsilon: 数值稳定参数
        
    Returns:
        平均熵
    """
    entropy = -torch.sum(probs * torch.log(probs + epsilon), dim=1)
    return torch.mean(entropy)


def diversity_loss(probs: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    多样性损失（防止类别坍塌）
    
    返回 batch 平均概率的熵值（正值）
    用法：total_loss = ... - weight * diversity_loss（减去熵以鼓励均匀分布）
    
    Args:
        probs: 预测概率 [B, C]
        epsilon: 数值稳定参数
        
    Returns:
        正的熵值（越大越多样）
    """
    mean_probs = probs.mean(dim=0)
    # 返回正的熵值，与原版 ProDe 一致
    gentropy = torch.sum(-mean_probs * torch.log(mean_probs + epsilon))
    return gentropy
