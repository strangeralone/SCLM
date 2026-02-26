"""
可视化工具模块
包含 T-SNE、混淆矩阵、置信度直方图等训练调试工具
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.colors as mcolors
import logging
from typing import Optional


def _ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def plot_consensus_tsne(
    features: np.ndarray,
    src_pred: np.ndarray,
    clip_pred: np.ndarray,
    clip_noisy_pred: np.ndarray,
    save_path: str,
    logger: Optional[logging.Logger] = None
):
    """
    共识图：检查 CLIP 和源模型的分歧
    绿色 = 三方预测一致，红色 = 存在分歧
    
    Args:
        features: 特征矩阵 [N, D]
        src_pred: 源模型预测 [N]
        clip_pred: CLIP 预测 [N]
        clip_noisy_pred: CLIP 扰动预测 [N]
        save_path: 保存路径（完整文件名）
    """
    _ensure_dir(os.path.dirname(save_path))
    
    # T-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    features_2d = tsne.fit_transform(features)
    
    # 计算共识
    consensus = (src_pred == clip_pred) & (clip_pred == clip_noisy_pred)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 先画红色（分歧），再画绿色（一致），让绿色在上层更清晰
    disagree_mask = ~consensus
    ax.scatter(features_2d[disagree_mask, 0], features_2d[disagree_mask, 1],
               c='red', alpha=0.4, s=8, label=f'Disagree ({disagree_mask.sum()})')
    ax.scatter(features_2d[consensus, 0], features_2d[consensus, 1],
               c='green', alpha=0.4, s=8, label=f'Consensus ({consensus.sum()})')
    
    consensus_ratio = consensus.sum() / len(consensus) * 100
    ax.set_title(f'Consensus T-SNE (Agreement: {consensus_ratio:.1f}%)', fontsize=14)
    ax.legend(fontsize=11, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if logger:
        logger.info(f"共识 T-SNE 已保存: {save_path} (一致率: {consensus_ratio:.1f}%)")


def plot_collapse_tsne(
    features_epoch1: np.ndarray,
    features_current: np.ndarray,
    current_epoch: int,
    save_path: str,
    logger: Optional[logging.Logger] = None
):
    """
    特征坍塌图：对比不同 epoch 的特征分布
    蓝色 = Epoch 1，橙色 = 当前 Epoch
    
    如果当前 epoch 的特征全部挤在一起（坍塌），说明 Loss 有问题
    
    Args:
        features_epoch1: Epoch 1 的特征 [N, D]
        features_current: 当前 Epoch 的特征 [N, D]
        current_epoch: 当前 epoch 编号
        save_path: 保存路径
    """
    _ensure_dir(os.path.dirname(save_path))
    
    # 合并后一起做 T-SNE（确保同一空间）
    n1 = len(features_epoch1)
    combined = np.concatenate([features_epoch1, features_current], axis=0)
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined) - 1))
    combined_2d = tsne.fit_transform(combined)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.scatter(combined_2d[:n1, 0], combined_2d[:n1, 1],
               c='dodgerblue', alpha=0.3, s=8, label='Epoch 1')
    ax.scatter(combined_2d[n1:, 0], combined_2d[n1:, 1],
               c='darkorange', alpha=0.3, s=8, label=f'Epoch {current_epoch}')
    
    ax.set_title(f'Feature Collapse Check: Epoch 1 vs Epoch {current_epoch}', fontsize=14)
    ax.legend(fontsize=11, markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if logger:
        logger.info(f"特征坍塌 T-SNE 已保存: {save_path}")


def plot_poisoning_tsne(
    features: np.ndarray,
    gt_labels: np.ndarray,
    pseudo_labels: np.ndarray,
    save_path: str,
    max_classes_display: int = 10,
    logger: Optional[logging.Logger] = None
):
    """
    伪标签毒化图：GT颜色 + 伪标签形状
    
    如果某个颜色（GT类）对应多种形状（伪标签类），说明该类被"毒化"
    
    Args:
        features: 特征矩阵 [N, D]
        gt_labels: 真实标签 [N]
        pseudo_labels: 伪标签 [N]
        save_path: 保存路径
        max_classes_display: 最多显示的类别数（自动挑选错误率最高的类）
    """
    _ensure_dir(os.path.dirname(save_path))
    
    # 找出错误率最高的类
    unique_classes = np.unique(gt_labels)
    error_rates = []
    for cls in unique_classes:
        mask = gt_labels == cls
        if mask.sum() > 0:
            err = (pseudo_labels[mask] != cls).mean()
            error_rates.append((cls, err))
    
    error_rates.sort(key=lambda x: -x[1])
    top_classes = [c for c, _ in error_rates[:max_classes_display]]
    
    # 只保留这些类的样本
    mask = np.isin(gt_labels, top_classes)
    features_sub = features[mask]
    gt_sub = gt_labels[mask]
    pseudo_sub = pseudo_labels[mask]
    
    if len(features_sub) < 5:
        if logger:
            logger.warning(f"伪标签毒化图样本过少 ({len(features_sub)})，跳过")
        return
    
    # T-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_sub) - 1))
    features_2d = tsne.fit_transform(features_sub)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 颜色 = GT，形状 = 匹配/不匹配
    colors = plt.cm.tab20(np.linspace(0, 1, len(top_classes)))
    class_to_color = {cls: colors[i] for i, cls in enumerate(top_classes)}
    
    # 正确的（圆圈），错误的（叉叉）
    correct = gt_sub == pseudo_sub
    
    for cls in top_classes:
        cls_mask = gt_sub == cls
        correct_mask = cls_mask & correct
        wrong_mask = cls_mask & ~correct
        
        c = class_to_color[cls]
        if correct_mask.sum() > 0:
            ax.scatter(features_2d[correct_mask, 0], features_2d[correct_mask, 1],
                      c=[c], marker='o', alpha=0.5, s=15, label=f'Class {cls} correct')
        if wrong_mask.sum() > 0:
            ax.scatter(features_2d[wrong_mask, 0], features_2d[wrong_mask, 1],
                      c=[c], marker='x', alpha=0.7, s=25, label=f'Class {cls} wrong')
    
    total_err = (~correct).sum() / len(correct) * 100
    ax.set_title(f'Pseudo-label Poisoning (Top-{len(top_classes)} Error Classes, Error: {total_err:.1f}%)', fontsize=13)
    ax.legend(fontsize=8, markerscale=2, ncol=2, loc='best')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if logger:
        logger.info(f"伪标签毒化 T-SNE 已保存: {save_path}")


def plot_confusion_matrix(
    gt_labels: np.ndarray,
    pseudo_labels: np.ndarray,
    save_path: str,
    num_classes: int = None,
    logger: Optional[logging.Logger] = None
):
    """
    归一化混淆矩阵（GT vs 伪标签）
    对角线越亮说明伪标签越准确
    
    Args:
        gt_labels: 真实标签 [N]
        pseudo_labels: 伪标签 [N]
        save_path: 保存路径
        num_classes: 类别总数
    """
    _ensure_dir(os.path.dirname(save_path))
    
    if num_classes is None:
        num_classes = max(gt_labels.max(), pseudo_labels.max()) + 1
    
    cm = confusion_matrix(gt_labels, pseudo_labels, labels=list(range(num_classes)))
    
    # 按行归一化（每个 GT 类归一化到1）
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除0
    cm_norm = cm.astype(float) / row_sums
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 总体准确率
    acc = (gt_labels == pseudo_labels).mean() * 100
    ax.set_title(f'Normalized Confusion Matrix (Pseudo Acc: {acc:.1f}%)', fontsize=14)
    ax.set_xlabel('Pseudo Label', fontsize=12)
    ax.set_ylabel('Ground Truth', fontsize=12)
    
    # 如果类别数不多，显示刻度
    if num_classes <= 30:
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.tick_params(axis='both', labelsize=6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if logger:
        logger.info(f"混淆矩阵已保存: {save_path} (伪标签准确率: {acc:.1f}%)")


def plot_confidence_histogram(
    max_probs: np.ndarray,
    is_correct: np.ndarray,
    save_path: str,
    logger: Optional[logging.Logger] = None
):
    """
    置信度分布直方图（正确预测 vs 错误预测分开画）
    
    理想情况：正确预测集中在高置信区间，错误预测集中在低置信区间
    如果大量错误预测也有高置信度 → 模型"自信但错误"，很危险
    
    Args:
        max_probs: 预测的最大概率值 [N]
        is_correct: 是否正确预测的布尔数组 [N]
        save_path: 保存路径
    """
    _ensure_dir(os.path.dirname(save_path))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    bins = np.linspace(0, 1, 51)
    
    correct_probs = max_probs[is_correct]
    wrong_probs = max_probs[~is_correct]
    
    ax.hist(correct_probs, bins=bins, alpha=0.6, color='green', 
            label=f'Correct ({len(correct_probs)})', density=True)
    ax.hist(wrong_probs, bins=bins, alpha=0.6, color='red',
            label=f'Wrong ({len(wrong_probs)})', density=True)
    
    # 画置信度阈值线
    ax.axvline(x=0.7, color='black', linestyle='--', linewidth=1.5, label='Threshold (0.7)')
    
    overall_acc = is_correct.mean() * 100
    mean_conf = max_probs.mean()
    ax.set_title(f'Confidence Distribution (Acc: {overall_acc:.1f}%, Mean Conf: {mean_conf:.3f})', fontsize=13)
    ax.set_xlabel('Max Softmax Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if logger:
        logger.info(f"置信度直方图已保存: {save_path}")


def generate_all_visualizations(
    features: np.ndarray,
    gt_labels: np.ndarray,
    pseudo_labels: np.ndarray,
    max_probs: np.ndarray,
    src_pred: np.ndarray,
    clip_pred: np.ndarray,
    clip_noisy_pred: np.ndarray,
    save_dir: str,
    num_classes: int,
    current_epoch: int,
    features_epoch1: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None
):
    """
    一次性生成所有可视化图表
    
    Args:
        features: bottleneck 特征 [N, D]
        gt_labels: 真实标签 [N]
        pseudo_labels: 最终伪标签 [N]
        max_probs: 最大 softmax 概率 [N]
        src_pred: 源模型预测 [N]
        clip_pred: CLIP 预测 [N]
        clip_noisy_pred: CLIP 扰动预测 [N]
        save_dir: 保存目录（如 logs/xxx/epoch1/）
        num_classes: 类别总数
        current_epoch: 当前 epoch
        features_epoch1: Epoch 1 的特征（用于坍塌图）
        logger: 日志器
    """
    _ensure_dir(save_dir)
    
    if logger:
        logger.info(f"开始生成 Epoch {current_epoch} 的可视化图表...")
    
    # 1. 共识 T-SNE
    plot_consensus_tsne(
        features, src_pred, clip_pred, clip_noisy_pred,
        os.path.join(save_dir, 't_sne_consensus.jpg'), logger
    )
    
    # 2. 特征坍塌 T-SNE
    if features_epoch1 is not None:
        plot_collapse_tsne(
            features_epoch1, features, current_epoch,
            os.path.join(save_dir, 't_sne_collapse.jpg'), logger
        )
    
    # 3. 伪标签毒化 T-SNE
    plot_poisoning_tsne(
        features, gt_labels, pseudo_labels,
        os.path.join(save_dir, 't_sne_poisoning.jpg'), logger=logger
    )
    
    # 4. 混淆矩阵
    plot_confusion_matrix(
        gt_labels, pseudo_labels,
        os.path.join(save_dir, 'confusion_matrix.jpg'),
        num_classes=num_classes, logger=logger
    )
    
    # 5. 置信度直方图
    is_correct = gt_labels == pseudo_labels
    plot_confidence_histogram(
        max_probs, is_correct,
        os.path.join(save_dir, 'confidence_histogram.jpg'), logger
    )
    
    if logger:
        logger.info(f"Epoch {current_epoch} 可视化图表生成完毕: {save_dir}")
