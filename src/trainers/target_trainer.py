"""
目标域适应模块
基于SHOT的信息最大化 + 伪标签训练 + 扰动一致性过滤

核心创新：利用扰动一致性筛选可靠的伪标签
- 正确的伪标签对扰动鲁棒（一致性高）
- 错误的伪标签对扰动敏感（一致性低）
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
import logging
import numpy as np

from ..utils.logger import AverageMeter


class TargetTrainer:
    """
    目标域适应训练器
    
    主要方法:
    1. 信息最大化 (SHOT): 熵最小化 + 多样性正则
    2. 伪标签训练
    3. 扰动一致性过滤 (Self-Guidance inspired)
    """
    
    def __init__(
        self,
        netG: nn.Module,
        netF: nn.Module,
        netC: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        logger: logging.Logger,
        device: torch.device
    ):
        """
        Args:
            netG: Backbone
            netF: Bottleneck
            netC: Classifier
            train_loader: 目标域训练数据加载器
            test_loader: 目标域测试数据加载器
            config: 配置字典
            logger: 日志器
            device: 计算设备
        """
        self.netG = netG
        self.netF = netF
        self.netC = netC
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.logger = logger
        self.device = device
        
        # 训练配置
        train_cfg = config['train']['target']
        self.epochs = train_cfg['epochs']
        self.lr = train_cfg['lr']
        self.lr_backbone = train_cfg['lr_backbone']
        self.ent_weight = train_cfg['ent_weight']
        self.div_weight = train_cfg['div_weight']
        self.cls_weight = train_cfg['cls_weight']
        
        self.log_interval = config['logging']['log_interval']
        self.save_dir = config['checkpoint']['save_dir']
        self.num_classes = config['data']['num_classes']
        
        # 扰动一致性配置
        self.perturbation_cfg = train_cfg.get('perturbation', {})
        self.use_perturbation = self.perturbation_cfg.get('enabled', False)
        self.noise_scale = self.perturbation_cfg.get('noise_scale', 0.1)
        self.num_perturbations = self.perturbation_cfg.get('num_perturbations', 10)
        self.consistency_threshold = self.perturbation_cfg.get('threshold', 0.8)
        self.use_soft_weight = self.perturbation_cfg.get('soft_weight', True)
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 优化器（只优化bottleneck和classifier，backbone可选）
        self.optimizer = optim.SGD([
            {"params": netG.parameters(), "lr": self.lr_backbone},
            {"params": netF.parameters(), "lr": self.lr},
            {"params": netC.parameters(), "lr": self.lr}
        ], momentum=train_cfg['momentum'], weight_decay=train_cfg['weight_decay'])
        
        # 伪标签和一致性分数存储
        self.pseudo_labels = None
        self.consistency_scores = None
        
    def train(self) -> None:
        """执行目标域适应"""
        self.logger.info(f"开始目标域适应，共 {self.epochs} 个epoch")
        if self.use_perturbation:
            self.logger.info(f"启用扰动一致性过滤: noise_scale={self.noise_scale}, "
                           f"num_perturbations={self.num_perturbations}, "
                           f"threshold={self.consistency_threshold}")
        
        # 初始测试
        init_acc = self._evaluate()
        self.logger.info(f"初始目标域准确率 (Source Only): {init_acc:.2f}%")
        
        best_acc = init_acc
        
        for epoch in range(1, self.epochs + 1):
            # 生成伪标签和一致性分数
            if self.use_perturbation:
                self.pseudo_labels, self.consistency_scores = self._generate_pseudo_labels_with_consistency()
                reliable_ratio = (self.consistency_scores >= self.consistency_threshold).float().mean() * 100
                self.logger.info(f"Epoch {epoch}: 可靠伪标签比例: {reliable_ratio:.2f}%")
            else:
                self.pseudo_labels = self._generate_pseudo_labels()
            
            # 训练一个epoch
            train_loss, ent_loss, div_loss, cls_loss = self._train_epoch(epoch)
            
            # 评估
            acc = self._evaluate()
            
            # 记录日志
            self.logger.info(
                f"Epoch [{epoch}/{self.epochs}] "
                f"Loss: {train_loss:.4f} (Ent: {ent_loss:.4f}, Div: {div_loss:.4f}, Cls: {cls_loss:.4f}) | "
                f"Target Acc: {acc:.2f}%"
            )
            
            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                self._save_checkpoint(epoch, acc, is_best=True)
        
        self.logger.info(f"目标域适应完成，最佳准确率: {best_acc:.2f}%")
    
    def _generate_pseudo_labels(self) -> torch.Tensor:
        """生成伪标签（不带一致性）"""
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        
        all_probs = []
        
        with torch.no_grad():
            for images, _, indices in self.train_loader:
                images = images.to(self.device)
                
                feat = self.netG(images)
                feat = self.netF(feat)
                logits = self.netC(feat)
                
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
        
        all_probs = torch.cat(all_probs, dim=0)
        pseudo_labels = all_probs.argmax(dim=1)
        
        return pseudo_labels
    
    def _generate_pseudo_labels_with_consistency(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成伪标签并计算扰动一致性分数
        
        Returns:
            pseudo_labels: 伪标签 [N]
            consistency_scores: 一致性分数 [N]，范围 0-1
        """
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        
        all_pseudo_labels = []
        all_consistency = []
        
        with torch.no_grad():
            for images, _, indices in tqdm(self.train_loader, desc="Computing consistency", leave=False):
                images = images.to(self.device)
                
                # 原始预测
                feat = self.netG(images)
                feat = self.netF(feat)
                logits_orig = self.netC(feat)
         
                probs_orig = F.softmax(logits_orig, dim=1)
                pred_orig = logits_orig.argmax(dim=1)
                
                # 计算扰动一致性
                consistent_count = torch.zeros(images.size(0), device=self.device)
                
                for _ in range(self.num_perturbations):
                    # 添加高斯噪声扰动
                    noise = torch.randn_like(images) * self.noise_scale
                    images_perturbed = torch.clamp(images + noise, -3, 3)
                    
                    # 扰动后预测
                    f = self.netG(images_perturbed)
                    f = self.netF(f)
                    logits_perturbed = self.netC(f)
                    
                    pred_perturbed = logits_perturbed.argmax(dim=1)
                    
                    consistent_count += (pred_perturbed == pred_orig).float()
                
                consistency = consistent_count / self.num_perturbations
                
                all_pseudo_labels.append(pred_orig.cpu())
                all_consistency.append(consistency.cpu())
        
        pseudo_labels = torch.cat(all_pseudo_labels, dim=0)
        consistency_scores = torch.cat(all_consistency, dim=0)
        
        return pseudo_labels, consistency_scores
    
    def _train_epoch(self, epoch: int) -> Tuple[float, float, float, float]:
        """
        训练一个epoch
        
        Returns:
            (总loss, 熵loss, 多样性loss, 分类loss)
        """
        self.netG.train()
        self.netF.train()
        self.netC.train()
        
        loss_meter = AverageMeter("Loss")
        ent_meter = AverageMeter("Ent")
        div_meter = AverageMeter("Div")
        cls_meter = AverageMeter("Cls")
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False
        )
        
        for batch_idx, (images, _, indices) in enumerate(pbar):
            images = images.to(self.device)
            
            # 前向传播
            # logits, features = self.model(images, return_features=True)
            features = self.netG(images)
            features = self.netF(features) # This is 'bottleneck' feature
            logits = self.netC(features)
            
            probs = F.softmax(logits, dim=1)
            
            # 1. 熵最小化损失
            ent_loss = self._entropy_loss(probs)
            
            # 2. 多样性正则（防止所有样本预测同一类）
            div_loss = self._diversity_loss(probs)
            
            # 3. 伪标签分类损失（带一致性加权）
            if self.pseudo_labels is not None:
                pseudo = self.pseudo_labels[indices].to(self.device)
                
                if self.use_perturbation and self.consistency_scores is not None:
                    # 获取当前batch的一致性分数
                    consistency = self.consistency_scores[indices].to(self.device)
                    
                    if self.use_soft_weight:
                        # 软加权：用一致性分数作为样本权重
                        sample_weights = consistency
                        cls_loss = (F.cross_entropy(logits, pseudo, reduction='none') * sample_weights).mean()
                    else:
                        # 硬过滤：只使用高一致性样本
                        reliable_mask = consistency >= self.consistency_threshold
                        if reliable_mask.sum() > 0:
                            cls_loss = F.cross_entropy(logits[reliable_mask], pseudo[reliable_mask])
                        else:
                            cls_loss = torch.tensor(0.0, device=self.device)
                else:
                    cls_loss = F.cross_entropy(logits, pseudo)
            else:
                cls_loss = torch.tensor(0.0, device=self.device)
            
            # 总损失
            loss = (
                self.ent_weight * ent_loss +
                self.div_weight * div_loss +
                self.cls_weight * cls_loss
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新meter
            loss_meter.update(loss.item(), images.size(0))
            ent_meter.update(ent_loss.item(), images.size(0))
            div_meter.update(div_loss.item(), images.size(0))
            cls_meter.update(cls_loss.item(), images.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'ent': f'{ent_meter.avg:.4f}'
            })
        
        return loss_meter.avg, ent_meter.avg, div_meter.avg, cls_meter.avg
    
    def _entropy_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """
        计算熵损失（信息最大化）
        
        Args:
            probs: 预测概率 [B, C]
            
        Returns:
            熵损失
        """
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        return torch.mean(entropy)
    
    def _diversity_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """
        计算多样性损失（防止类别坍塌）
        
        Args:
            probs: 预测概率 [B, C]
            
        Returns:
            多样性损失（负熵，希望batch内预测均匀分布）
        """
        mean_probs = probs.mean(dim=0)
        diversity = torch.sum(mean_probs * torch.log(mean_probs + 1e-10))
        return diversity
    
    def _evaluate(self) -> float:
        """
        评估目标域准确率
        
        Returns:
            准确率 (%)
        """
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                f = self.netG(images)
                f = self.netF(f)
                logits = self.netC(f)
                
                pred = logits.argmax(dim=1)
                
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        return 100.0 * correct / total
    
    def _save_checkpoint(
        self,
        epoch: int,
        acc: float,
        is_best: bool = False
    ) -> None:
        """保存checkpoint"""
        source = self.config['data']['source']
        target = self.config['data']['target']
        
        checkpoint = {
            'epoch': epoch,
            'netG_state_dict': self.netG.state_dict(),
            'netF_state_dict': self.netF.state_dict(),
            'netC_state_dict': self.netC.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'acc': acc,
            'config': self.config
        }
        
        if is_best:
            save_path = os.path.join(
                self.save_dir,
                f"target_{source}2{target}_best.pth"
            )
        else:
            save_path = os.path.join(
                self.save_dir,
                f"target_{source}2{target}_epoch{epoch}.pth"
            )
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint 已保存: {save_path}")
    
    # ============ 扩展方法：单独计算一致性 ============
    
    def compute_perturbation_consistency(
        self,
        images: torch.Tensor,
        num_perturbations: int = None
    ) -> torch.Tensor:
        """
        计算扰动一致性分数
        
        核心思想：正确的伪标签对扰动鲁棒
        
        Args:
            images: 输入图像 [B, C, H, W]
            num_perturbations: 扰动次数（None则使用配置值）
            
        Returns:
            一致性分数 [B]，越高越可靠
        """
        if num_perturbations is None:
            num_perturbations = self.num_perturbations
            
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        
        with torch.no_grad():
            # 原始预测
            f = self.netG(images)
            f = self.netF(f)
            logits_orig = self.netC(f)
            
            pred_orig = logits_orig.argmax(dim=1)
            
            # 扰动预测
            consistent_count = torch.zeros(images.size(0), device=self.device)
            
            for _ in range(num_perturbations):
                noise = torch.randn_like(images) * self.noise_scale
                images_perturbed = torch.clamp(images + noise, -3, 3)
                
                f = self.netG(images_perturbed)
                f = self.netF(f)
                logits_perturbed = self.netC(f)
                
                pred_perturbed = logits_perturbed.argmax(dim=1)
                
                consistent_count += (pred_perturbed == pred_orig).float()
            
            consistency_score = consistent_count / num_perturbations
        
        return consistency_score
    
    def get_reliability_statistics(self) -> Dict[str, float]:
        """
        获取当前伪标签的可靠性统计信息
        
        Returns:
            包含可靠性统计的字典
        """
        if self.consistency_scores is None:
            return {}
        
        scores = self.consistency_scores.numpy()
        
        return {
            'mean_consistency': float(np.mean(scores)),
            'std_consistency': float(np.std(scores)),
            'reliable_ratio_0.8': float((scores >= 0.8).mean()),
            'reliable_ratio_0.9': float((scores >= 0.9).mean()),
            'unreliable_ratio_0.5': float((scores < 0.5).mean()),
        }
