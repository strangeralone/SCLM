"""
Adapter-based 目标域适应模块 (改进版)

核心改进：
1. 动态原型更新 (EMA) - 原型随目标域特征漂移
2. 余弦距离伪标签 - 比全连接层更鲁棒
3. Huber Loss 特征对齐

Loss 组成：
1. Alignment Loss: Huber(f_new, dynamic_prototype) - 特征对齐
2. Diversity Loss: 防止所有特征坍塌到同一类
3. Entropy Loss: 让预测更自信
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Tuple
import logging

from ..utils.logger import AverageMeter


class AdapterTrainer:
    """
    Adapter-based 目标域适应训练器 (改进版)
    
    改进点:
    1. 动态原型: 用 EMA 更新，跟随目标域分布
    2. 余弦伪标签: 用距离而非全连接层
    3. Huber Loss: 鲁棒对齐
    """
    
    def __init__(
        self,
        netG: nn.Module,
        netF: nn.Module,
        netC: nn.Module,
        adapter: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        logger: logging.Logger,
        device: torch.device
    ):
        """
        Args:
            netG: Backbone (冻结)
            netF: Bottleneck (冻结)
            netC: Classifier (冻结，用于初始化原型)
            adapter: FeatureAdapter (可训练)
            train_loader: 目标域训练数据加载器
            test_loader: 目标域测试数据加载器
            config: 配置字典
            logger: 日志器
            device: 计算设备
        """
        self.netG = netG
        self.netF = netF
        self.netC = netC
        self.adapter = adapter
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.logger = logger
        self.device = device
        
        # 冻结源模型
        self._freeze_source_model()
        
        # 训练配置
        adapter_cfg = config['train'].get('adapter', {})
        self.epochs = adapter_cfg.get('epochs', 15)
        self.lr = adapter_cfg.get('lr', 1e-3)
        
        # Loss 权重
        self.align_weight = adapter_cfg.get('align_weight', 1.0)
        self.div_weight = adapter_cfg.get('div_weight', 0.1)
        self.ent_weight = adapter_cfg.get('ent_weight', 0.1)
        
        # Huber Loss 参数
        self.huber_delta = adapter_cfg.get('huber_delta', 1.0)
        
        # 置信度阈值 (用于筛选可靠样本)
        self.confidence_threshold = adapter_cfg.get('confidence_threshold', 0.7)
        
        # ===== 新增: 动态原型配置 =====
        self.prototype_momentum = adapter_cfg.get('prototype_momentum', 0.9)
        self.use_dynamic_prototype = adapter_cfg.get('use_dynamic_prototype', True)
        
        self.save_dir = config['checkpoint']['save_dir']
        self.num_classes = config['data']['num_classes']
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # ===== 初始化原型 (从分类器权重) =====
        self._init_prototypes()
        
        # 优化器 (只优化 Adapter)
        self.optimizer = optim.Adam(
            adapter.parameters(),
            lr=self.lr,
            weight_decay=adapter_cfg.get('weight_decay', 1e-4)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs
        )
        
        # Huber Loss
        self.huber_loss = nn.HuberLoss(reduction='none', delta=self.huber_delta)
        
    def _freeze_source_model(self):
        """冻结源模型参数"""
        for model in [self.netG, self.netF, self.netC]:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
        self.logger.info("源模型已冻结 (netG, netF, netC)")
    
    def _init_prototypes(self):
        """从分类器权重初始化原型"""
        # 分类器权重: [num_classes, feat_dim]
        with torch.no_grad():
            self.prototypes = self.netC.fc.weight.data.clone().to(self.device)
            # L2 归一化，便于余弦计算
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        self.logger.info(f"原型初始化完成: shape={self.prototypes.shape}")
        
    def _cosine_similarity(self, features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        计算余弦相似度
        
        Args:
            features: [B, feat_dim]
            prototypes: [num_classes, feat_dim]
            
        Returns:
            similarity: [B, num_classes]
        """
        # L2 归一化
        features_norm = F.normalize(features, p=2, dim=1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=1)
        
        # 余弦相似度 = 内积
        similarity = torch.mm(features_norm, prototypes_norm.t())
        return similarity
    
    def _get_pseudo_labels(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于余弦距离生成伪标签
        
        Args:
            features: [B, feat_dim]
            
        Returns:
            pseudo_labels: [B]
            confidence: [B] (余弦相似度，0~1)
            mask: [B] (高置信度 mask)
        """
        # 计算与所有原型的余弦相似度
        similarity = self._cosine_similarity(features, self.prototypes)
        
        # 伪标签 = 最相似的类
        confidence, pseudo_labels = similarity.max(dim=1)
        
        # 高置信度 mask
        mask = (confidence > self.confidence_threshold).float()
        
        return pseudo_labels, confidence, mask
    
    def _update_prototypes(self, features: torch.Tensor, pseudo_labels: torch.Tensor, mask: torch.Tensor):
        """
        EMA 更新原型
        
        Args:
            features: [B, feat_dim] - 适应后的特征 f_new
            pseudo_labels: [B]
            mask: [B] - 只用高置信度样本
        """
        if not self.use_dynamic_prototype:
            return
            
        with torch.no_grad():
            for c in range(self.num_classes):
                # 找到这个类的高置信度样本
                class_mask = (pseudo_labels == c) & (mask > 0.5)
                
                if class_mask.sum() > 0:
                    # 计算这些样本的均值
                    class_features = features[class_mask]
                    class_mean = class_features.mean(dim=0)
                    class_mean = F.normalize(class_mean, p=2, dim=0)
                    
                    # EMA 更新
                    self.prototypes[c] = (
                        self.prototype_momentum * self.prototypes[c] +
                        (1 - self.prototype_momentum) * class_mean
                    )
                    # 重新归一化
                    self.prototypes[c] = F.normalize(self.prototypes[c], p=2, dim=0)
        
    def train(self) -> None:
        """执行 Adapter 训练"""
        self.logger.info(f"开始 Adapter 训练，共 {self.epochs} 个epoch")
        self.logger.info(f"Loss 权重: align={self.align_weight}, div={self.div_weight}, ent={self.ent_weight}")
        self.logger.info(f"动态原型: {self.use_dynamic_prototype}, momentum={self.prototype_momentum}")
        
        # 初始测试 (Source Only)
        init_acc = self._evaluate()
        self.logger.info(f"初始目标域准确率 (Source Only): {init_acc:.2f}%")
        
        best_acc = init_acc
        
        for epoch in range(1, self.epochs + 1):
            # 训练一个 epoch
            losses, stats = self._train_epoch(epoch)
            
            # 评估
            acc = self._evaluate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 记录日志
            self.logger.info(
                f"Epoch [{epoch}/{self.epochs}] "
                f"Loss: {losses['total']:.4f} "
                f"(Align: {losses['align']:.4f}, Div: {losses['div']:.4f}, Ent: {losses['ent']:.4f}) | "
                f"Acc: {acc:.2f}% | Reliable: {stats['reliable_ratio']:.1f}% | LR: {current_lr:.6f}"
            )
            
            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                self._save_checkpoint(epoch, acc, is_best=True)
        
        self.logger.info(f"Adapter 训练完成，最佳准确率: {best_acc:.2f}% (提升 {best_acc - init_acc:.2f}%)")
        
    def _train_epoch(self, epoch: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        训练一个 epoch
        
        Returns:
            损失字典, 统计字典
        """
        # Adapter 设为训练模式
        self.adapter.train()
        
        loss_meter = AverageMeter("Total")
        align_meter = AverageMeter("Align")
        div_meter = AverageMeter("Div")
        ent_meter = AverageMeter("Ent")
        reliable_meter = AverageMeter("Reliable")
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False
        )
        
        for batch_idx, (images, _, indices) in enumerate(pbar):
            images = images.to(self.device)
            
            # ========== Step 1: 提取原始特征 (冻结) ==========
            with torch.no_grad():
                feat_backbone = self.netG(images)
                f_t = self.netF(feat_backbone)  # [B, feat_dim]
            
            # ========== Step 2: Adapter 修正 ==========
            velocity = self.adapter(f_t)
            f_new = f_t + velocity  # 适应后的特征
            
            # ========== Step 3: 用余弦距离生成伪标签 ==========
            # 注意：用 f_t 生成伪标签（稳定锚点），不是 f_new
            with torch.no_grad():
                pseudo_labels, confidence, mask = self._get_pseudo_labels(f_t)
            
            # 获取目标原型
            target_proto = self.prototypes[pseudo_labels]  # [B, feat_dim]
            
            # ========== Step 4: 计算 Loss ==========
            
            # Loss 1: Alignment Loss (Huber)
            # 将 f_new 推向对应的原型
            raw_align = self.huber_loss(f_new, target_proto).mean(dim=1)  # [B]
            align_loss = (raw_align * mask).sum() / (mask.sum() + 1e-8)
            
            # 用 f_new 通过余弦相似度计算 "logits" (用于 div/ent loss)
            similarity_new = self._cosine_similarity(f_new, self.prototypes)
            probs_new = F.softmax(similarity_new / 0.1, dim=1)  # temperature=0.1
            
            # Loss 2: Diversity Loss (防止类别坍塌)
            mean_probs = probs_new.mean(dim=0)
            div_loss = torch.sum(mean_probs * torch.log(mean_probs + 1e-10))
            
            # Loss 3: Entropy Loss (让预测更自信)
            ent_loss = -torch.sum(probs_new * torch.log(probs_new + 1e-10), dim=1).mean()
            
            # 总 Loss
            total_loss = (
                self.align_weight * align_loss +
                self.div_weight * div_loss +
                self.ent_weight * ent_loss
            )
            
            # ========== Step 5: 反向传播 ==========
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # ========== Step 6: EMA 更新原型 ==========
            # 用适应后的高置信度特征更新原型
            with torch.no_grad():
                _, conf_new, mask_new = self._get_pseudo_labels(f_new.detach())
            self._update_prototypes(f_new.detach(), pseudo_labels, mask_new)
            
            # 更新统计
            loss_meter.update(total_loss.item(), images.size(0))
            align_meter.update(align_loss.item(), images.size(0))
            div_meter.update(div_loss.item(), images.size(0))
            ent_meter.update(ent_loss.item(), images.size(0))
            reliable_meter.update(mask.mean().item() * 100, images.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'rel': f'{reliable_meter.avg:.1f}%'
            })
        
        return {
            'total': loss_meter.avg,
            'align': align_meter.avg,
            'div': div_meter.avg,
            'ent': ent_meter.avg
        }, {
            'reliable_ratio': reliable_meter.avg
        }
    
    def _evaluate(self) -> float:
        """
        评估目标域准确率
        使用余弦距离分类（与训练一致）
        
        Returns:
            准确率 (%)
        """
        self.adapter.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                feat_backbone = self.netG(images)
                f_t = self.netF(feat_backbone)
                
                # Adapter 修正
                f_new = self.adapter.adapt(f_t)
                
                # 余弦距离分类
                similarity = self._cosine_similarity(f_new, self.prototypes)
                pred = similarity.argmax(dim=1)
                
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        self.adapter.train()
        return 100.0 * correct / total
    
    def _save_checkpoint(
        self,
        epoch: int,
        acc: float,
        is_best: bool = False
    ) -> None:
        """保存 checkpoint"""
        source = self.config['data']['source']
        target = self.config['data']['target']
        
        checkpoint = {
            'epoch': epoch,
            'adapter_state_dict': self.adapter.state_dict(),
            'prototypes': self.prototypes.cpu(),  # 保存更新后的原型
            'optimizer_state_dict': self.optimizer.state_dict(),
            'acc': acc,
            'config': self.config
        }
        
        if is_best:
            save_path = os.path.join(
                self.save_dir,
                f"adapter_{source}2{target}_best.pth"
            )
        else:
            save_path = os.path.join(
                self.save_dir,
                f"adapter_{source}2{target}_epoch{epoch}.pth"
            )
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Adapter checkpoint 已保存: {save_path}")
