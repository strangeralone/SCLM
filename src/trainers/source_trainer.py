"""
源域预训练模块
标准的监督学习训练
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional
import logging

from ..utils.logger import AverageMeter


class SourceTrainer:
    """
    源域预训练器
    使用交叉熵损失进行标准监督学习
    """
    
    def __init__(
        self,
        netG: nn.Module,
        netF: nn.Module,
        netC: nn.Module,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader],
        config: Dict[str, Any],
        logger: logging.Logger,
        device: torch.device
    ):
        """
        Args:
            netG: Backbone
            netF: Bottleneck
            netC: Classifier
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器（可选）
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
        train_cfg = config['train']['source']
        self.epochs = train_cfg['epochs']
        self.lr = train_cfg['lr']
        self.lr_backbone = train_cfg['lr_backbone']
        self.log_interval = config['logging']['log_interval']
        self.save_interval = config['logging']['save_interval']
        self.save_dir = config['checkpoint']['save_dir']
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器（差异化学习率）
        self.optimizer = optim.SGD([
            {"params": netG.parameters(), "lr": self.lr_backbone},
            {"params": netF.parameters(), "lr": self.lr},
            {"params": netC.parameters(), "lr": self.lr}
        ], momentum=train_cfg['momentum'], weight_decay=train_cfg['weight_decay'])
        
        # 学习率调度器
        if train_cfg['lr_scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_cfg['lr_step'],
                gamma=train_cfg['lr_gamma']
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs
            )
        
        # 最佳模型追踪
        self.best_acc = 0.0
        
    def train(self) -> None:
        """执行完整训练流程"""
        self.logger.info(f"开始源域预训练，共 {self.epochs} 个epoch")
        
        for epoch in range(1, self.epochs + 1):
            # 训练一个epoch
            train_loss, train_acc = self._train_epoch(epoch)
            
            # 测试
            if self.test_loader is not None:
                test_loss, test_acc = self._test(epoch)
            else:
                test_loss, test_acc = 0.0, 0.0
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 记录日志
            self.logger.info(
                f"Epoch [{epoch}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )
            
            # 保存checkpoint
            if epoch % self.save_interval == 0:
                self._save_checkpoint(epoch, test_acc)
            
            # 保存最佳模型
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self._save_checkpoint(epoch, test_acc, is_best=True)
        
        self.logger.info(f"源域预训练完成，最佳测试准确率: {self.best_acc:.2f}%")
    
    def _train_epoch(self, epoch: int) -> tuple:
        """
        训练一个epoch
        
        Returns:
            (平均loss, 准确率)
        """
        self.netG.train()
        self.netF.train()
        self.netC.train()
        
        loss_meter = AverageMeter("Loss")
        acc_meter = AverageMeter("Acc")
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.epochs}",
            leave=False
        )
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            feat = self.netG(images)
            feat = self.netF(feat)
            logits = self.netC(feat)
            
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算准确率
            pred = logits.argmax(dim=1)
            acc = (pred == labels).float().mean().item() * 100
            
            # 更新meter
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(acc, images.size(0))
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'acc': f'{acc_meter.avg:.2f}%'
            })
        
        return loss_meter.avg, acc_meter.avg
    
    def _test(self, epoch: int) -> tuple:
        """
        测试
        
        Returns:
            (平均loss, 准确率)
        """
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        
        loss_meter = AverageMeter("Loss")
        acc_meter = AverageMeter("Acc")
        
        with torch.no_grad():
            for images, labels, _ in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                feat = self.netG(images)
                feat = self.netF(feat)
                logits = self.netC(feat)
                
                loss = self.criterion(logits, labels)
                
                pred = logits.argmax(dim=1)
                acc = (pred == labels).float().mean().item() * 100
                
                loss_meter.update(loss.item(), images.size(0))
                acc_meter.update(acc, images.size(0))
        
        return loss_meter.avg, acc_meter.avg
    
    def _save_checkpoint(
        self,
        epoch: int,
        test_acc: float,
        is_best: bool = False
    ) -> None:
        """保存checkpoint"""
        source_domain = self.config['data']['source']
        
        checkpoint = {
            'epoch': epoch,
            'netG_state_dict': self.netG.state_dict(),
            'netF_state_dict': self.netF.state_dict(),
            'netC_state_dict': self.netC.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'test_acc': test_acc,
            'config': self.config
        }
        
        if is_best:
            save_path = os.path.join(
                self.save_dir,
                f"source_{source_domain}_best.pth"
            )
        else:
            save_path = os.path.join(
                self.save_dir,
                f"source_{source_domain}_epoch{epoch}.pth"
            )
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint 已保存: {save_path}")
