"""
源域预训练模块
标准的监督学习训练（贴近 source.py 风格）
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
from ..utils.loss import CrossEntropyLabelSmooth


def op_copy(optimizer):
    """保存优化器初始学习率"""
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    """
    Power decay 学习率调度器
    公式: lr = lr0 * (1 + gamma * iter_num / max_iter)^(-power)
    """
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class SourceTrainer:
    """
    源域预训练器
    使用标签平滑交叉熵损失进行监督学习
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
            netG: Backbone (对应 source.py 的 netF)
            netF: Bottleneck (对应 source.py 的 netB)
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
        self.max_epoch = train_cfg['epochs']
        self.lr = train_cfg['lr']
        self.lr_backbone = train_cfg['lr_backbone']
        self.log_interval = config['logging']['log_interval']
        self.save_dir = config['checkpoint']['save_dir']
        
        # 标签平滑参数
        self.label_smooth_epsilon = train_cfg.get('label_smooth_epsilon', 0.1)
        self.num_classes = config['data']['num_classes']
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 损失函数（使用标签平滑）
        self.criterion = CrossEntropyLabelSmooth(
            num_classes=self.num_classes,
            epsilon=self.label_smooth_epsilon
        )
        
        # 优化器（差异化学习率，贴近 source.py）
        param_group = []
        for k, v in netG.named_parameters():
            param_group += [{'params': v, 'lr': self.lr_backbone}]
        for k, v in netF.named_parameters():
            param_group += [{'params': v, 'lr': self.lr}]
        for k, v in netC.named_parameters():
            param_group += [{'params': v, 'lr': self.lr}]
        
        self.optimizer = optim.SGD(param_group)
        self.optimizer = op_copy(self.optimizer)
        
        # 计算 max_iter
        self.max_iter = self.max_epoch * len(self.train_loader)
        self.interval_iter = self.max_iter // 10
        
        # 最佳模型追踪
        self.best_acc = 0.0
        
    def train(self) -> None:
        """执行完整训练流程（按 iteration 循环）"""
        self.logger.info(f"开始源域预训练，共 {self.max_epoch} 个epoch，{self.max_iter} 次iteration")
        
        iter_num = 0
        iter_source = iter(self.train_loader)
        
        self.netG.train()
        self.netF.train()
        self.netC.train()
        
        # 用于存储最佳模型
        best_netG = None
        best_netF = None
        best_netC = None
        
        pbar = tqdm(total=self.max_iter, desc="Source Training")
        
        while iter_num < self.max_iter:
            # 获取数据
            try:
                inputs, labels, _ = next(iter_source)
            except StopIteration:
                iter_source = iter(self.train_loader)
                inputs, labels, _ = next(iter_source)
            
            # 跳过 batch_size 为 1 的情况（BatchNorm 问题）
            if inputs.size(0) == 1:
                continue
            
            iter_num += 1
            
            # 更新学习率
            lr_scheduler(self.optimizer, iter_num=iter_num, max_iter=self.max_iter)
            
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            feat = self.netG(inputs)
            feat = self.netF(feat)
            logits = self.netC(feat)
            
            # 计算损失（使用标签平滑）
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 定期测试和保存
            if iter_num % self.interval_iter == 0 or iter_num == self.max_iter:
                self.netG.eval()
                self.netF.eval()
                self.netC.eval()
                
                if self.test_loader is not None:
                    test_acc, mean_ent = self._test()
                    
                    self.logger.info(
                        f"Iter [{iter_num}/{self.max_iter}] "
                        f"Test Acc: {test_acc:.2f}%, Mean Entropy: {mean_ent:.4f}"
                    )
                    
                    # 保存最佳模型
                    if test_acc >= self.best_acc:
                        self.best_acc = test_acc
                        best_netG = self.netG.state_dict().copy()
                        best_netF = self.netF.state_dict().copy()
                        best_netC = self.netC.state_dict().copy()
                
                self.netG.train()
                self.netF.train()
                self.netC.train()
        
        pbar.close()
        
        # 保存最佳模型权重（只保存 state_dict，贴近 source.py）
        if best_netG is not None:
            self._save_checkpoint(best_netG, best_netF, best_netC)
        
        self.logger.info(f"源域预训练完成，最佳测试准确率: {self.best_acc:.2f}%")
    
    def _test(self) -> tuple:
        """
        测试
        
        Returns:
            (准确率, 平均熵)
        """
        all_output = []
        all_label = []
        
        with torch.no_grad():
            for images, labels, _ in self.test_loader:
                images = images.to(self.device)
                
                feat = self.netG(images)
                feat = self.netF(feat)
                logits = self.netC(feat)
                
                all_output.append(logits.float().cpu())
                all_label.append(labels.float())
        
        all_output = torch.cat(all_output, dim=0)
        all_label = torch.cat(all_label, dim=0)
        
        # Softmax
        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        
        # 计算准确率
        accuracy = torch.sum(predict.float() == all_label).item() / float(all_label.size(0))
        
        # 计算平均熵
        entropy = -torch.sum(all_output * torch.log(all_output + 1e-5), dim=1)
        mean_ent = torch.mean(entropy).item()
        
        return accuracy * 100, mean_ent
    
    def _save_checkpoint(
        self,
        netG_state: dict,
        netF_state: dict,
        netC_state: dict
    ) -> None:
        """保存模型权重（只保存 state_dict，贴近 source.py）"""
        source_domain = self.config['data']['source']
        
        # 保存为三个独立文件（贴近 source.py 的 source_F.pt, source_B.pt, source_C.pt）
        torch.save(netG_state, os.path.join(self.save_dir, f"source_{source_domain}_G.pt"))
        torch.save(netF_state, os.path.join(self.save_dir, f"source_{source_domain}_F.pt"))
        torch.save(netC_state, os.path.join(self.save_dir, f"source_{source_domain}_C.pt"))
        
        self.logger.info(f"模型权重已保存到: {self.save_dir}")
