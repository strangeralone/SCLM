"""
目标域适应模块 - ProDe 风格
基于 Prompt Learning + IIC Loss 的伪标签训练
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Tuple, List
import logging
from copy import deepcopy

from ..utils.logger import AverageMeter
from ..utils.losses import IID_loss, entropy_loss
from ..models.clip_module import test_time_tuning
from clip.custom_clip import ClipTestTimeTuning

def lr_scheduler(optimizer, iter_num: int, max_iter: int, gamma: float = 10, power: float = 0.75):
    """学习率调度器"""
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
    return optimizer


def op_copy(optimizer):
    """保存优化器初始学习率"""
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


class TargetTrainer:
    """
    目标域适应训练器 - ProDe 风格
    
    核心组件:
    1. Logits Bank: 存储源模型初始预测
    2. CoOp TTA: Test-time prompt tuning
    3. IIC Loss: 对齐源模型和 CLIP 输出
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
        self.max_epoch = train_cfg['epochs']
        self.lr = train_cfg['lr']
        self.lr_backbone = train_cfg['lr_backbone']
        
        # ProDe 配置
        prode_cfg = train_cfg.get('prode', {})
        self.gent_par = prode_cfg.get('gent_par', 0.4)  # CE loss 权重
        self.iic_par = prode_cfg.get('iic_par', 1.3)    # IIC loss 权重
        self.div_weight = prode_cfg.get('div_weight', 1.0)  # 多样性 loss 权重
        self.epsilon = prode_cfg.get('epsilon', 1e-5)
        self.tta_steps = prode_cfg.get('tta_steps', 1)
        self.logits_bank_decay = prode_cfg.get('logits_bank_decay', 0.1)
        
        # CLIP 配置
        clip_cfg = prode_cfg.get('clip', {})
        self.clip_arch = clip_cfg.get('arch', 'ViT-B-32')
        self.n_ctx = clip_cfg.get('n_ctx', 4)
        self.ctx_init = clip_cfg.get('ctx_init', 'a_photo_of_a')
        self.clip_lr_decay = clip_cfg.get('lr_decay', 0.1)
        
        self.log_interval = config['logging']['log_interval']
        self.save_dir = config['checkpoint']['save_dir']
        self.num_classes = config['data']['num_classes']
        
        # 获取类别名称
        self.classnames = self._get_classnames()
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 初始化优化器
        self._init_optimizers(train_cfg)
        
        # 初始化 CLIP 模块
        self._init_clip_module()
        
        # 初始化 Logits Bank
        self.logits_bank = None
    
    def _get_classnames(self) -> List[str]:
        """获取类别名称（从配置或文件加载）"""
        # 优先从配置读取
        classnames = self.config.get('data', {}).get('classnames', None)
        if classnames is not None:
            return classnames
        
        # 尝试从类名文件加载
        data_root = self.config.get('data', {}).get('root', 'data/officehome')
        classnames_file = os.path.join(data_root, 'classname.txt')
        
        if os.path.exists(classnames_file):
            with open(classnames_file, 'r') as f:
                classnames = [line.strip() for line in f if line.strip()]
            self.logger.info(f"从 {classnames_file} 加载了 {len(classnames)} 个类名")
            return classnames
        
        # 默认使用数字（不推荐，CLIP 无法理解语义）
        self.logger.warning("未找到类名文件，使用数字作为类名（CLIP 效果会很差）")
        return [str(i) for i in range(self.num_classes)]
    
    def _init_optimizers(self, train_cfg: Dict):
        """初始化优化器（与 ProDe 一致：所有参数使用相同 LR）"""
        # 源模型优化器 - 所有参数使用相同学习率
        param_group = []
        for name, param in self.netG.named_parameters():
            param_group.append({'params': param, 'lr': self.lr})
        for name, param in self.netF.named_parameters():
            param_group.append({'params': param, 'lr': self.lr})
        for name, param in self.netC.named_parameters():
            param_group.append({'params': param, 'lr': self.lr})
        
        self.optimizer = optim.SGD(
            param_group,
            momentum=train_cfg.get('momentum', 0.9),
            weight_decay=train_cfg.get('weight_decay', 1e-3),
            nesterov=True
        )
        self.optimizer = op_copy(self.optimizer)
    
    def _init_clip_module(self):
        """初始化 CLIP 模块"""
        self.logger.info(f"初始化 CLIP 模块: arch={self.clip_arch}, n_ctx={self.n_ctx}")
        
        self.clip_model = ClipTestTimeTuning(
            classnames=self.classnames,
            arch=self.clip_arch,
            batch_size=None,
            n_ctx=self.n_ctx,
            ctx_init=self.ctx_init,
            device=self.device
        )
        
        # CLIP prompt 优化器
        param_group_clip = []
        for name, param in self.clip_model.prompt_learner.named_parameters():
            if param.requires_grad:
                param_group_clip.append({
                    'params': param,
                    'lr': self.lr * self.clip_lr_decay
                })
        
        self.optimizer_clip = optim.SGD(param_group_clip)
        self.optimizer_clip = op_copy(self.optimizer_clip)
        self.optim_clip_state = deepcopy(self.optimizer_clip.state_dict())
        
        self.logger.info("CLIP 模块初始化完成")
    
    def _init_logits_bank(self):
        """初始化 Logits Bank（用源模型预测，与原版 ProDe 一致）"""
        self.logger.info("初始化 Logits Bank（用源模型预测）...")
        
        num_samples = len(self.train_loader.dataset)
        self.logits_bank = torch.zeros(num_samples, self.num_classes).to(self.device)
        
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        
        with torch.no_grad():
            for images, _, indices in tqdm(self.train_loader, desc="Building logits bank", leave=False):
                images = images.to(self.device)
                
                feat = self.netG(images)
                feat = self.netF(feat)
                logits = self.netC(feat)
                
                # logits_softmax = nn.Softmax(dim=1)(logits)
                self.logits_bank[indices] = logits.detach()
        
        self.logger.info(f"Logits Bank 初始化完成，形状: {self.logits_bank.shape}")
    
    def train(self) -> None:
        """执行目标域适应训练"""
        self.logger.info(f"开始 ProDe 风格目标域适应，共 {self.max_epoch} 个 epoch")
        self.logger.info(f"超参数: gent_par={self.gent_par}, iic_par={self.iic_par}, "
                        f"tta_steps={self.tta_steps}")
        
        # 初始化 Logits Bank
        self._init_logits_bank()
        
        # 初始评估
        init_acc = self._evaluate()
        self.logger.info(f"初始目标域准确率 (Source Only): {init_acc:.2f}%")
        
        best_acc = init_acc
        max_iter = self.max_epoch * len(self.train_loader)
        iter_num = 0
        
        for epoch in range(1, self.max_epoch + 1):
            loss_meter = AverageMeter("Loss")
            iic_meter = AverageMeter("IIC")
            ce_meter = AverageMeter("CE")
            div_meter = AverageMeter("Div")
            
            self.netG.train()
            self.netF.train()
            self.netC.train()
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epoch}", leave=False)
            
            for batch_idx, (images, labels, indices) in enumerate(pbar):
                if images.size(0) == 1:
                    continue
                
                images = images.to(self.device)
                iter_num += 1
                
                # 更新学习率
                lr_scheduler(self.optimizer, iter_num, max_iter)
                
                # 源模型前向传播
                feat = self.netG(images)
                feat = self.netF(feat)
                outputs = self.netC(feat)
                softmax_out = F.softmax(outputs, dim=1)
                
                # CLIP TTA
                outputs_detach = outputs.clone().detach()
                self.optimizer_clip.load_state_dict(self.optim_clip_state)
                
                clip_score = test_time_tuning(
                    self.clip_model,
                    images,
                    self.optimizer_clip,
                    outputs_detach,
                    self.tta_steps
                )
                clip_score = clip_score.float()
                
                # 计算修正后的 CLIP 分数
                with torch.no_grad():
                    # 确保 logits_bank 在正确设备上
                    bank_logits = self.logits_bank[indices].to(self.device)
                    new_clip = (outputs_detach - self.logits_bank_decay * bank_logits) + clip_score
                    clip_score_sm = F.softmax(new_clip, dim=1)
                    clip_pred = new_clip.argmax(dim=1)
                
                # 损失计算（严格按照原版 ProDe 顺序）
                # 1. IIC Loss: 对齐源模型输出和 CLIP 输出
                iic_loss = IID_loss(softmax_out, clip_score_sm)
                classifier_loss = self.iic_par * iic_loss
                
                # 2. 多样性损失（gentropy）
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + self.epsilon))
                classifier_loss = classifier_loss - self.div_weight * gentropy_loss
                
                # 3. CE Loss: 用 CLIP 预测作为伪标签（最后加）
                ce_loss = F.cross_entropy(outputs, clip_pred)
                total_loss = self.gent_par * ce_loss + classifier_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # 更新 meters
                loss_meter.update(total_loss.item(), images.size(0))
                iic_meter.update(iic_loss.item(), images.size(0))
                ce_meter.update(ce_loss.item(), images.size(0))
                div_meter.update(gentropy_loss.item(), images.size(0))
                
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'iic': f'{iic_meter.avg:.4f}'
                })
            
            # Epoch 结束评估
            acc = self._evaluate()
            
            self.logger.info(
                f"Epoch [{epoch}/{self.max_epoch}] "
                f"Loss: {loss_meter.avg:.4f} (IIC: {iic_meter.avg:.4f}, CE: {ce_meter.avg:.4f}, Gentropy: {div_meter.avg:.4f}) | "
                f"Target Acc: {acc:.2f}%"
            )
            
            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                self._save_checkpoint(epoch, acc, is_best=True)
        
        self.logger.info(f"目标域适应完成，最佳准确率: {best_acc:.2f}%")
    
    def _evaluate(self) -> float:
        """评估目标域准确率"""
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                feat = self.netG(images)
                feat = self.netF(feat)
                logits = self.netC(feat)
                
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        return 100.0 * correct / total
    
    def _save_checkpoint(self, epoch: int, acc: float, is_best: bool = False) -> None:
        """保存 checkpoint"""
        source = self.config['data']['source']
        target = self.config['data']['target']
        
        checkpoint = {
            'epoch': epoch,
            'netG_state_dict': self.netG.state_dict(),
            'netF_state_dict': self.netF.state_dict(),
            'netC_state_dict': self.netC.state_dict(),
            'prompt_learner_state_dict': self.clip_model.prompt_learner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'acc': acc,
            'config': self.config
        }
        
        if is_best:
            save_path = os.path.join(self.save_dir, f"target_{source}2{target}_best.pth")
            # 单独保存 prompt 权重
            prompt_path = os.path.join(self.save_dir, f"prompt_{source}2{target}_best.pt")
            torch.save(self.clip_model.prompt_learner.state_dict(), prompt_path)
            self.logger.info(f"Prompt 权重已保存: {prompt_path}")
        else:
            save_path = os.path.join(self.save_dir, f"target_{source}2{target}_epoch{epoch}.pth")
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint 已保存: {save_path}")
