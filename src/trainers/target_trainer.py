"""
目标域适应模块 - SCLM (Source-CLIP Learning with Mixup)
基于三方一致性 + KNN标签修正 + Mixup + CLIP文本锚点修正
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
import numpy as np
from copy import deepcopy
from pathlib import Path

from ..utils.logger import AverageMeter
from ..utils.losses import IID_loss, entropy_loss
from ..utils.visualization import generate_all_visualizations
from ..models.clip_module import test_time_tuning
from ..models.network import FeatureProjector
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
    目标域适应训练器 - SCLM
    
    核心流程:
    1. 三方一致性检验（源模型 + CLIP正常 + CLIP高斯扰动）→ 筛选高可信样本
    2. KNN 用高可信样本的 bottleneck 特征修正低可信样本标签
    3. Mixup 高可信种子与低可信样本 + CLIP文本锚点方向修正
    4. Loss: IIC + CE(三方一致伪标签) + Mixup CE + 锚点对齐 - Diversity
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
        
        # ProDe 基础配置
        prode_cfg = train_cfg.get('prode', {})
        self.gent_par = prode_cfg.get('gent_par', 0.4)
        self.iic_par = prode_cfg.get('iic_par', 1.3)
        self.div_weight = prode_cfg.get('div_weight', 1.0)
        self.epsilon = prode_cfg.get('epsilon', 1e-5)
        self.tta_steps = prode_cfg.get('tta_steps', 1)
        self.logits_bank_decay = prode_cfg.get('logits_bank_decay', 0.1)
        
        # CLIP 配置
        clip_cfg = prode_cfg.get('clip', {})
        self.clip_arch = clip_cfg.get('arch', 'ViT-B-32')
        self.n_ctx = clip_cfg.get('n_ctx', 4)
        self.ctx_init = clip_cfg.get('ctx_init', 'a_photo_of_a')
        self.clip_lr_decay = clip_cfg.get('lr_decay', 0.1)
        
        # ========== 新增配置 ==========
        # 三方一致性
        self.consensus_conf_threshold = prode_cfg.get('consensus_conf_threshold', 0.7)
        self.noise_std = prode_cfg.get('noise_std', 0.02)
        
        # KNN
        self.knn_k = prode_cfg.get('knn_k', 7)
        
        # Mixup
        self.mixup_alpha = prode_cfg.get('mixup_alpha', 0.8)
        self.mixup_weight = prode_cfg.get('mixup_weight', 0.3)
        self.mixup_label_bias = prode_cfg.get('mixup_label_bias', 0.7)
        
        # CLIP 文本锚点
        self.anchor_weight = prode_cfg.get('anchor_weight', 0.1)
        self.projector_hidden_dim = prode_cfg.get('projector_hidden_dim', 384)
        
        # 可视化
        self.vis_interval = prode_cfg.get('vis_interval', 5)
        
        # 通用配置
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
        
        # 初始化投影层
        self._init_projector()
        
        # 初始化 Logits Bank
        self.logits_bank = None
        
        # 可视化: 存储 Epoch 1 特征用于坍塌图对比
        self.epoch1_features = None
        
        # 确定可视化保存目录
        self.vis_base_dir = self._get_vis_base_dir()
    
    def _get_vis_base_dir(self) -> str:
        """获取可视化保存的基础目录（与 log 文件同名）"""
        # 尝试从 logger 获取 log 文件路径
        log_path = getattr(self.logger, 'log_path', None)
        if log_path:
            # 例如 logs/source_Art2Clipart_20260130_153424.log -> logs/source_Art2Clipart_20260130_153424/
            base = os.path.splitext(log_path)[0]
            return base
        else:
            # 兜底方案
            return os.path.join('logs', 'vis_output')
    
    def _get_classnames(self) -> List[str]:
        """获取类别名称（从配置或文件加载）"""
        classnames = self.config.get('data', {}).get('classnames', None)
        if classnames is not None:
            return classnames
        
        data_root = self.config.get('data', {}).get('root', 'data/officehome')
        classnames_file = os.path.join(data_root, 'classname.txt')
        
        if os.path.exists(classnames_file):
            with open(classnames_file, 'r') as f:
                classnames = [line.strip() for line in f if line.strip()]
            self.logger.info(f"从 {classnames_file} 加载了 {len(classnames)} 个类名")
            return classnames
        
        self.logger.warning("未找到类名文件，使用数字作为类名（CLIP 效果会很差）")
        return [str(i) for i in range(self.num_classes)]
    
    def _init_optimizers(self, train_cfg: Dict):
        """初始化优化器（与 ProDe 一致：分层学习率）"""
        param_group = []
        for name, param in self.netG.named_parameters():
            param_group.append({'params': param, 'lr': self.lr * 0.1})
        for name, param in self.netF.named_parameters():
            param_group.append({'params': param, 'lr': self.lr * 1.0})
        for name, param in self.netC.named_parameters():
            param_group.append({'params': param, 'lr': self.lr * 0.1})
        
        self.optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
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
    
    def _init_projector(self):
        """初始化 CLIP -> 源模型 特征空间投影层"""
        # CLIP ViT-B/32 输出 512 维，Bottleneck 输出 256 维
        clip_dim = 512  # ViT-B/32
        if 'ViT-L' in self.clip_arch:
            clip_dim = 768
        
        bottleneck_dim = self.config['model']['bottleneck_dim']
        
        self.projector = FeatureProjector(
            clip_dim=clip_dim,
            target_dim=bottleneck_dim,
            hidden_dim=self.projector_hidden_dim
        ).to(self.device)
        
        # 投影层优化器
        self.optimizer_proj = optim.Adam(
            self.projector.parameters(), lr=self.lr * 0.1, weight_decay=1e-4
        )
        
        # 缓存 CLIP 文本锚点（不变的，只需算一次）
        self.clip_model.eval()
        with torch.no_grad():
            self.text_anchors = self.clip_model.get_text_features()  # [n_cls, clip_dim]
        
        self.logger.info(f"投影层初始化完成: CLIP {clip_dim}D -> Bottleneck {bottleneck_dim}D")
        self.logger.info(f"文本锚点缓存: {self.text_anchors.shape}")
    
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
                
                self.logits_bank[indices] = logits.detach()
        
        self.logger.info(f"Logits Bank 初始化完成，形状: {self.logits_bank.shape}")
    
    def _knn_label_correction(self, features: torch.Tensor, consensus_mask: torch.Tensor,
                                consensus_labels: torch.Tensor) -> torch.Tensor:
        """
        KNN 标签修正: 用高可信样本的特征做最近邻，修正低可信样本的标签
        
        Args:
            features: 当前 batch 的 bottleneck 特征 [B, D]
            consensus_mask: 高可信样本掩码 [B] (bool)
            consensus_labels: 高可信样本的伪标签 [B]（低可信位置的值无意义）
        
        Returns:
            corrected_labels: 所有样本的修正标签 [B]
        """
        B = features.size(0)
        corrected_labels = consensus_labels.clone()
        
        # 如果高可信样本不足，无法做 KNN
        n_high = consensus_mask.sum().item()
        if n_high < self.knn_k or n_high == B:
            return corrected_labels
        
        # 高可信样本的特征和标签
        high_features = features[consensus_mask]   # [n_high, D]
        high_labels = consensus_labels[consensus_mask]  # [n_high]
        
        # 低可信样本的特征
        low_mask = ~consensus_mask
        low_features = features[low_mask]  # [n_low, D]
        
        # 计算余弦相似度 (低可信 vs 高可信)
        low_norm = F.normalize(low_features, dim=1)
        high_norm = F.normalize(high_features, dim=1)
        sim = torch.mm(low_norm, high_norm.t())  # [n_low, n_high]
        
        # 取 top-K 最近邻
        k = min(self.knn_k, n_high)
        _, topk_indices = sim.topk(k, dim=1)  # [n_low, K]
        
        # 多数投票
        topk_labels = high_labels[topk_indices]  # [n_low, K]
        
        # 对每个低可信样本投票
        knn_pred = torch.zeros(low_features.size(0), dtype=torch.long, device=features.device)
        for i in range(low_features.size(0)):
            labels_k = topk_labels[i]
            # bincount 投票
            counts = torch.bincount(labels_k, minlength=self.num_classes)
            knn_pred[i] = counts.argmax()
        
        corrected_labels[low_mask] = knn_pred
        return corrected_labels
    
    def _feature_mixup(self, high_features: torch.Tensor, high_labels: torch.Tensor,
                        low_features: torch.Tensor, low_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        特征层面 Mixup: 高可信种子 + 低可信样本混合
        标签权重偏向高可信样本
        
        Args:
            high_features: 高可信样本特征 [N_h, D]
            high_labels: 高可信样本标签 [N_h] (long)
            low_features: 低可信样本特征 [N_l, D]
            low_labels: 低可信样本标签 [N_l] (long, KNN修正后)
        
        Returns:
            mixed_features: 混合特征 [N_l, D]
            mixed_labels: 混合的软标签 [N_l, num_classes]
        """
        N_l = low_features.size(0)
        N_h = high_features.size(0)
        
        if N_h == 0 or N_l == 0:
            return low_features, F.one_hot(low_labels, self.num_classes).float()
        
        # 为每个低可信样本随机配对一个高可信样本
        pair_indices = torch.randint(0, N_h, (N_l,), device=low_features.device)
        paired_high_feat = high_features[pair_indices]
        paired_high_labels = high_labels[pair_indices]
        
        # Lambda ~ Beta(alpha, alpha)
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample(
            (N_l, 1)
        ).to(low_features.device)
        
        # 特征混合
        mixed_features = lam * paired_high_feat + (1 - lam) * low_features
        
        # 标签混合 (偏向高可信)
        lam_label = torch.clamp(lam.squeeze(1), min=self.mixup_label_bias)  # 至少 0.7 给高可信
        
        high_onehot = F.one_hot(paired_high_labels, self.num_classes).float()
        low_onehot = F.one_hot(low_labels, self.num_classes).float()
        mixed_labels = lam_label.unsqueeze(1) * high_onehot + (1 - lam_label.unsqueeze(1)) * low_onehot
        
        return mixed_features, mixed_labels
    
    def _anchor_align_loss(self, low_features: torch.Tensor, low_labels: torch.Tensor,
                            clip_image_features: torch.Tensor) -> torch.Tensor:
        """
        CLIP 文本锚点方向修正:
        让低可信样本的 CLIP 视觉特征向对应类的文本锚点靠拢
        
        通过投影层将 CLIP 特征映射到源模型空间后计算余弦对齐损失
        
        Args:
            low_features: 低可信样本的 bottleneck 特征 [N_l, D_src]（用于对齐方向参考）
            low_labels: 低可信样本的标签 [N_l]（KNN修正后）
            clip_image_features: 低可信样本的 CLIP 视觉特征 [N_l, D_clip]
        
        Returns:
            align_loss: 对齐损失标量
        """
        if low_features.size(0) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 投影 CLIP 视觉特征到源模型空间
        proj_clip_feat = self.projector(clip_image_features)  # [N_l, D_src]
        
        # 获取对应类别的文本锚点，也投影到源模型空间
        text_anchor_for_samples = self.text_anchors[low_labels]  # [N_l, D_clip]
        proj_text_anchor = self.projector(text_anchor_for_samples)  # [N_l, D_src]
        
        # 余弦相似度对齐: 让投影后的 CLIP 视觉特征和文本锚点一致
        proj_clip_norm = F.normalize(proj_clip_feat, dim=1)
        proj_anchor_norm = F.normalize(proj_text_anchor, dim=1)
        
        # 最大化余弦相似度 = 最小化 1 - cos_sim
        cos_sim = (proj_clip_norm * proj_anchor_norm).sum(dim=1)
        align_loss = (1 - cos_sim).mean()
        
        return align_loss
    
    def _collect_vis_data(self) -> Dict[str, np.ndarray]:
        """
        收集可视化所需的全量数据（遍历整个训练集）
        
        Returns:
            字典包含: features, gt_labels, pseudo_labels, max_probs,
                     src_pred, clip_pred, clip_noisy_pred
        """
        self.netG.eval()
        self.netF.eval()
        self.netC.eval()
        self.clip_model.eval()
        
        all_features = []
        all_gt_labels = []
        all_src_pred = []
        all_clip_pred = []
        all_clip_noisy_pred = []
        all_max_probs = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.train_loader, desc="Collecting vis data", leave=False):
                images = images.to(self.device)
                
                # 源模型
                feat = self.netG(images)
                feat = self.netF(feat)
                outputs = self.netC(feat)
                softmax_out = F.softmax(outputs, dim=1)
                
                src_pred = outputs.argmax(dim=1)
                max_probs = softmax_out.max(dim=1)[0]
                
                # CLIP 正常
                clip_logits = self.clip_model(images)
                if isinstance(clip_logits, tuple):
                    clip_logits = clip_logits[0]
                clip_pred = clip_logits.argmax(dim=1)
                
                # CLIP 扰动
                noise = torch.randn_like(images) * self.noise_std
                clip_logits_noisy = self.clip_model(images + noise)
                if isinstance(clip_logits_noisy, tuple):
                    clip_logits_noisy = clip_logits_noisy[0]
                clip_noisy_pred = clip_logits_noisy.argmax(dim=1)
                
                all_features.append(feat.cpu().numpy())
                all_gt_labels.append(labels.numpy())
                all_src_pred.append(src_pred.cpu().numpy())
                all_clip_pred.append(clip_pred.cpu().numpy())
                all_clip_noisy_pred.append(clip_noisy_pred.cpu().numpy())
                all_max_probs.append(max_probs.cpu().numpy())
        
        # 伪标签 = 用三方一致性 + KNN 修正（简化版，用 src_pred 作为 fallback）
        all_src_pred_cat = np.concatenate(all_src_pred)
        all_clip_pred_cat = np.concatenate(all_clip_pred)
        all_clip_noisy_pred_cat = np.concatenate(all_clip_noisy_pred)
        all_max_probs_cat = np.concatenate(all_max_probs)
        
        # 三方一致 → 伪标签
        consensus = (all_src_pred_cat == all_clip_pred_cat) & \
                    (all_clip_pred_cat == all_clip_noisy_pred_cat) & \
                    (all_max_probs_cat > self.consensus_conf_threshold)
        
        pseudo_labels = all_src_pred_cat.copy()  # 默认用源模型预测
        
        return {
            'features': np.concatenate(all_features),
            'gt_labels': np.concatenate(all_gt_labels),
            'pseudo_labels': pseudo_labels,
            'max_probs': all_max_probs_cat,
            'src_pred': all_src_pred_cat,
            'clip_pred': all_clip_pred_cat,
            'clip_noisy_pred': all_clip_noisy_pred_cat
        }
    
    def _maybe_visualize(self, epoch: int, is_best: bool = False):
        """
        按条件生成可视化图表
        
        保存时机: Epoch 1 / 每 vis_interval 个 epoch / best acc 更新时
        """
        should_vis = (epoch == 1) or (epoch % self.vis_interval == 0) or is_best
        if not should_vis:
            return
        
        # 确定保存目录
        if is_best:
            save_dir = os.path.join(self.vis_base_dir, 'best')
        else:
            save_dir = os.path.join(self.vis_base_dir, f'epoch{epoch}')
        
        self.logger.info(f"生成 Epoch {epoch} 可视化 ({'best' if is_best else 'periodic'})...")
        
        # 收集数据
        vis_data = self._collect_vis_data()
        
        # Epoch 1 时缓存特征用于坍塌图
        if epoch == 1:
            self.epoch1_features = vis_data['features'].copy()
        
        # 生成所有图表
        generate_all_visualizations(
            features=vis_data['features'],
            gt_labels=vis_data['gt_labels'],
            pseudo_labels=vis_data['pseudo_labels'],
            max_probs=vis_data['max_probs'],
            src_pred=vis_data['src_pred'],
            clip_pred=vis_data['clip_pred'],
            clip_noisy_pred=vis_data['clip_noisy_pred'],
            save_dir=save_dir,
            num_classes=self.num_classes,
            current_epoch=epoch,
            features_epoch1=self.epoch1_features,
            logger=self.logger
        )
    
    def train(self) -> None:
        """执行目标域适应训练"""
        self.logger.info(f"开始 SCLM 目标域适应，共 {self.max_epoch} 个 epoch")
        self.logger.info(f"超参数: gent_par={self.gent_par}, iic_par={self.iic_par}, "
                        f"mixup_weight={self.mixup_weight}, anchor_weight={self.anchor_weight}")
        self.logger.info(f"三方一致性: conf_threshold={self.consensus_conf_threshold}, "
                        f"noise_std={self.noise_std}, knn_k={self.knn_k}")
        
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
            mixup_meter = AverageMeter("Mixup")
            anchor_meter = AverageMeter("Anchor")
            consensus_meter = AverageMeter("Consensus%")
            
            self.netG.train()
            self.netF.train()
            self.netC.train()
            self.projector.train()
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.max_epoch}", leave=False)
            
            for batch_idx, (images, labels, indices) in enumerate(pbar):
                if images.size(0) <= 1:
                    continue
                
                images = images.to(self.device)
                iter_num += 1
                
                # 更新学习率
                lr_scheduler(self.optimizer, iter_num, max_iter)
                
                # ==================== 1. 源模型前向传播 ====================
                feat = self.netG(images)
                feat = self.netF(feat)           # bottleneck 特征 [B, 256]
                outputs = self.netC(feat)          # logits [B, C]
                softmax_out = F.softmax(outputs, dim=1)
                
                src_pred = outputs.argmax(dim=1)
                src_conf = softmax_out.max(dim=1)[0]
                
                # ==================== 2. CLIP 预测（正常 + 扰动）====================
                self.clip_model.eval()
                with torch.no_grad():
                    # CLIP 正常预测
                    clip_logits = self.clip_model(images)
                    if isinstance(clip_logits, tuple):
                        clip_logits = clip_logits[0]
                    clip_score_sm = F.softmax(clip_logits, dim=1)
                    clip_pred = clip_logits.argmax(dim=1)
                    
                    # CLIP 高斯扰动预测
                    noise = torch.randn_like(images) * self.noise_std
                    clip_logits_noisy = self.clip_model(images + noise)
                    if isinstance(clip_logits_noisy, tuple):
                        clip_logits_noisy = clip_logits_noisy[0]
                    clip_noisy_pred = clip_logits_noisy.argmax(dim=1)
                    
                    # 获取 CLIP 视觉特征（用于锚点修正）
                    clip_image_feat = self.clip_model.get_image_features(images)  # [B, 512]
                
                # ==================== 3. 三方一致性检验 ====================
                consensus_mask = (
                    (src_pred == clip_pred) & 
                    (clip_pred == clip_noisy_pred) & 
                    (src_conf > self.consensus_conf_threshold)
                )
                
                consensus_ratio = consensus_mask.float().mean().item()
                consensus_meter.update(consensus_ratio * 100, images.size(0))
                
                # 高可信样本的伪标签 = 三方一致的预测
                consensus_labels = src_pred.clone()
                
                # ==================== 4. KNN 标签修正 ====================
                with torch.no_grad():
                    corrected_labels = self._knn_label_correction(
                        feat.detach(), consensus_mask, consensus_labels
                    )
                
                # ==================== 5. Loss 计算 ====================
                
                # 5a. IIC Loss: 对齐源模型和 CLIP 输出（保留）
                iic_loss = IID_loss(softmax_out, clip_score_sm)
                
                # 5b. 多样性损失（保留）
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + self.epsilon))
                
                # 5c. CE Loss: 用三方一致的伪标签（只对高可信样本）
                if consensus_mask.sum() > 0:
                    ce_loss = F.cross_entropy(outputs[consensus_mask], corrected_labels[consensus_mask])
                else:
                    ce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # 5d. Mixup CE Loss
                high_mask = consensus_mask
                low_mask = ~consensus_mask
                
                if high_mask.sum() > 0 and low_mask.sum() > 0:
                    mixed_feat, mixed_labels = self._feature_mixup(
                        feat[high_mask].detach(), corrected_labels[high_mask],
                        feat[low_mask].detach(), corrected_labels[low_mask]
                    )
                    # 用混合特征过分类器
                    mixed_logits = self.netC(mixed_feat)
                    # 软标签 CE: -sum(y * log_softmax(logits))
                    mixup_ce_loss = -torch.sum(
                        mixed_labels * F.log_softmax(mixed_logits, dim=1), dim=1
                    ).mean()
                else:
                    mixup_ce_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # 5e. CLIP 文本锚点对齐 Loss（只对低可信样本）
                if low_mask.sum() > 0:
                    anchor_loss = self._anchor_align_loss(
                        feat[low_mask].detach(),
                        corrected_labels[low_mask],
                        clip_image_feat[low_mask]
                    )
                else:
                    anchor_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # 5f. 总 Loss
                total_loss = (
                    self.iic_par * iic_loss
                    + self.gent_par * ce_loss
                    - self.div_weight * gentropy_loss
                    + self.mixup_weight * mixup_ce_loss
                    + self.anchor_weight * anchor_loss
                )
                
                # ==================== 6. 反向传播 ====================
                self.optimizer.zero_grad()
                self.optimizer_proj.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.optimizer_proj.step()
                
                # 更新 meters
                loss_meter.update(total_loss.item(), images.size(0))
                iic_meter.update(iic_loss.item(), images.size(0))
                ce_meter.update(ce_loss.item(), images.size(0))
                div_meter.update(gentropy_loss.item(), images.size(0))
                mixup_meter.update(mixup_ce_loss.item(), images.size(0))
                anchor_meter.update(anchor_loss.item(), images.size(0))
                
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'cns%': f'{consensus_meter.avg:.1f}'
                })
            
            # Epoch 结束评估
            acc = self._evaluate()
            
            is_best = acc > best_acc
            
            self.logger.info(
                f"Epoch [{epoch}/{self.max_epoch}] "
                f"Loss: {loss_meter.avg:.4f} "
                f"(IIC: {iic_meter.avg:.4f}, CE: {ce_meter.avg:.4f}, "
                f"Div: {div_meter.avg:.4f}, Mixup: {mixup_meter.avg:.4f}, "
                f"Anchor: {anchor_meter.avg:.4f}) | "
                f"Consensus: {consensus_meter.avg:.1f}% | "
                f"Target Acc: {acc:.2f}%"
            )
            
            # 可视化
            self._maybe_visualize(epoch, is_best=is_best)
            
            # 保存最佳模型
            if is_best:
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
            'projector_state_dict': self.projector.state_dict(),
            'prompt_learner_state_dict': self.clip_model.prompt_learner.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'acc': acc,
            'config': self.config
        }
        
        if is_best:
            save_path = os.path.join(self.save_dir, f"target_{source}2{target}_best.pth")
            prompt_path = os.path.join(self.save_dir, f"prompt_{source}2{target}_best.pt")
            torch.save(self.clip_model.prompt_learner.state_dict(), prompt_path)
            self.logger.info(f"Prompt 权重已保存: {prompt_path}")
        else:
            save_path = os.path.join(self.save_dir, f"target_{source}2{target}_epoch{epoch}.pth")
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint 已保存: {save_path}")
