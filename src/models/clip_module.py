"""
CLIP 模块
包含 Prompt Learning 和 Test-Time Adaptation 相关组件
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

import open_clip

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TextEncoder(nn.Module):
    """CLIP 文本编码器包装"""
    
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.register_buffer('attn_mask', clip_model.attn_mask)
        # open_clip 没有 dtype 属性，从权重推断
        self.dtype = clip_model.ln_final.weight.dtype
    
    def forward(self, prompts: torch.Tensor, tokenized_prompts: torch.Tensor) -> torch.Tensor:
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # 取 EOT token 的特征
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    """
    可学习的 Prompt 模块（CoOp 风格）
    
    将 "a photo of a [CLASS]" 中的 "a photo of a" 替换为可学习的 token
    """
    
    def __init__(
        self,
        clip_model,
        classnames: List[str],
        n_ctx: int = 4,
        ctx_init: str = "a_photo_of_a",
        device: torch.device = None
    ):
        super().__init__()
        
        n_cls = len(classnames)
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        
        # open_clip 没有 dtype 属性，从权重推断
        dtype = clip_model.ln_final.weight.dtype
        self.dtype = dtype
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        
        # 初始化 context vectors
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            self.n_ctx = n_ctx
            
            # 用 CLIP tokenizer 编码初始 prompt
            prompt = open_clip.tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix
        
        # 保存初始状态用于 reset
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors)  # 可学习参数
        
        # 处理类别名
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        # Tokenize
        tokenized_prompts = open_clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # 保存 token embedding 的前缀和后缀
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS + EOS
        
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames
        self.clip_model = clip_model
    
    def reset(self):
        """重置 prompt 到初始状态"""
        self.ctx.data.copy_(self.ctx_init_state)
    
    def forward(self) -> torch.Tensor:
        """
        生成完整的 prompt embeddings
        
        Returns:
            prompts: [n_cls, n_tokens, ctx_dim]
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix  # [n_cls, 1, dim]
        suffix = self.token_suffix  # [n_cls, *, dim]
        
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts


class ClipTestTimeTuning(nn.Module):
    """
    CLIP Test-Time Tuning 模块
    
    在测试时微调 prompt，使 CLIP 输出与源模型对齐
    """
    
    def __init__(
        self,
        classnames: List[str],
        arch: str = "ViT-B-32",
        n_ctx: int = 4,
        ctx_init: str = "a_photo_of_a",
        device: torch.device = None
    ):
        super().__init__()
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载 CLIP 模型
        vlm_cache_dir = os.path.join(PROJECT_ROOT, 'vlm')
        os.makedirs(vlm_cache_dir, exist_ok=True)
        
        clip_model, _, _ = open_clip.create_model_and_transforms(
            arch,
            pretrained='openai',
            cache_dir=vlm_cache_dir
        )
        clip_model = clip_model.to(self.device)
        
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale.data
        
        # Prompt learner
        self.prompt_learner = PromptLearner(
            clip_model, classnames, n_ctx, ctx_init, self.device
        )
        
        # 冻结除 prompt_learner 之外的所有参数
        for name, param in self.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
    
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
    
    def reset(self):
        """重置 prompt 到初始状态"""
        self.prompt_learner.reset()
    
    def get_text_features(self) -> torch.Tensor:
        """获取文本特征"""
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            images: 图像张量 [B, C, H, W] (ImageNet Normalized)
        Returns:
            logits: CLIP Logits [B, n_cls]
        """
        # 1. 反归一化 (ImageNet) -> [0, 1]
        mean_in = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std_in = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        x = images * std_in + mean_in
        
        # 2. 重新归一化 (CLIP)
        mean_out = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
        std_out = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
        x = (x - mean_out) / std_out
        
        # 3. CLIP Inference
        with torch.no_grad():
            image_features = self.image_encoder(x.type(self.dtype))
        
        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        return logits


def test_time_tuning(
    model: ClipTestTimeTuning,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    target_output: torch.Tensor,
    tta_steps: int = 1
) -> torch.Tensor:
    """
    Test-Time Tuning
    
    用 IIC loss 微调 prompt，使 CLIP 输出与目标输出对齐
    
    Args:
        model: ClipTestTimeTuning 模型
        images: 输入图像
        optimizer: prompt 优化器
        target_output: 目标输出（源模型的 logits）
        tta_steps: TTA 步数
        
    Returns:
        output: TTA 后的 CLIP logits
    """
    from ..utils.losses import IID_loss
    
    # 将目标输出转为 softmax
    target_output = target_output.to(images.device)
    target_softmax = F.softmax(target_output, dim=1)
    
    model.train()
    for _ in range(tta_steps):
        output_logits = model(images)
        output = F.softmax(output_logits, dim=1)
        
        iic_loss = IID_loss(output, target_softmax)
        
        optimizer.zero_grad()
        iic_loss.backward()
        optimizer.step()
    
    # TTA 完成后，切换到 eval 模式，返回最终 logits
    model.eval()
    with torch.no_grad():
        output_logits = model(images)
    
    return output_logits
