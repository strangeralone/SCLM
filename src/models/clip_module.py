"""
CLIP 模块 (Adapter for local clip library)
适配本地 clip/custom_clip.py，并修正归一化问题。
"""
import torch
import torch.nn as nn
from typing import List, Optional

# Import from local clip library
import clip.custom_clip as custom_clip
from src.utils.losses import IID_loss

class ClipTestTimeTuning(nn.Module):
    """
    Wrapper around clip.custom_clip.ClipTestTimeTuning
    Adds normalization correction (ImageNet -> CLIP).
    """
    def __init__(
        self,
        device: torch.device,
        classnames: List[str],
        batch_size: int = None,
        arch: str = "ViT-B/32",
        n_ctx: int = 16,
        ctx_init: str = None,
        ctx_position: str = 'end',
        learned_cls: bool = False
    ):
        super().__init__()
        
        # Initialize internal CoOp model from local clip library
        self.model = custom_clip.ClipTestTimeTuning(
            device=device,
            classnames=classnames,
            batch_size=batch_size,
            criterion='cosine', # Default in custom_clip
            arch=arch,
            n_ctx=n_ctx,
            ctx_init=ctx_init,
            ctx_position=ctx_position,
            learned_cls=learned_cls
        )
        
        # Freeze backbone weights explicitly (ImageEncoder & TextEncoder)
        # custom_clip.ClipTestTimeTuning uses self.image_encoder and self.text_encoder
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.model.text_encoder.parameters():
            param.requires_grad = False
            
        self.dtype = self.model.dtype

    @property
    def prompt_learner(self):
        # Expose prompt_learner for external access/optimization
        return self.model.prompt_learner

    def _convert_normalization(self, images: torch.Tensor) -> torch.Tensor:
        """ImageNet 归一化 -> CLIP 归一化"""
        mean_in = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std_in = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        mean_out = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
        std_out = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
        x = images * std_in + mean_in
        x = (x - mean_out) / std_out
        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, C, H, W] (ImageNet Normalized)
        Returns:
            logits: [B, n_cls]
        """
        x = self._convert_normalization(images)
        logits, _ = self.model.inference(x)
        return logits
    
    def get_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        获取 CLIP 视觉特征（L2归一化后）
        
        Args:
            images: [B, C, H, W] (ImageNet Normalized)
        Returns:
            image_features: [B, clip_dim] (已L2归一化)
        """
        x = self._convert_normalization(images)
        with torch.no_grad():
            image_features = self.model.image_encoder(x.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.float()
    
    def get_text_features(self) -> torch.Tensor:
        """
        获取每个类的 CLIP 文本特征作为锚点（L2归一化后）
        
        Returns:
            text_features: [n_cls, clip_dim] (已L2归一化)
        """
        with torch.no_grad():
            text_features = self.model.get_text_features()
        return text_features.float()

# Re-export PromptLearner and TextEncoder for compatibility if needed elsewhere
# But mostly ClipTestTimeTuning is checked.

def test_time_tuning(
    model: ClipTestTimeTuning,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    target_output: torch.Tensor,
    tta_steps: int = 1
) -> torch.Tensor:
    """
    Test-Time Tuning for CLIP Prompt
    Args:
        model: ClipTestTimeTuning instance
        images: Input images [B, C, H, W]
        optimizer: Optimizer for prompt learner
        target_output: Source model output logits [B, C]
        tta_steps: Number of TTA steps
    Returns:
        final_logits: [B, C]
    """
    # Source model probability
    target_prob = torch.softmax(target_output, dim=1)
    
    # TTA Loop
    for _ in range(tta_steps):
        # Forward pass (Normalization is handled inside model.forward)
        logits, _ = model(images)
        prob = torch.softmax(logits, dim=1)
        
        # Loss: IIC / Mutual Information Maximization with Source
        # Note: IID_loss input order (output, target). 
        # Minimizing IID loss = Maximizing Mutual Information
        loss = IID_loss(prob, target_prob)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Final forward to get updated logits
    with torch.no_grad():
        model.eval()
        final_logits, _ = model(images)
        
    return final_logits
    