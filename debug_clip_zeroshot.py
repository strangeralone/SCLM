
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from src.models.clip_module import ClipTestTimeTuning
import open_clip

# 配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 加载类名
classnames_file = "data/officehome/classname.txt"
with open(classnames_file, "r") as f:
    classnames = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(classnames)} classes from {classnames_file}")

# 2. 初始化 CLIP 模块
print("Initializing CLIP module...")
clip_module = ClipTestTimeTuning(
    classnames=classnames,
    arch="ViT-B/32",
    n_ctx=4,
    ctx_init="a_photo_of_a",
    device=device
)
clip_module.eval()
print("CLIP module initialized.")

# 3. 准备测试图像
# 找一张存在的图片
img_path = "data/officehome/Art/Alarm_Clock/00001.jpg"
if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    # 尝试找其他图片
    for root, dirs, files in os.walk("data/officehome"):
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                break
        if img_path: break

print(f"Testing with image: {img_path}")
image = Image.open(img_path).convert("RGB")

# 4. 预处理 (使用与 Training 相同的 ImageNet Normalize)
preprocess_imagenet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. 预处理 (使用 open_clip 推荐的 Normalize)
_, _, preprocess_clip = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
# 注意：preprocess_clip 通常包含 Resize, CenterCrop, ToTensor, Normalize

print("Running inference...")

# 测试 ImageNet Normalize
input_tensor = preprocess_imagenet(image).unsqueeze(0).to(device)
with torch.no_grad():
    logits = clip_module(input_tensor)
    probs = F.softmax(logits, dim=1)
    top_vals, top_idxs = torch.topk(probs, 5)

print("\n--- Result with ImageNet Normalize (Used in Training) ---")
for i in range(5):
    idx = top_idxs[0, i].item()
    score = top_vals[0, i].item()
    print(f"{i+1}. {classnames[idx]}: {score:.4f}")

# 测试 open_clip Normalize
# 注意：ClipTestTimeTuning 内部只接受 Tensor，所以我们需要手动处理
# 获取 open_clip 的 Normalize 参数
# 通常在 preprocess_clip.transforms[-1]
print("\n--- OpenAI CLIP Preprocess Info ---")
print(preprocess_clip)

# 手动构建符合 open_clip 的 tensor
input_tensor_clip = preprocess_clip(image).unsqueeze(0).to(device)

with torch.no_grad():
    logits_clip = clip_module(input_tensor_clip)
    probs_clip = F.softmax(logits_clip, dim=1)
    top_vals_clip, top_idxs_clip = torch.topk(probs_clip, 5)

print("\n--- Result with open_clip Preprocess ---")
for i in range(5):
    idx = top_idxs_clip[0, i].item()
    score = top_vals_clip[0, i].item()
    print(f"{i+1}. {classnames[idx]}: {score:.4f}")

# 6. 对比 logits_scale
print(f"\nLogit Scale: {clip_module.logit_scale.exp().item():.4f}")
