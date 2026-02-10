#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFDA 扰动一致性实验框架
入口文件

Usage:
    # 使用配置文件运行（推荐）
    python main.py --mode source -c configs/art2clipart.yaml
    
    # 目标域适应
    python main.py --mode target -c configs/art2clipart.yaml
    
    # 使用默认配置
    python main.py --mode source
"""
import os
import sys
import argparse
import random
import numpy as np
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config, merge_config, update_config_from_args
from src.utils.logger import setup_logger, log_config
from src.data.office_home import get_loader
from src.models.network import create_split_models, FeatureAdapter
from src.trainers.source_trainer import SourceTrainer
from src.trainers.target_trainer import TargetTrainer
from src.trainers.adapter_trainer import AdapterTrainer


def set_random_seed(seed: int) -> None:
    """设置随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config: dict) -> torch.device:
    """获取计算设备"""
    device_type = config['device']['type'].lower()
    
    if device_type == "cuda":
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{config['device']['gpu_id']}")
        else:
            print("警告: CUDA不可用，回退到CPU")
            device = torch.device("cpu")
    elif device_type == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("警告: MPS不可用，回退到CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    
    return device


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SFDA 扰动一致性实验框架",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 必需参数
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["source", "target", "adapter", "test"],
        help="运行模式: source(源域预训练), target(目标域适应), adapter(Adapter适应), test(测试)"
    )
    
    # 可选参数
    parser.add_argument(
        "-c", "--config", type=str, default=None,
        help="实验配置文件路径，会与默认配置合并"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="源域名称 (Art, Clipart, Product, RealWorld)"
    )
    parser.add_argument(
        "--target", type=str, default=None,
        help="目标域名称 (Art, Clipart, Product, RealWorld)"
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="GPU ID"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="恢复训练的checkpoint路径"
    )
    
    return parser.parse_args()


def run_source_training(config: dict, logger, device: torch.device) -> None:
    """执行源域预训练"""
    source = config['data']['source']
    logger.info(f"源域预训练: {source}")
    
    # 创建数据加载器
    train_loader = get_loader(
        root=config['data']['root'],
        domain=source,
        batch_size=config['train']['source']['batch_size'],
        train=True,
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size']
    )
    
    # 使用源域数据（无增强）作为测试
    test_loader = get_loader(
        root=config['data']['root'],
        domain=source,
        batch_size=config['train']['source']['batch_size'],
        train=False,  # 不使用数据增强
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size']
    )
    
    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    netG, netF, netC = create_split_models(config)
    netG = netG.to(device)
    netF = netF.to(device)
    netC = netC.to(device)
    logger.info(f"模型创建完成，设备: {device}")
    
    # 创建训练器
    trainer = SourceTrainer(
        netG=netG,
        netF=netF,
        netC=netC,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        logger=logger,
        device=device
    )
    
    # 开始训练
    trainer.train()


def run_target_adaptation(config: dict, logger, device: torch.device, resume: str = None) -> None:
    """执行目标域适应"""
    source = config['data']['source']
    target = config['data']['target']
    logger.info(f"目标域适应: {source} -> {target}")
    
    # 创建数据加载器
    train_loader = get_loader(
        root=config['data']['root'],
        domain=target,
        batch_size=config['train']['target']['batch_size'],
        train=True,
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size']
    )
    
    test_loader = get_loader(
        root=config['data']['root'],
        domain=target,
        batch_size=config['train']['target']['batch_size'],
        train=False,
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size']
    )
    
    logger.info(f"目标域训练集大小: {len(train_loader.dataset)}")
    logger.info(f"目标域测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    netG, netF, netC = create_split_models(config)
    
    # 加载checkpoints权重
    if resume:
        checkpoint_path = resume

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # 兼容旧版本checkpoint (如果之前的key是 backbone., etc，现在可能是 netG_state_dict)
        # 假设新的SourceTrainer保存的是 netG_state_dict, netF_state_dict, netC_state_dict
        if 'netG_state_dict' in checkpoint:
            netG.load_state_dict(checkpoint['netG_state_dict'])
            netF.load_state_dict(checkpoint['netF_state_dict'])
            netC.load_state_dict(checkpoint['netC_state_dict'])
            logger.info(f"已加载源域预训练权重 (New Format): {checkpoint_path}")
        else:
             logger.warning(f"无法识别checkpoint格式: {checkpoint_path}")
             
    else:
        checkpoint_path = config['checkpoint']['save_dir']
        source_domain = config['data']['source']
        netG.load_state_dict(torch.load(os.path.join(checkpoint_path, f'source_{source_domain}_G.pt')))
        netF.load_state_dict(torch.load(os.path.join(checkpoint_path, f'source_{source_domain}_F.pt')))
        netC.load_state_dict(torch.load(os.path.join(checkpoint_path, f'source_{source_domain}_C.pt')))
    
    netG = netG.to(device)
    netF = netF.to(device)
    netC = netC.to(device)
    
    # 创建训练器
    trainer = TargetTrainer(
        netG=netG,
        netF=netF,
        netC=netC,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        logger=logger,
        device=device
    )
    
    # 开始适应
    trainer.train()


def run_test(config: dict, logger, device: torch.device, checkpoint_path: str = None) -> None:
    """测试模型"""
    target = config['data']['target']
    logger.info(f"测试目标域: {target}")
    
    # 创建数据加载器
    test_loader = get_loader(
        root=config['data']['root'],
        domain=target,
        batch_size=config['train']['target']['batch_size'],
        train=False,
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size']
    )
    
    # 创建模型
    netG, netF, netC = create_split_models(config)
    
    # 加载权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'netG_state_dict' in checkpoint:
            netG.load_state_dict(checkpoint['netG_state_dict'])
            netF.load_state_dict(checkpoint['netF_state_dict'])
            netC.load_state_dict(checkpoint['netC_state_dict'])
        elif 'model_state_dict' in checkpoint:
             state_dict = checkpoint['model_state_dict']
             netG.load_state_dict({k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')})
             netF.load_state_dict({k.replace('bottleneck.', ''): v for k, v in state_dict.items() if k.startswith('bottleneck.')})
             netC.load_state_dict({k.replace('classifier.', ''): v for k, v in state_dict.items() if k.startswith('classifier.')})
        
        logger.info(f"已加载模型权重: {checkpoint_path}")
    
    netG = netG.to(device)
    netF = netF.to(device)
    netC = netC.to(device)
    
    netG.eval()
    netF.eval()
    netC.eval()
    
    # 测试
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            f = netG(images)
            f = netF(f)
            logits = netC(f)
            
            pred = logits.argmax(dim=1)
            
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    acc = 100.0 * correct / total
    logger.info(f"目标域 {target} 准确率: {acc:.2f}%")


def run_adapter_training(config: dict, logger, device: torch.device, resume: str = None) -> None:
    """执行 Adapter-based 目标域适应"""
    source = config['data']['source']
    target = config['data']['target']
    logger.info(f"Adapter 目标域适应: {source} -> {target}")
    
    # 创建数据加载器
    train_loader = get_loader(
        root=config['data']['root'],
        domain=target,
        batch_size=config['train'].get('adapter', {}).get('batch_size', 32),
        train=True,
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size']
    )
    
    test_loader = get_loader(
        root=config['data']['root'],
        domain=target,
        batch_size=config['train'].get('adapter', {}).get('batch_size', 32),
        train=False,
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size']
    )
    
    logger.info(f"目标域训练集大小: {len(train_loader.dataset)}")
    logger.info(f"目标域测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    netG, netF, netC = create_split_models(config)
    
    # 加载源域预训练权重
    if resume:
        checkpoint_path = resume
    else:
        checkpoint_path = os.path.join(
            config['checkpoint']['save_dir'],
            f"source_{source}_best.pth"
        )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'netG_state_dict' in checkpoint:
            netG.load_state_dict(checkpoint['netG_state_dict'])
            netF.load_state_dict(checkpoint['netF_state_dict'])
            netC.load_state_dict(checkpoint['netC_state_dict'])
            logger.info(f"已加载源域预训练权重 (New Format): {checkpoint_path}")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            netG.load_state_dict({k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')})
            netF.load_state_dict({k.replace('bottleneck.', ''): v for k, v in state_dict.items() if k.startswith('bottleneck.')})
            netC.load_state_dict({k.replace('classifier.', ''): v for k, v in state_dict.items() if k.startswith('classifier.')})
            logger.info(f"已加载源域预训练权重 (Old Format): {checkpoint_path}")
        else:
            logger.warning(f"无法识别checkpoint格式: {checkpoint_path}")
    else:
        logger.warning(f"未找到源域预训练权重: {checkpoint_path}，使用随机初始化")
    
    netG = netG.to(device)
    netF = netF.to(device)
    netC = netC.to(device)
    
    # 创建 Adapter
    adapter_cfg = config['train'].get('adapter', {})
    adapter = FeatureAdapter(
        in_features=config['model']['bottleneck_dim'],
        hidden_dim=adapter_cfg.get('hidden_dim', 512),
        num_layers=adapter_cfg.get('num_layers', 2),
        dropout=adapter_cfg.get('dropout', 0.1)
    )
    adapter = adapter.to(device)
    logger.info(f"Adapter 创建完成: hidden_dim={adapter_cfg.get('hidden_dim', 512)}, num_layers={adapter_cfg.get('num_layers', 2)}")
    
    # 创建训练器
    trainer = AdapterTrainer(
        netG=netG,
        netF=netF,
        netC=netC,
        adapter=adapter,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        logger=logger,
        device=device
    )
    
    # 开始训练
    trainer.train()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 加载配置：先加载默认配置，再合并自定义配置
    default_config = load_config("configs/default.yaml")
    if args.config:
        custom_config = load_config(args.config)
        config = merge_config(default_config, custom_config)
    else:
        config = default_config
    config = update_config_from_args(config, args)
    
    # 设置随机种子
    set_random_seed(config['seed'])
    
    # 获取设备
    device = get_device(config)
    
    # 设置日志
    source = config['data']['source']
    target = config['data']['target']
    log_name = f"{args.mode}_{source}2{target}"
    logger = setup_logger(log_name, log_dir=config['logging']['log_dir'])
    
    # 记录配置信息（消融实验基础）
    log_config(logger, config, title="Experiment Configuration")
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"计算设备: {device}")
    
    # 执行对应模式
    if args.mode == "source":
        run_source_training(config, logger, device)
    elif args.mode == "target":
        run_target_adaptation(config, logger, device, args.resume)
    elif args.mode == "adapter":
        run_adapter_training(config, logger, device, args.resume)
    elif args.mode == "test":
        run_test(config, logger, device, args.resume)
    else:
        logger.error(f"未知模式: {args.mode}")
        sys.exit(1)
    
    logger.info("实验完成！")


if __name__ == "__main__":
    main()
