"""
配置文件加载和管理模块
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并两个配置字典，override会覆盖base中的同名项
    
    Args:
        base: 基础配置
        override: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    
    return result


def update_config_from_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """
    从命令行参数更新配置
    
    Args:
        config: 原始配置
        args: argparse解析后的参数
        
    Returns:
        更新后的配置
    """
    # 更新源域和目标域
    if hasattr(args, 'source') and args.source:
        config['data']['source'] = args.source
    if hasattr(args, 'target') and args.target:
        config['data']['target'] = args.target
    
    # 更新GPU
    if hasattr(args, 'gpu') and args.gpu is not None:
        config['device']['gpu_id'] = args.gpu
    
    # 更新随机种子
    if hasattr(args, 'seed') and args.seed is not None:
        config['seed'] = args.seed
    
    return config


def config_to_str(config: Dict[str, Any], indent: int = 0) -> str:
    """
    将配置字典转换为可读字符串
    
    Args:
        config: 配置字典
        indent: 缩进级别
        
    Returns:
        格式化的配置字符串
    """
    lines = []
    prefix = "  " * indent
    
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(config_to_str(value, indent + 1))
        else:
            lines.append(f"{prefix}{key}: {value}")
    
    return "\n".join(lines)
