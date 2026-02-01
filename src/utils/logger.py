"""
日志管理模块
同时输出到控制台和文件，支持配置信息记录
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def setup_logger(
    name: str,
    log_dir: str = "logs",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    设置日志器，同时输出到控制台和文件
    
    Args:
        name: 日志器名称
        log_dir: 日志目录
        log_file: 日志文件名（默认自动生成）
        level: 日志级别
        
    Returns:
        配置好的日志器
    """
    # 创建日志目录
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"
    
    log_path = log_dir / log_file
    
    # 创建日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已有的handlers（防止重复）
    logger.handlers.clear()
    
    # 日志格式
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 文件Handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 保存日志文件路径到logger
    logger.log_path = str(log_path)
    
    return logger


def log_config(logger: logging.Logger, config: Dict[str, Any], title: str = "Configuration") -> None:
    """
    记录配置信息到日志
    
    Args:
        logger: 日志器
        config: 配置字典
        title: 标题
    """
    logger.info("=" * 60)
    logger.info(f" {title}")
    logger.info("=" * 60)
    _log_dict(logger, config, indent=0)
    logger.info("=" * 60)


def _log_dict(logger: logging.Logger, d: Dict[str, Any], indent: int = 0) -> None:
    """
    递归记录字典内容
    
    Args:
        logger: 日志器
        d: 字典
        indent: 缩进级别
    """
    prefix = "  " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            logger.info(f"{prefix}{key}:")
            _log_dict(logger, value, indent + 1)
        else:
            logger.info(f"{prefix}{key}: {value}")


class AverageMeter:
    """
    计算和存储平均值和当前值
    用于追踪训练过程中的loss、accuracy等指标
    """
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"
