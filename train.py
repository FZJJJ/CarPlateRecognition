# -*- coding: utf-8 -*-
"""
模型训练脚本
用于训练YOLO11车牌检测模型
"""

import os
import sys
import argparse
from pathlib import Path

from ultralytics import YOLO
from loguru import logger

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config import TRAIN_CONFIG, PATHS, YOLO_CONFIG
from utils.data_loader import DatasetLoader


def setup_logging():
    """设置日志"""
    try:
        # 确保日志目录存在
        PATHS['logs'].mkdir(parents=True, exist_ok=True)
        
        # 配置loguru
        logger.remove()  # 移除默认处理器
        
        # 添加控制台输出
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # 添加文件输出
        logger.add(
            PATHS['training_log'],
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
        
        logger.info("训练日志系统初始化完成")
        
    except Exception as e:
        print(f"日志设置失败: {str(e)}")


def validate_dataset(dataset_path: str) -> bool:
    """验证数据集
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        是否验证通过
    """
    try:
        logger.info(f"验证数据集: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        
        # 检查数据集配置文件
        config_file = dataset_path / "dataset.yaml"
        if not config_file.exists():
            logger.error(f"数据集配置文件不存在: {config_file}")
            return False
        
        # 检查必要目录
        required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                logger.error(f"必要目录不存在: {dir_path}")
                return False
        
        # 使用数据加载器验证
        loader = DatasetLoader(str(dataset_path))
        
        # 检查训练集
        train_images = loader.get_image_paths('train')
        train_labels = loader.get_label_paths('train')
        
        if not train_images:
            logger.error("训练集图像为空")
            return False
        
        if len(train_images) != len(train_labels):
            logger.warning(f"训练集图像和标签数量不匹配: {len(train_images)} vs {len(train_labels)}")
        
        # 检查验证集
        val_images = loader.get_image_paths('val')
        val_labels = loader.get_label_paths('val')
        
        if not val_images:
            logger.warning("验证集图像为空")
        
        logger.info(f"数据集验证通过:")
        logger.info(f"  训练集: {len(train_images)} 图像, {len(train_labels)} 标签")
        logger.info(f"  验证集: {len(val_images)} 图像, {len(val_labels)} 标签")
        logger.info(f"  类别数: {loader.num_classes}")
        logger.info(f"  类别名: {loader.class_names}")
        
        return True
        
    except Exception as e:
        logger.error(f"数据集验证失败: {str(e)}")
        return False


def train_model(dataset_path: str, 
                model_path: str = None,
                epochs: int = None,
                batch_size: int = None,
                device: str = None,
                resume: bool = False) -> bool:
    """训练模型
    
    Args:
        dataset_path: 数据集路径
        model_path: 预训练模型路径
        epochs: 训练轮数
        batch_size: 批次大小
        device: 训练设备
        resume: 是否恢复训练
        
    Returns:
        是否训练成功
    """
    try:
        logger.info("开始模型训练...")
        
        # 设置参数
        model_path = model_path or str(PATHS['yolo_pretrained'])
        epochs = epochs or TRAIN_CONFIG['epochs']
        batch_size = batch_size or TRAIN_CONFIG['batch_size']
        device = device or TRAIN_CONFIG['device']
        
        # 检查预训练模型
        if not Path(model_path).exists():
            logger.warning(f"预训练模型不存在: {model_path}，将下载默认模型")
            model_path = 'yolo11n.pt'
        
        # 加载模型
        logger.info(f"加载模型: {model_path}")
        model = YOLO(model_path)
        
        # 训练参数
        train_args = {
            'data': str(Path(dataset_path) / 'dataset.yaml'),
            'epochs': epochs,
            'batch': batch_size,
            'device': device,
            'project': TRAIN_CONFIG['project'],
            'name': TRAIN_CONFIG['name'],
            'save_period': TRAIN_CONFIG['save_period'],
            'patience': TRAIN_CONFIG['patience'],
            'workers': TRAIN_CONFIG['workers'],
            'lr0': TRAIN_CONFIG['learning_rate'],
            'momentum': TRAIN_CONFIG['momentum'],
            'weight_decay': TRAIN_CONFIG['weight_decay'],
            'warmup_epochs': TRAIN_CONFIG['warmup_epochs'],
            'resume': resume
        }
        
        logger.info(f"训练参数: {train_args}")
        
        # 开始训练
        results = model.train(**train_args)
        
        # 保存最佳模型到weights目录
        best_model_path = Path(TRAIN_CONFIG['project']) / TRAIN_CONFIG['name'] / 'weights' / 'best.pt'
        if best_model_path.exists():
            target_path = PATHS['weights'] / 'best.pt'
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy2(str(best_model_path), str(target_path))
            logger.info(f"最佳模型已保存到: {target_path}")
        
        logger.info("模型训练完成")
        return True
        
    except Exception as e:
        logger.error(f"模型训练失败: {str(e)}")
        return False


def evaluate_model(model_path: str, dataset_path: str, device: str = None) -> bool:
    """评估模型
    
    Args:
        model_path: 模型路径
        dataset_path: 数据集路径
        device: 评估设备
        
    Returns:
        是否评估成功
    """
    try:
        logger.info("开始模型评估...")
        
        device = device or TRAIN_CONFIG['device']
        
        # 检查模型文件
        if not Path(model_path).exists():
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        # 加载模型
        model = YOLO(model_path)
        
        # 评估参数
        eval_args = {
            'data': str(Path(dataset_path) / 'dataset.yaml'),
            'device': device,
            'project': TRAIN_CONFIG['project'],
            'name': f"{TRAIN_CONFIG['name']}_eval"
        }
        
        # 开始评估
        results = model.val(**eval_args)
        
        # 打印结果
        logger.info("评估结果:")
        logger.info(f"  mAP50: {results.box.map50:.4f}")
        logger.info(f"  mAP50-95: {results.box.map:.4f}")
        logger.info(f"  Precision: {results.box.mp:.4f}")
        logger.info(f"  Recall: {results.box.mr:.4f}")
        
        logger.info("模型评估完成")
        return True
        
    except Exception as e:
        logger.error(f"模型评估失败: {str(e)}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='YOLO11车牌检测模型训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python train.py --data data/datasets --epochs 100                    # 训练模型
  python train.py --data data/datasets --eval weights/best.pt         # 评估模型
  python train.py --data data/datasets --resume                       # 恢复训练
        """
    )
    
    # 数据集参数
    parser.add_argument('--data', '-d', type=str, required=True, help='数据集路径')
    
    # 训练参数
    parser.add_argument('--epochs', '-e', type=int, help=f'训练轮数 (默认: {TRAIN_CONFIG["epochs"]})')
    parser.add_argument('--batch', '-b', type=int, help=f'批次大小 (默认: {TRAIN_CONFIG["batch_size"]})')
    parser.add_argument('--device', type=str, help=f'训练设备 (默认: {TRAIN_CONFIG["device"]})')
    parser.add_argument('--model', '-m', type=str, help='预训练模型路径')
    
    # 操作选项
    parser.add_argument('--resume', '-r', action='store_true', help='恢复训练')
    parser.add_argument('--eval', type=str, help='评估指定模型')
    parser.add_argument('--validate-only', action='store_true', help='仅验证数据集')
    
    # 其他选项
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    try:
        # 验证数据集
        if not validate_dataset(args.data):
            logger.error("数据集验证失败，程序退出")
            sys.exit(1)
        
        if args.validate_only:
            logger.info("数据集验证完成")
            return
        
        # 评估模式
        if args.eval:
            if not evaluate_model(args.eval, args.data, args.device):
                logger.error("模型评估失败")
                sys.exit(1)
            return
        
        # 训练模式
        if not train_model(
            dataset_path=args.data,
            model_path=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            device=args.device,
            resume=args.resume
        ):
            logger.error("模型训练失败")
            sys.exit(1)
        
        logger.info("程序执行完成")
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()