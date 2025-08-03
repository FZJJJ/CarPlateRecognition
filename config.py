# -*- coding: utf-8 -*-
"""
全局配置文件
包含模型路径、识别参数、日志设置等配置项
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.absolute()

# ==================== 路径配置 ====================
# 模型权重路径
WEIGHTS_DIR = ROOT_DIR / "weights"
YOLO_WEIGHTS = WEIGHTS_DIR / "best.pt"
YOLO_PRETRAINED = WEIGHTS_DIR / "yolo11n.pt"

# 数据路径
DATA_DIR = ROOT_DIR / "data"
DATASET_DIR = DATA_DIR / "datasets"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
DATASET_YAML = DATA_DIR / "dataset.yaml"

# 结果输出路径
RESULTS_DIR = ROOT_DIR / "results"
RESULT_IMAGES_DIR = RESULTS_DIR / "images"
DETECTION_LOG = RESULTS_DIR / "detection_log.csv"
PERFORMANCE_LOG = RESULTS_DIR / "performance_log.txt"

# Web应用路径
WEB_DIR = ROOT_DIR / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"
UPLOADS_DIR = STATIC_DIR / "uploads"

# 日志路径
LOGS_DIR = ROOT_DIR / "logs"
TRAINING_LOG = LOGS_DIR / "training.log"
APP_LOG = LOGS_DIR / "app.log"

# 数据库路径
DATABASE_DIR = ROOT_DIR / "database"

# ==================== 模型配置 ====================
# YOLO检测参数
YOLO_CONFIG = {
    "conf_threshold": 0.5,      # 置信度阈值
    "iou_threshold": 0.45,      # NMS IoU阈值
    "max_det": 10,              # 最大检测数量
    "device": "auto",           # 设备选择: 'auto', 'cpu', 'cuda:0'
    "half": False,              # 是否使用FP16推理
    "augment": False,           # 是否使用TTA增强
}

# PaddleOCR配置
OCR_CONFIG = {
    "use_angle_cls": True,       # 是否使用角度分类器
    "lang": "ch",               # 语言类型 - 中文模式对车牌识别更适合
    "use_gpu": True,            # 是否使用GPU
    "show_log": True,           # 显示日志以便调试
    "det_db_thresh": 0.2,       # 降低检测阈值提高敏感度
    "det_db_box_thresh": 0.5,   # 降低文本框阈值
    "rec_char_dict_path": None, # 字符字典路径
}

# ==================== 图像处理配置 ====================
IMAGE_CONFIG = {
    "input_size": (640, 640),    # 输入图像尺寸
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "max_file_size": 10 * 1024 * 1024,  # 最大文件大小 (10MB)
    "quality": 95,               # 保存质量
}

# ==================== 车牌识别配置 ====================
PLATE_CONFIG = {
    "min_plate_area": 1000,      # 最小车牌面积
    "max_plate_area": 50000,     # 最大车牌面积
    "min_aspect_ratio": 2.0,     # 最小宽高比
    "max_aspect_ratio": 6.0,     # 最大宽高比
    "char_whitelist": "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领",
}

# ==================== Web应用配置 ====================
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
    "threaded": True,
    "upload_folder": str(UPLOADS_DIR),
    "max_content_length": 16 * 1024 * 1024,  # 16MB
    "secret_key": "car_plate_recognition_secret_key_2024",
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "patience": 50,
    "save_period": 10,
    "workers": 8,
    "device": "auto",
    "project": "runs/train",
    "name": "plate_detection",
}

# ==================== 性能配置 ====================
PERFORMANCE_CONFIG = {
    "enable_profiling": False,
    "memory_limit": 4 * 1024 * 1024 * 1024,  # 4GB
    "timeout": 30,  # 处理超时时间(秒)
    "max_concurrent": 4,  # 最大并发数
}

# ==================== 可视化配置 ====================
VISUALIZATION_CONFIG = {
    "colors": {
        "detection": (0, 255, 0),    # 绿色
        "text": (255, 255, 255),     # 白色
        "background": (0, 0, 0),     # 黑色
        "confidence": (255, 255, 0), # 黄色
        "error": (0, 0, 255),        # 红色
        "warning": (0, 165, 255)     # 橙色
    },
    "font": {
        "scale": 0.7,
        "thickness": 2,
        "type": "FONT_HERSHEY_SIMPLEX"
    },
    "box": {
        "thickness": 2,
        "min_thickness": 1
    },
    "output": {
        "quality": 95,
        "format": "jpg"
    }
}

# ==================== 数据配置 ====================
DATA_CONFIG = {
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "min_image_size": (32, 32),
    "max_image_size": (4096, 4096),
    "batch_size": 32,
    "shuffle": True,
    "num_workers": 4,
    "pin_memory": True,
    "drop_last": False,
}

# ==================== 路径配置字典 ====================
PATHS = {
    "root": ROOT_DIR,
    "weights": WEIGHTS_DIR,
    "data": DATA_DIR,
    "datasets": DATASET_DIR,
    "sample_images": SAMPLE_IMAGES_DIR,
    "results": RESULTS_DIR,
    "logs": LOGS_DIR,
    "web": WEB_DIR,
    "static": STATIC_DIR,
    "templates": TEMPLATES_DIR,
    "uploads": UPLOADS_DIR,
    "yolo_weights": YOLO_WEIGHTS,
    "yolo_pretrained": YOLO_PRETRAINED,
    "dataset_yaml": DATASET_YAML,
    "detection_log": DETECTION_LOG,
    "performance_log": PERFORMANCE_LOG,
    "training_log": TRAINING_LOG,
    "app_log": APP_LOG,
}

# ==================== 创建必要目录 ====================
def create_directories():
    """创建项目所需的目录结构"""
    directories = [
        WEIGHTS_DIR,
        DATA_DIR,
        DATASET_DIR,
        SAMPLE_IMAGES_DIR,
        RESULTS_DIR,
        RESULT_IMAGES_DIR,
        WEB_DIR,
        STATIC_DIR,
        TEMPLATES_DIR,
        UPLOADS_DIR,
        LOGS_DIR,
        DATABASE_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        
if __name__ == "__main__":
    create_directories()
    print("目录结构创建完成！")