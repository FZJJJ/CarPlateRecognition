# -*- coding: utf-8 -*-
"""
工具函数模块
包含图像处理、文本过滤、数据加载和可视化等工具函数
"""

from .image_utils import (
    load_image,
    save_image,
    resize_image,
    enhance_image,
    convert_color_space
)

from .text_filter import (
    validate_plate_number,
    clean_plate_text,
    filter_valid_plates,
    extract_plate_info
)

from .data_loader import (
    prepare_dataset,
    validate_dataset,
    load_annotations,
    split_dataset
)

from .visualization import (
    draw_detection_results,
    draw_bounding_boxes,
    create_result_image,
    plot_statistics
)

__all__ = [
    # 图像工具
    'load_image',
    'save_image', 
    'resize_image',
    'enhance_image',
    'convert_color_space',
    
    # 文本过滤
    'validate_plate_number',
    'clean_plate_text',
    'filter_valid_plates',
    'extract_plate_info',
    
    # 数据加载
    'prepare_dataset',
    'validate_dataset',
    'load_annotations',
    'split_dataset',
    
    # 可视化
    'draw_detection_results',
    'draw_bounding_boxes',
    'create_result_image',
    'plot_statistics'
]

__version__ = '1.0.0'
__author__ = 'Car Plate Recognition Team'
__description__ = '车牌识别系统工具函数集合'