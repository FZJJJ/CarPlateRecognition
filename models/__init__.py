# -*- coding: utf-8 -*-
"""
车牌识别模型模块
包含YOLO11检测器、PaddleOCR识别器和完整的识别流程管理
"""

from .yolo_detector import YOLODetector
from .paddle_ocr import PaddleOCRRecognizer
from .pipeline import PlateRecognitionPipeline

__all__ = [
    'YOLODetector',
    'PaddleOCRRecognizer', 
    'PlateRecognitionPipeline'
]

__version__ = '1.0.0'
__author__ = 'Car Plate Recognition Team'
__description__ = '基于YOLO11和PaddleOCR的车牌识别系统'