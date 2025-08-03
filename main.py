# -*- coding: utf-8 -*-
"""
车牌识别系统主程序
提供命令行接口进行车牌检测和识别
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from loguru import logger

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

from config import (
    YOLO_CONFIG, OCR_CONFIG, PATHS, 
    VISUALIZATION_CONFIG, PERFORMANCE_CONFIG
)
from core.detector import PlateDetector
from core.recognizer import PlateRecognizer
from core.pipeline import PlateRecognitionPipeline
from utils.image_utils import load_image, save_image
from utils.visualization import PlateVisualizer
from utils.text_filter import validate_plate_number


class CarPlateRecognitionCLI:
    """车牌识别命令行接口"""
    
    def __init__(self):
        """初始化CLI"""
        self.setup_logging()
        self.pipeline = None
        self.visualizer = PlateVisualizer()
        
    def setup_logging(self):
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
                PATHS['app_log'],
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="10 MB",
                retention="7 days",
                compression="zip"
            )
            
            logger.info("日志系统初始化完成")
            
        except Exception as e:
            print(f"日志设置失败: {str(e)}")
    
    def initialize_pipeline(self) -> bool:
        """初始化识别管道"""
        try:
            logger.info("正在初始化车牌识别管道...")
            
            # 检查模型文件
            if not PATHS['yolo_weights'].exists():
                logger.error(f"YOLO模型文件不存在: {PATHS['yolo_weights']}")
                logger.info("请确保模型文件存在或运行训练脚本")
                return False
            
            # 初始化管道
            self.pipeline = PlateRecognitionPipeline(
                detector_weights=str(PATHS['yolo_weights']),
                detector_config=YOLO_CONFIG,
                ocr_config=OCR_CONFIG
            )
            
            logger.info("车牌识别管道初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"管道初始化失败: {str(e)}")
            return False
    
    def process_single_image(self, image_path: str, 
                           output_dir: Optional[str] = None,
                           save_result: bool = True,
                           show_result: bool = False) -> List[Dict[str, Any]]:
        """处理单张图像
        
        Args:
            image_path: 图像路径
            output_dir: 输出目录
            save_result: 是否保存结果
            show_result: 是否显示结果
            
        Returns:
            识别结果列表
        """
        try:
            logger.info(f"处理图像: {image_path}")
            
            # 加载图像
            image = load_image(image_path)
            if image is None:
                logger.error(f"无法加载图像: {image_path}")
                return []
            
            # 执行识别
            start_time = time.time()
            results = self.pipeline.process_image(image)
            process_time = time.time() - start_time
            
            logger.info(f"识别完成，耗时: {process_time:.2f}秒")
            logger.info(f"检测到 {len(results)} 个车牌")
            
            # 打印结果
            for i, result in enumerate(results, 1):
                plate_number = result.get('plate_number', 'Unknown')
                confidence = result.get('confidence', 0.0)
                bbox = result.get('bbox', [])
                
                logger.info(f"车牌 {i}: {plate_number} (置信度: {confidence:.2f})")
                
                if len(bbox) >= 4:
                    logger.debug(f"位置: x={bbox[0]:.0f}, y={bbox[1]:.0f}, w={bbox[2]:.0f}, h={bbox[3]:.0f}")
            
            # 可视化和保存结果
            if results and (save_result or show_result):
                result_image = self.visualizer.visualize_result(image, results)
                
                if save_result:
                    # 确定输出路径
                    if output_dir:
                        output_path = Path(output_dir)
                    else:
                        output_path = PATHS['result_images']
                    
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    # 生成输出文件名
                    input_name = Path(image_path).stem
                    output_file = output_path / f"{input_name}_result.jpg"
                    
                    # 保存结果图像
                    if save_image(result_image, output_file):
                        logger.info(f"结果已保存: {output_file}")
                    else:
                        logger.error(f"保存结果失败: {output_file}")
                
                if show_result:
                    # 显示结果
                    cv2.imshow('Car Plate Recognition Result', result_image)
                    logger.info("按任意键继续...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            
            return results
            
        except Exception as e:
            logger.error(f"处理图像失败: {str(e)}")
            return []
    
    def process_batch_images(self, input_dir: str, 
                           output_dir: Optional[str] = None,
                           recursive: bool = False) -> Dict[str, Any]:
        """批量处理图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            recursive: 是否递归搜索
            
        Returns:
            处理统计信息
        """
        try:
            logger.info(f"开始批量处理: {input_dir}")
            
            input_path = Path(input_dir)
            if not input_path.exists():
                logger.error(f"输入目录不存在: {input_dir}")
                return {}
            
            # 查找图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            
            if recursive:
                for ext in image_extensions:
                    image_files.extend(input_path.rglob(f'*{ext}'))
                    image_files.extend(input_path.rglob(f'*{ext.upper()}'))
            else:
                for ext in image_extensions:
                    image_files.extend(input_path.glob(f'*{ext}'))
                    image_files.extend(input_path.glob(f'*{ext.upper()}'))
            
            if not image_files:
                logger.warning(f"在目录中未找到图像文件: {input_dir}")
                return {}
            
            logger.info(f"找到 {len(image_files)} 个图像文件")
            
            # 处理统计
            stats = {
                'total_images': len(image_files),
                'processed_images': 0,
                'total_plates': 0,
                'valid_plates': 0,
                'processing_time': 0,
                'results': []
            }
            
            start_time = time.time()
            
            # 逐个处理图像
            for i, image_file in enumerate(image_files, 1):
                logger.info(f"处理进度: {i}/{len(image_files)} - {image_file.name}")
                
                try:
                    results = self.process_single_image(
                        str(image_file),
                        output_dir=output_dir,
                        save_result=True,
                        show_result=False
                    )
                    
                    stats['processed_images'] += 1
                    stats['total_plates'] += len(results)
                    
                    # 统计有效车牌
                    valid_count = sum(1 for r in results 
                                    if validate_plate_number(r.get('plate_number', '')))
                    stats['valid_plates'] += valid_count
                    
                    # 记录结果
                    stats['results'].append({
                        'file': str(image_file),
                        'plates_detected': len(results),
                        'valid_plates': valid_count,
                        'results': results
                    })
                    
                except Exception as e:
                    logger.error(f"处理文件失败 {image_file}: {str(e)}")
                    continue
            
            stats['processing_time'] = time.time() - start_time
            
            # 打印统计信息
            logger.info("=" * 50)
            logger.info("批量处理完成")
            logger.info(f"总图像数: {stats['total_images']}")
            logger.info(f"成功处理: {stats['processed_images']}")
            logger.info(f"检测车牌: {stats['total_plates']}")
            logger.info(f"有效车牌: {stats['valid_plates']}")
            logger.info(f"总耗时: {stats['processing_time']:.2f}秒")
            logger.info(f"平均耗时: {stats['processing_time']/max(1, stats['processed_images']):.2f}秒/图")
            logger.info("=" * 50)
            
            return stats
            
        except Exception as e:
            logger.error(f"批量处理失败: {str(e)}")
            return {}
    
    def process_camera(self, camera_id: int = 0, 
                      save_results: bool = False,
                      output_dir: Optional[str] = None) -> None:
        """实时摄像头处理
        
        Args:
            camera_id: 摄像头ID
            save_results: 是否保存结果
            output_dir: 输出目录
        """
        try:
            logger.info(f"启动摄像头: {camera_id}")
            
            # 打开摄像头
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                logger.error(f"无法打开摄像头: {camera_id}")
                return
            
            # 设置摄像头参数
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("摄像头已启动，按 'q' 退出，按 's' 保存当前帧")
            
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("无法读取摄像头帧")
                    break
                
                frame_count += 1
                
                # 每隔几帧处理一次（提高性能）
                if frame_count % 5 == 0:
                    try:
                        # 执行识别
                        results = self.pipeline.process_image(frame)
                        
                        # 可视化结果
                        if results:
                            frame = self.visualizer.visualize_result(frame, results)
                    
                    except Exception as e:
                        logger.debug(f"处理帧失败: {str(e)}")
                
                # 显示帧
                cv2.imshow('Car Plate Recognition - Camera', frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_results:
                    # 保存当前帧
                    if output_dir:
                        output_path = Path(output_dir)
                    else:
                        output_path = PATHS['result_images']
                    
                    output_path.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = int(time.time())
                    output_file = output_path / f"camera_capture_{timestamp}.jpg"
                    
                    if save_image(frame, output_file):
                        logger.info(f"帧已保存: {output_file}")
            
            # 释放资源
            cap.release()
            cv2.destroyAllWindows()
            logger.info("摄像头已关闭")
            
        except Exception as e:
            logger.error(f"摄像头处理失败: {str(e)}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='车牌识别系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py --image path/to/image.jpg                    # 处理单张图像
  python main.py --batch path/to/images --output results     # 批量处理
  python main.py --camera 0 --save                          # 实时摄像头
        """
    )
    
    # 输入选项
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', '-i', type=str, help='输入图像路径')
    input_group.add_argument('--batch', '-b', type=str, help='批量处理目录')
    input_group.add_argument('--camera', '-c', type=int, nargs='?', const=0, help='摄像头ID (默认: 0)')
    
    # 输出选项
    parser.add_argument('--output', '-o', type=str, help='输出目录')
    parser.add_argument('--save', '-s', action='store_true', help='保存结果')
    parser.add_argument('--show', action='store_true', help='显示结果')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归搜索子目录')
    
    # 其他选项
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    
    args = parser.parse_args()
    
    # 调整日志级别
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    elif args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    
    # 初始化CLI
    cli = CarPlateRecognitionCLI()
    
    # 初始化识别管道
    if not cli.initialize_pipeline():
        logger.error("初始化失败，程序退出")
        sys.exit(1)
    
    try:
        if args.image:
            # 单张图像处理
            results = cli.process_single_image(
                args.image,
                output_dir=args.output,
                save_result=args.save,
                show_result=args.show
            )
            
            if not results:
                logger.warning("未检测到车牌")
                sys.exit(1)
        
        elif args.batch:
            # 批量处理
            stats = cli.process_batch_images(
                args.batch,
                output_dir=args.output,
                recursive=args.recursive
            )
            
            if not stats or stats.get('processed_images', 0) == 0:
                logger.error("批量处理失败")
                sys.exit(1)
        
        elif args.camera is not None:
            # 摄像头处理
            cli.process_camera(
                camera_id=args.camera,
                save_results=args.save,
                output_dir=args.output
            )
        
        logger.info("程序执行完成")
        
    except KeyboardInterrupt:
        logger.info("用户中断程序")
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()