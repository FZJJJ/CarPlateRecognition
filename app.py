# -*- coding: utf-8 -*-
"""
Flask Web应用
提供车牌识别的Web界面
"""

import os
import time
import uuid
from pathlib import Path
from typing import Dict, Any

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from loguru import logger

from config import WEB_CONFIG, PATHS, IMAGE_CONFIG
from models.pipeline import PlateRecognitionPipeline
from utils.image_utils import load_image, save_image
from utils.visualization import PlateVisualizer
from models.history import HistoryDatabase


# 创建Flask应用
app = Flask(__name__)
app.config.update(WEB_CONFIG)

# 启用CORS
CORS(app)

# 全局变量
pipeline = None
visualizer = None
history_db = None


def init_app():
    """初始化应用"""
    global pipeline, visualizer, history_db
    
    try:
        logger.info("初始化Web应用...")
        
        # 创建必要目录
        PATHS['uploads'].mkdir(parents=True, exist_ok=True)
        PATHS['result_images'].mkdir(parents=True, exist_ok=True)
        
        # 初始化识别管道
        logger.info("初始化识别管道...")
        pipeline = PlateRecognitionPipeline()
        
        # 初始化可视化器
        visualizer = PlateVisualizer()
        
        # 初始化历史数据库
        history_db = HistoryDatabase()
        
        logger.info("Web应用初始化完成")
        
    except Exception as e:
        logger.error(f"Web应用初始化失败: {str(e)}")
        raise


def allowed_file(filename: str) -> bool:
    """检查文件类型是否允许"""
    if not filename:
        return False
    
    ext = Path(filename).suffix.lower()
    return ext in IMAGE_CONFIG['supported_formats']


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """文件上传和识别"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            }), 400
        
        file = request.files['file']
        
        # 检查文件名
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': '没有选择文件'
            }), 400
        
        # 检查文件类型
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'不支持的文件类型，支持的格式: {", ".join(IMAGE_CONFIG["supported_formats"])}'
            }), 400
        
        # 生成安全的文件名
        original_filename = secure_filename(file.filename)
        file_ext = Path(original_filename).suffix
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        
        # 保存上传的文件
        upload_path = PATHS['uploads'] / unique_filename
        file.save(str(upload_path))
        
        logger.info(f"文件上传成功: {original_filename} -> {unique_filename}")
        
        # 加载和验证图像
        image = load_image(str(upload_path))
        if image is None:
            return jsonify({
                'success': False,
                'error': '无法读取图像文件'
            }), 400
        
        # 执行车牌识别
        start_time = time.time()
        result = pipeline.process_image(image)
        process_time = time.time() - start_time
        
        # 生成结果图像
        result_image_path = None
        if result['success'] and result['detections']:
            # 可视化结果
            result_image = visualizer.visualize_result(image, result)
            
            # 保存结果图像
            result_filename = f"result_{uuid.uuid4().hex}.jpg"
            result_image_path = PATHS['result_images'] / result_filename
            
            if save_image(result_image, str(result_image_path)):
                result['result_image_url'] = url_for('get_result_image', filename=result_filename)
            else:
                logger.warning("结果图像保存失败")
        
        # 添加到历史记录
        try:
            history_db.add_record(
                original_filename=original_filename,
                original_image_path=str(upload_path),
                result_image_path=str(result_image_path) if result_image_path else None,
                plate_numbers=result.get('plate_numbers', []),
                confidence_scores=result.get('confidence_scores', []),
                detections_count=result.get('detection_count', 0),
                process_time=process_time,
                success=result['success'],
                error_message=result.get('error')
            )
        except Exception as e:
            logger.warning(f"添加历史记录失败: {str(e)}")
        
        # 添加额外信息
        result.update({
            'original_filename': original_filename,
            'upload_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'file_size': upload_path.stat().st_size,
            'image_dimensions': f"{image.shape[1]}x{image.shape[0]}"
        })
        
        logger.info(f"识别完成: {original_filename}, 耗时: {process_time:.2f}秒")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'处理失败: {str(e)}'
        }), 500


@app.route('/result_image/<filename>')
def get_result_image(filename):
    """获取结果图像"""
    try:
        file_path = PATHS['result_images'] / filename
        if file_path.exists():
            return send_file(str(file_path), mimetype='image/jpeg')
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        logger.error(f"获取结果图像失败: {str(e)}")
        return jsonify({'error': '获取图像失败'}), 500


@app.route('/history')
def get_history():
    """获取历史记录"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        success_only = request.args.get('success_only', type=bool)
        
        offset = (page - 1) * per_page
        
        records = history_db.get_records(
            limit=per_page,
            offset=offset,
            success_only=success_only
        )
        
        return jsonify({
            'success': True,
            'records': records,
            'page': page,
            'per_page': per_page
        })
        
    except Exception as e:
        logger.error(f"获取历史记录失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/stats')
def get_stats():
    """获取统计信息"""
    try:
        stats = history_db.get_statistics()
        
        # 添加管道统计信息
        if pipeline and hasattr(pipeline, 'stats'):
            stats.update({
                'pipeline_stats': pipeline.stats
            })
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health')
def health_check():
    """健康检查"""
    try:
        # 检查各组件状态
        status = {
            'app': 'healthy',
            'pipeline': 'healthy' if pipeline else 'error',
            'detector': 'healthy' if pipeline and pipeline.detector and pipeline.detector.is_loaded else 'error',
            'recognizer': 'healthy' if pipeline and pipeline.recognizer and pipeline.recognizer.is_loaded else 'error',
            'database': 'healthy' if history_db else 'error'
        }
        
        overall_status = 'healthy' if all(s == 'healthy' for s in status.values()) else 'degraded'
        
        return jsonify({
            'status': overall_status,
            'components': status,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({'error': '页面不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    logger.error(f"内部服务器错误: {str(error)}")
    return jsonify({'error': '内部服务器错误'}), 500


if __name__ == '__main__':
    # 初始化应用
    init_app()
    
    # 启动应用
    logger.info(f"启动Web服务器: http://{WEB_CONFIG['host']}:{WEB_CONFIG['port']}")
    app.run(
        host=WEB_CONFIG['host'],
        port=WEB_CONFIG['port'],
        debug=WEB_CONFIG['debug'],
        threaded=WEB_CONFIG['threaded']
    )