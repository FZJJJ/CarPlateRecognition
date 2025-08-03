# 车牌识别系统 (Car Plate Recognition)

基于YOLO11和PaddleOCR的智能车牌识别系统，提供高精度的车牌检测与识别功能，支持命令行、Web界面和实时摄像头识别等多种使用方式。

## ✨ 功能特点

- 🚗 **高精度车牌检测**: 基于YOLO11深度学习模型，支持多种车牌类型
- 📝 **准确文字识别**: 集成PaddleOCR光学字符识别，支持中文车牌
- 🌐 **Web界面**: 现代化的Web界面，支持图片上传和实时预览
- ⚡ **实时处理**: 支持摄像头实时车牌识别
- 📊 **历史记录**: 完整的识别历史记录和统计分析
- 🔧 **模块化设计**: 易于扩展和自定义
- 🎯 **多种部署方式**: 命令行、Web服务、API接口
- 📈 **性能监控**: 详细的处理时间和准确率统计

## 🏗️ 项目结构

```
CarPlateRecgonition/
├── README.md                  # 项目说明文档
├── requirements.txt           # Python依赖包列表
├── config.py                  # 全局配置文件
├── main.py                    # 命令行程序入口
├── app.py                     # Flask Web应用入口
├── train.py                   # YOLO11模型训练脚本
├── convert_data.py            # 数据格式转换工具
├── DATA_CONVERSION_GUIDE.md   # 数据转换指南
│
├── models/                    # 核心模型模块
│   ├── __init__.py
│   ├── pipeline.py            # 识别流水线
│   ├── yolo_detector.py       # YOLO检测器
│   ├── paddle_ocr.py          # PaddleOCR识别器
│   └── history.py             # 历史记录管理
│
├── utils/                     # 工具函数模块
│   ├── __init__.py
│   ├── image_utils.py         # 图像处理工具
│   ├── text_filter.py         # 文本过滤和验证
│   ├── visualization.py       # 结果可视化
│   └── data_loader.py         # 数据加载器
│
├── web/                       # Web应用相关
│   ├── static/                # 静态资源
│   │   ├── css/               # 样式文件
│   │   ├── js/                # JavaScript文件
│   │   └── uploads/           # 上传文件目录
│   └── templates/             # HTML模板
│       └── index.html         # 主页模板
│
├── data/                      # 数据相关
│   ├── dataset.yaml           # 数据集配置
│   ├── datasets/              # 训练数据集
│   │   ├── images/            # 图像文件
│   │   └── labels/            # 标注文件
│   └── sample_images/         # 示例图片
│
├── weights/                   # 模型权重文件
│   ├── README.md              # 模型说明
│   ├── best.pt                # 训练好的车牌检测模型
│   └── yolo11n.pt             # YOLO11预训练模型
│
├── results/                   # 结果输出
│   ├── README.md              # 结果说明
│   ├── detection_log.csv      # 检测日志
│   └── images/                # 结果图片
│
├── logs/                      # 日志文件
│   ├── app.log                # 应用日志
│   └── training.log           # 训练日志
│
├── database/                  # 数据库文件
│   └── history.db             # SQLite历史记录数据库
│
├── tests/                     # 测试模块
│   ├── __init__.py
│   ├── test_detector.py       # 检测器测试
│   ├── test_ocr.py            # OCR测试
│   └── test_pipeline.py       # 流水线测试
│
├── runs/                      # 训练运行记录
│   ├── detect/                # 检测结果
│   └── train/                 # 训练结果
│
└── archive/                   # 原始数据存档
    ├── annotations/           # 原始标注
    └── images/                # 原始图片
```

## 🚀 快速开始

### 1. 环境准备

**系统要求:**
- Python 3.8+
- CUDA 11.0+ (可选，用于GPU加速)
- 内存: 4GB+
- 存储: 2GB+

**安装步骤:**

```bash
# 克隆项目
git clone https://github.com/FZJJJ/CarPlateRecognition.git
cd CarPlateRecognition

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 模型准备

**下载预训练模型:**

将以下模型文件放置到 `weights/` 目录：
- `yolo11n.pt` - YOLO11预训练模型（自动下载）
- `best.pt` - 车牌检测专用模型

### 3. 使用方式

#### 🖥️ 命令行模式

```bash
# 单张图片识别
python main.py --image path/to/image.jpg

# 批量处理目录
python main.py --batch path/to/images/ --output results/

# 实时摄像头识别
python main.py --realtime

# 使用指定摄像头
python main.py --realtime --camera 1

# 详细输出模式
python main.py --image test.jpg --verbose
```

#### 🌐 Web界面模式

```bash
# 启动Web服务
python app.py

# 访问 http://localhost:5000
# 支持功能:
# - 图片上传识别
# - 实时摄像头识别
# - 历史记录查看
# - 统计分析
```

#### 🎯 模型训练

```bash
# 使用默认配置训练
python train.py --data data/dataset.yaml

# 自定义训练参数
python train.py --data data/dataset.yaml --epochs 200 --batch 32 --lr 0.01

# 恢复训练
python train.py --data data/dataset.yaml --resume
```

## ⚙️ 配置说明

### 主要配置文件: `config.py`

```python
# YOLO检测参数
YOLO_CONFIG = {
    "conf_threshold": 0.5,      # 置信度阈值
    "iou_threshold": 0.45,      # NMS IoU阈值
    "max_det": 10,              # 最大检测数量
    "device": "auto",           # 设备选择
}

# PaddleOCR配置
OCR_CONFIG = {
    "use_angle_cls": True,       # 角度分类器
    "lang": "ch",               # 中文模式
    "use_gpu": True,            # GPU加速
}

# Web应用配置
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False
}
```

## 📊 性能指标

- **检测精度**: mAP@0.5 > 95%
- **识别准确率**: > 98%
- **处理速度**: 单张图片 < 200ms
- **支持车牌类型**: 蓝牌、绿牌、黄牌等

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

- 项目链接: [https://github.com/FZJJJ/CarPlateRecognition](https://github.com/FZJJJ/CarPlateRecognition)
- 问题反馈: [Issues](https://github.com/FZJJJ/CarPlateRecognition/issues)

## 🙏 致谢

- [YOLO11](https://github.com/ultralytics/ultralytics) - 目标检测框架
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 光学字符识别
- [Flask](https://flask.palletsprojects.com/) - Web框架