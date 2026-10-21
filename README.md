深度学习YOLOV8模型训练 高空抛物检测数据集 识别检测高空抛物中的矿泉水瓶`、`易拉罐`、`垃圾袋`、`垃圾箱`、`纸团`、`纸杯`、`购物袋
@[toc]

### **高空抛物检测数据集**

| **项目** | **内容** |
|----------|----------|
| **数据集名称** | 高空抛物检测数据集 |
| **图像总数** | 696 张 |
| **检测类别** | 7 类（具体如下） |
| **类别列表** | `矿泉水瓶`、`易拉罐`、`垃圾袋`、`垃圾箱`、`纸团`、`纸杯`、`购物袋` |
| **统一类别标签（可选）** | `obj`（用于统一检测任务，共 729 个标注框） |
| **标注方式** | 真人专业标注，保证标注质量与边界框精度 |
| **标注格式** | 同时提供 YOLO `.txt` 格式 和 PASCAL VOC `.xml` 格式 |
| **适用任务** | 目标检测（单类或细分类）、智能安防、社区安全监控 |
| **使用说明** | 可直接用于 YOLO（如 YOLOv5/v8）、Faster R-CNN 等主流框架训练 |

---

### **类别明细表**

| 类别编号 | 类别名称（中文） | 类别名称（英文建议） | 说明 |
|----------|------------------|------------------------|------|
| 0 | 矿泉水瓶 | water_bottle | 塑料瓶，常见于高空抛物 |
| 1 | 易拉罐 | can | 金属饮料罐 |
| 2 | 垃圾袋 | trash_bag | 装有垃圾的塑料袋 |
| 3 | 垃圾箱 | trash_bin | 小型垃圾桶被抛下（较少见但存在） |
| 4 | 纸团 | paper_ball | 废纸揉成团抛出 |
| 5 | 纸杯 | paper_cup | 一次性纸杯 |
| 6 | 购物袋 | shopping_bag | 手提式塑料或无纺布袋 |

> ✅ 支持两种训练模式：
> - **细分类检测**：7 类独立识别
> - **通用抛物检测**：合并为 1 类 `obj`（共 729 个标注框），提升召回率

---

### **标注统计信息**

| 统计项 | 数量 |
|--------|------|
| 图像总数 | 696 张 |
| 标注框总数（obj） | 729 个 |
| 平均每图目标数 | ~1.05 个 |
| 标注质量 | 真人标注，边界框紧密贴合目标 |

---

### **数据集特点**

- ✅ **真实场景模拟**：包含多种常见高空抛物对象，贴近实际社区监控需求  
- ✅ **双格式支持**：同时提供 YOLO 和 VOC 格式，兼容性强  
- ✅ **高质量标注**：人工标注，避免误标漏标，提升模型训练效果  


---

### **推荐使用方式**

| 任务类型 | 推荐做法 |
|----------|----------|
| **通用抛物检测** | 将所有类别合并为 `obj` 类，训练单类检测器，提高泛化能力 |
| **细粒度识别** | 保留 7 类，用于分析抛物类型分布 |
| **模型选择** | 推荐 YOLOv8n / YOLOv5s（轻量、适合实时监控） |
| **数据增强** | 建议使用 Mosaic、随机旋转、色彩抖动提升鲁棒性 |

---

> 📌 **应用场景**：  
> - 智慧社区安防系统  
> - 高层建筑监控AI预警  
> - 物业管理智能化升级  
> - 科研项目、课程设计、毕业设计  



![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b7e7e441f0a041118dbb46d2237a499c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d2614d193af4428d8dbb13c2bf451dcd.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/841bd25407794726827274e33fac764c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c53a1e9acac344208da1269086e3efe4.png)
1
使用 **YOLOv8（Ultralytics 实现）** 来完成从环境搭建 → 模型训练 → 推理 → 评估 的全流程。

---

## ✅ 一、项目结构建议

```
falling_object_project/
│
├── datasets/
│   └── falling_objects/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       ├── labels/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── falling_objects.yaml
│
├── models/
│   └── yolov8n.pt          # 官方预训练权重
│
├── runs/                   # 自动保存训练结果
│
├── train.py                # 训练脚本
├── detect.py               # 推理脚本
├── val.py                  # 评估脚本
├── install_env.py          # 环境安装脚本
└── README.md
```

---

## ✅ 二、环境搭建全流程

### 1. 安装 CUDA 驱动（GPU 支持）

- 确保你的 NVIDIA 显卡驱动支持 CUDA。
- 打开命令行，输入：
  ```bash
  nvidia-smi
  ```
  若显示 GPU 信息（如显存、驱动版本），则 CUDA 驱动正常。

> 推荐 CUDA 版本：11.8 或 12.1（与 PyTorch 兼容）

---

### 2. 安装 Anaconda

- 下载地址：[https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- 安装后验证：
  ```bash
  conda --version
  ```

---

### 3. 创建 Python 虚拟环境并安装依赖

#### `install_env.py`

```python
# install_env.py
import subprocess
import sys

def install_packages():
    # 创建虚拟环境
    subprocess.run([
        "conda", "create", "-n", "falling_yolo", "python=3.9", "-y"
    ], shell=True)

    # 激活并安装包
    commands = [
        "conda activate falling_yolo",
        "pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html",
        "pip install ultralytics opencv-python PyQt5 matplotlib pillow"
    ]

    for cmd in commands:
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    install_packages()
```

> 📌 使用 `torch 2.0.1 + CUDA 11.8`，兼容性好，支持 YOLOv8 最新版。

激活环境：
```bash
conda activate falling_yolo
```

---

## ✅ 三、数据集配置文件 `falling_objects.yaml`

```yaml
# falling_objects.yaml
train: ../datasets/falling_objects/images/train
val: ../datasets/falling_objects/images/val
test: ../datasets/falling_objects/images/test

nc: 7
names: ['bottle', 'can', 'garbage_bag', 'trash_bin', 'paper_ball', 'cup', 'shopping_bag']
```

> ✅ `nc: 7` 表示 7 个类别，请根据你的实际类别顺序填写 `names`。

---

## ✅ 四、训练代码：`train.py`

```python
# train.py
from ultralytics import YOLO

# 1. 加载官方预训练模型（推荐 yolov8n 或 yolov8s）
model = YOLO('models/yolov8s.pt')  # 可换为 yolov8n.pt, yolov8m.pt 等

# 2. 开始训练
results = model.train(
    data='datasets/falling_objects/falling_objects.yaml',  # 数据集配置
    epochs=200,               # 训练轮数
    imgsz=640,                # 输入尺寸
    batch=16,                 # 批次大小（根据显存调整，696张图可用16）
    name='falling_yolov8s_640',  # 实验名称
    device='cuda',            # 使用 GPU
    workers=8,                # 数据加载线程
    optimizer='AdamW',        # 优化器
    lr0=0.001,                # 初始学习率
    weight_decay=0.0005,
    momentum=0.937,
    patience=50,              # 早停：50轮无提升则停止
    project='runs/falling',   # 保存路径
    amp=True,                 # 混合精度训练（加快速度）
    close_mosaic=30,          # 关闭 Mosaic 数据增强的轮数
    cache='disk'              # 缓存图像到磁盘，加快训练
)

# 3. 打印结果
print("训练完成！")
print(results)
```

### 🚀 训练命令：
```bash
conda activate falling_yolo
python train.py
```

训练完成后，最佳模型保存在：
```
runs/falling/falling_yolov8s_640/weights/best.pt
```

---

## ✅ 五、推理代码：`detect.py`

```python
# detect.py
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/falling/falling_yolov8s_640/weights/best.pt')

conf_threshold = 0.3

# --- 1. 检测单张图片 ---
def detect_image():
    results = model.predict(
        source='datasets/falling_objects/images/test/img001.jpg',
        conf=conf_threshold,
        save=True,
        project='runs/detect',
        name='image_result'
    )
    print(f"图片检测完成，结果保存在 runs/detect/image_result/")

# --- 2. 检测整个文件夹 ---
def detect_folder():
    results = model.predict(
        source='datasets/falling_objects/images/test/',
        conf=conf_threshold,
        save=True,
        project='runs/detect',
        name='batch_result'
    )
    print(f"批量检测完成，结果保存在 runs/detect/batch_result/")

# --- 3. 实时摄像头检测 ---
def detect_camera():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf_threshold)
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow('Falling Object Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# --- 4. 检测视频文件 ---
def detect_video():
    results = model.predict(
        source='test_video.mp4',
        conf=conf_threshold,
        save=True,
        project='runs/detect',
        name='video_result'
    )

if __name__ == "__main__":
    detect_folder()  # 可切换为 detect_image(), detect_camera(), detect_video()
```

---

## ✅ 六、评估代码：`val.py`

评估模型在验证集上的性能（mAP, Precision, Recall 等）

```python
# val.py
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/falling/falling_yolov8s_640/weights/best.pt')

# 在验证集上评估
metrics = model.val(
    data='datasets/falling_objects/falling_objects.yaml',
    imgsz=640,
    batch=16,
    conf=0.001,
    iou=0.6,
    device='cuda'
)

# 打印关键指标
print("类别数量:", metrics.box.nc)
print("各类别 mAP50:")
for i, name in enumerate(metrics.names):
    print(f"  {name}: {metrics.box.map50_per_class[i]:.4f}")

print("\n整体性能:")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p.mean():.4f}")
print(f"Recall: {metrics.box.r.mean():.4f}")
print(f"F1 Score: {metrics.box.f1.mean():.4f}")

# 混淆矩阵等结果保存在 runs/val/falling_yolov8s_640/
```

---


