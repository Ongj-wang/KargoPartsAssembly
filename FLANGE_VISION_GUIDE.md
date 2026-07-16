# 法兰视觉检测模块技术文档

> 文档对应文件：
> - [`prepare_flange_dataset.py`](./prepare_flange_dataset.py)
> - [`flange_detector.py`](./flange_detector.py)
>
> 本文档用于技术交接、后续调试及训练流程参考。

---

## 1. 模块概述

本模块为项目新增的视觉子系统，**不修改任何现有代码**，用于实现“法兰检测 + 位姿估计”。

| 文件 | 职责 |
|------|------|
| `prepare_flange_dataset.py` | 准备 YOLOv8 训练数据集：目录结构生成、训练/验证集划分、标注格式转换、交互式画框、生成 `data.yaml` |
| `flange_detector.py` | 法兰推理检测器：加载 YOLOv8 模型，输出法兰 bbox；进一步检测螺栓孔，输出精确中心与朝向角 |

整体链路：

```
原始图片 → prepare_flange_dataset.py → YOLO 训练 → yolo_flange.pt
                                                    ↓
相机/图片 → flange_detector.py → 法兰 bbox + 螺栓孔 → 像素中心/角度 → Box_Transfer_Map → base 系位姿
```

---

## 2. prepare_flange_dataset.py

### 2.1 功能说明

该脚本负责把原始法兰图片整理成 YOLOv8 标准训练格式，支持三种数据来源：

1. **已有 YOLO txt 标注**（与图片同目录）
2. **已有 LabelImg XML 标注**
3. **无标注图片**，提供简易 OpenCV 交互式画框工具

### 2.2 依赖

- Python 3.8+
- `opencv-python`
- `numpy`

无需 ROS2、CUDA 或 YOLO 环境即可运行。

### 2.3 命令行参数

```text
--images_dir    原始图片目录（必填）
--output_dir    输出数据集目录（必填）
--xml_dir       LabelImg XML 标注目录（可选）
--coco_json     COCO JSON 标注文件（可选）
--train_ratio   训练集比例，默认 0.8
--annotate      启用交互式画框（无标签时使用）
--class_name    类别名称，默认 "flange"
--seed          随机划分种子，默认 42
```

### 2.4 使用示例

#### 场景 A：已有 LabelImg XML

```powershell
python prepare_flange_dataset.py `
  --images_dir ./raw_flange_images `
  --xml_dir ./raw_flange_xml `
  --output_dir ./flange_dataset
```

#### 场景 B：已有 YOLO txt（与图片同目录）

```powershell
python prepare_flange_dataset.py `
  --images_dir ./raw_flange_images `
  --output_dir ./flange_dataset
```

#### 场景 C：无标注，交互式画框

```powershell
python prepare_flange_dataset.py `
  --images_dir ./raw_flange_images `
  --output_dir ./flange_dataset `
  --annotate
```

交互快捷键：

| 按键 | 动作 |
|------|------|
| 左键拖拽 | 画框 |
| `d` | 删除上一个框 |
| 空格 / Enter | 保存当前图片标注 |
| `q` | 跳过当前图片 |
| Esc | 退出标注 |

### 2.5 输出目录结构

```
flange_dataset/
├── images/
│   ├── train/           # 训练图片
│   └── val/             # 验证图片
├── labels/
│   ├── train/           # YOLO 格式标签：class x_center y_center width height
│   └── val/
└── data.yaml            # YOLO 训练配置文件
```

`data.yaml` 示例：

```yaml
path: /absolute/path/to/flange_dataset
train: images/train
val: images/val

nc: 1
names:
  0: flange
```

### 2.6 标签格式

每个 `.txt` 文件对应一张图片，每行一个目标：

```text
0 0.5123 0.4821 0.2156 0.1987
```

含义：`class_id x_center y_center width height`，均为相对于图片宽高的 0~1 归一化值。

### 2.7 校验逻辑

脚本最后会执行 `validate_dataset()`，检查：

- 图片数量与标签数量是否一致
- 是否存在无标签的图片
- 是否存在无对应图片的标签
- 是否存在空标签文件

### 2.8 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 提示“未找到图片” | 目录错误或图片格式不在支持列表 | 检查路径，支持 jpg/jpeg/png/bmp/tiff/webp |
| 训练时提示标签为空 | 空标签文件未处理 | 使用 `--annotate` 补标注，或删除空标签对应图片 |
| XML 转换后角度不对 | XML 中 x/y 顺序错误 | 确认 LabelImg 导出的是 PascalVOC 格式 |

---

## 3. flange_detector.py

### 3.1 功能说明

基于 YOLOv8 的法兰检测器，提供两级输出：

1. **基础检测** `detect()`：输出法兰 bbox、置信度、粗略中心、bbox 角度。
2. **位姿检测** `detect_with_pose()`：在 bbox 内进一步用传统 CV 检测螺栓孔，输出更精确的中心和朝向角。

同时提供与项目现有 `ROS2Client_Camera` 的对接接口。

### 3.2 依赖

- Python 3.8+
- `opencv-python`
- `numpy`
- `ultralytics`（YOLOv8）
- `torch`（YOLOv8 自动依赖）

### 3.3 类说明

#### `FlangeDetectionResult`

封装单次检测结果。

| 属性 | 类型 | 说明 |
|------|------|------|
| `image` | `np.ndarray \| None` | 原始输入图片（BGR） |
| `bboxes` | `List[Dict]` | 检测目标列表 |
| `inference_ms` | `float` | YOLO 推理耗时（毫秒） |

每个 `bbox` 字段示例：

```python
{
    "bbox": (x1, y1, x2, y2),          # 像素坐标，整数
    "conf": 0.92,                       # 置信度
    "class_id": 0,                      # 类别 ID
    "center": (cx, cy),                 # bbox 中心（粗略）
    "area": 12345,                      # bbox 面积
    "angle_deg": 12.5,                  # 仅 detect() 启用时：bbox 角度
    "holes": [(hx, hy), ...],           # 螺栓孔中心（detect_with_pose）
    "hole_count": 4,                    # 检测到的孔数
    "pose_center": (px, py),            # 基于螺栓孔的精确中心
    "pose_angle_deg": 12.3              # 基于螺栓孔的朝向角
}
```

方法：

- `best_bbox(strategy="largest", image_center=None)`：选出最佳目标
  - `"largest"`：面积最大
  - `"conf"`：置信度最高
  - `"center"`：距离图像中心最近

#### `FlangeDetector`

主要检测类。

##### 初始化

```python
det = FlangeDetector(
    model_path="yolo_flange.pt",  # YOLO 权重路径
    conf=0.5,                      # 置信度阈值
    iou=0.45,                      # NMS IoU 阈值
    device=None,                   # 推理设备，None 自动选 cuda/cpu
    verbose=False                  # 是否打印 ultralytics 内部日志
)
```

##### 核心方法

| 方法 | 说明 |
|------|------|
| `detect(image)` | 基础法兰检测，返回 bbox、conf、center、angle_deg |
| `detect_from_camera(camera)` | 从 `ROS2Client_Camera` 取图并做基础检测 |
| `detect_with_pose(image)` | 检测法兰 + 螺栓孔，返回 pose_center、pose_angle_deg |
| `detect_with_pose_from_camera(camera)` | 从相机取图并做位姿检测 |
| `draw_detections(image, result, ...)` | 绘制检测/位姿结果 |
| `save_result_image(result, output_path, ...)` | 保存绘制结果图 |

### 3.4 使用示例

#### 独立检测图片（推荐）

```python
from flange_detector import FlangeDetector

det = FlangeDetector(model_path="yolo_flange.pt")
result = det.detect_with_pose("flange_001.jpg")

best = result.best_bbox(strategy="largest")
if best and best.get("pose_center"):
    print(f"精确中心: {best['pose_center']}")
    print(f"朝向角: {best['pose_angle_deg']:.1f}°")
    print(f"螺栓孔数: {best['hole_count']}")

# 保存可视化结果
det.save_result_image(result, "result.jpg", show_pose=True, show_holes=True)
```

#### 与现有相机配合

```python
from flange_detector import FlangeDetector
from box_camera import ROS2Client_Camera
from box_transfer_map import Box_Transfer_Map

camera = ROS2Client_Camera()
camera.start_camera()

det = FlangeDetector(model_path="yolo_flange.pt")
result = det.detect_with_pose_from_camera(camera)

best = result.best_bbox()
if best and best.get("pose_center"):
    u, v = best["pose_center"]
    pair = camera.get_latest_rgbd_pair()
    depth_mm = pair["depth"][v, u]
    head_pos = camera.get_normalize_robot_pose()

    transfer = Box_Transfer_Map()
    pts_base = transfer.calc_box_pos_in_base(
        [{"u": u, "v": v, "depth": depth_mm}], head_pos
    )
    print("法兰 base 系坐标:", pts_base[0])

camera.stop_camera()
```

### 3.5 螺栓孔检测原理

`detect_with_pose()` 内部调用 `_detect_holes_in_bbox()`：

1. 以 YOLO 检测框为 ROI，外扩 5px 裁剪。
2. 灰度化 + 高斯模糊。
3. OTSU 自动二值化（默认孔比背景暗，若现场相反需改阈值方向）。
4. 形态学闭运算（5×5 椭圆核）填补孔内缺口。
5. 轮廓检测，筛选条件：
   - 面积：`20 < area < ROI_AREA × 0.25`
   - 周长 > 0
   - 圆度 `4π·area / perimeter² ≥ 0.5`
6. 按面积排序，取前 `max_holes` 个。
7. 对孔中心做 `minAreaRect` 得到主轴角度。

### 3.6 可视化说明

`draw_detections()` 绘制内容：

- 绿色矩形：YOLO 法兰 bbox
- 红色圆点：bbox 中心
- 蓝色圆点：检测到的螺栓孔
- 黄色圆点 + 黄色射线：基于螺栓孔的精确中心与朝向角

可控制开关：

```python
drawn = FlangeDetector.draw_detections(
    image,
    result,
    show_conf=True,      # 显示置信度
    show_angle=True,     # 显示 bbox 角度
    show_holes=True,     # 显示螺栓孔
    show_pose=True,      # 显示精确中心与朝向
)
```

### 3.7 调试参数

在 `_detect_holes_in_bbox()` 中，以下参数需根据现场图像微调：

| 参数 | 默认值 | 调试建议 |
|------|--------|---------|
| `min_holes` | 3 | 至少检测到几个孔才算位姿有效；完整法兰有 4 孔，可设为 3 容忍遮挡 |
| `max_holes` | 8 | 最多保留几个候选，防止噪点干扰 |
| 圆度阈值 | 0.5 | 孔变形大（透视/倾斜）可调低；噪点多则调高 |
| 面积下限 | 20 | 孔在图像中很小时可调低 |
| 二值化方向 | `THRESH_BINARY_INV` | 若孔是亮的，改为 `THRESH_BINARY` |

### 3.8 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 孔检测为空 | 二值化方向错误 / 孔被反光覆盖 | 改为 `THRESH_BINARY` 或调整光照 |
| 孔数多于 4 | 噪点被误检 | 提高圆度阈值或调小 `max_holes` |
| 角度跳变 90° | `minAreaRect` 长宽接近正方形 | 4 孔分布接近正方形时正常，可结合机械约束做后处理 |
| YOLO 检测不到法兰 | 模型未训练或 conf 过高 | 降低 `conf`，确认模型路径正确 |
| Windows 上无法导入 | `box_camera` 依赖 ROS2 | 独立使用时只 `from flange_detector import FlangeDetector`，不要导入 `box_camera` |

---

## 4. 完整训练与部署流程

### 4.1 数据采集

- 拍摄 50~200 张法兰图片，覆盖不同：
  - 摆放角度
  - 距离 / 缩放
  - 光照条件
  - 部分遮挡情况

### 4.2 数据集准备

```powershell
python prepare_flange_dataset.py `
  --images_dir ./raw_flange_images `
  --output_dir ./flange_dataset `
  --annotate
```

### 4.3 训练 YOLO 模型

```powershell
yolo detect train data=./flange_dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=8
```

- 模型可选：`yolov8n.pt`（快） / `yolov8s.pt`（平衡） / `yolov8m.pt`（准）
- 训练完成后最佳权重默认位于：`runs/detect/train/weights/best.pt`

### 4.4 部署检测

```python
from flange_detector import FlangeDetector

det = FlangeDetector(model_path="runs/detect/train/weights/best.pt")
result = det.detect_with_pose("test.jpg")
```

---

## 5. 与项目现有模块的关系

| 本项目模块 | 新增模块 | 说明 |
|-----------|---------|------|
| `box_camera.py` | 被 `flange_detector.py` 可选导入 | 提供 ROS2 相机取图能力 |
| `box_transfer_map.py` | 被业务代码导入使用 | 把像素 + 深度转成 base/世界系坐标 |
| `box_yolov8.py` | 不被新增模块导入 | 旧料箱检测器，保持独立 |
| `box_dashscope.py` | 不被新增模块导入 | 旧 VLM 检测器，保持独立 |

**零侵入原则**：新增两个文件均不修改现有代码，旧料箱业务不受影响。

---

## 6. 后续优化方向

1. **关键点模型**：若孔检测不稳定，可训练 YOLOv8-pose，直接输出 4 个螺栓孔关键点。
2. **点云配准**：若需要完整 6D 位姿（含倾斜），可复用 `box_o3d.py` 的 ICP 模板匹配。
3. **角度后处理**：对 4 孔正方形分布导致的 90° 歧义，可引入机械臂抓取姿态约束。
4. **自动标注**：可基于当前 `detect_with_pose()` 对已有图片预标注，再人工校验，减少标注工作量。
