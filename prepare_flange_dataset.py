"""
法兰 YOLOv8 训练数据集准备脚本

功能：
    1. 从原始图片目录构建 YOLOv8 标准数据集结构。
    2. 随机划分为 train / val（默认 80/20）。
    3. 生成 data.yaml。
    4. 支持 LabelImg XML / COCO JSON 格式一键转换为 YOLO txt。
    5. 提供简易 OpenCV 画框标注工具（无标签时）。
    6. 校验标签与图片是否一一对应。

用法示例：
    # 仅整理图片并生成 data.yaml（假设已用 LabelImg 导出 YOLO txt 并放在同目录）
    python prepare_flange_dataset.py --images_dir ./raw_flange_images --output_dir ./flange_dataset

    # 从 LabelImg XML 转换
    python prepare_flange_dataset.py --images_dir ./raw_flange_images \
                                     --xml_dir ./raw_flange_xml \
                                     --output_dir ./flange_dataset

    # 交互式画框标注（无预标注时）
    python prepare_flange_dataset.py --images_dir ./raw_flange_images \
                                     --output_dir ./flange_dataset \
                                     --annotate

训练命令（生成后终端执行）：
    yolo detect train data=./flange_dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640
"""

import argparse
import os
import shutil
import random
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np


# -----------------------------------------------------------------------------
# 默认配置
# -----------------------------------------------------------------------------
DEFAULT_CLASS_NAME = "flange"           # 可根据需要改成"一体法兰"等，但 YOLO 建议英文/无空格
DEFAULT_TRAIN_RATIO = 0.8             # 训练集比例
DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


# -----------------------------------------------------------------------------
# 路径工具
# -----------------------------------------------------------------------------
def list_image_files(image_dir: str) -> List[Path]:
    """列出目录下所有常见格式图片。"""
    image_dir = Path(image_dir)
    files = []
    for ext in DEFAULT_IMAGE_EXTS:
        files.extend(image_dir.glob(f"*{ext}"))
        files.extend(image_dir.glob(f"*{ext.upper()}"))
    # 去重并排序
    return sorted(list(set(files)))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# -----------------------------------------------------------------------------
# 标注格式转换
# -----------------------------------------------------------------------------
def xml_to_yolo(xml_path: Path, img_w: int, img_h: int) -> List[List[float]]:
    """
    将 LabelImg PascalVOC XML 转换为 YOLO 格式：
    [class_id, x_center, y_center, width, height]（均已归一化到 0~1）
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        x_min = float(bndbox.find("xmin").text)
        y_min = float(bndbox.find("ymin").text)
        x_max = float(bndbox.find("xmax").text)
        y_max = float(bndbox.find("ymax").text)

        x_center = (x_min + x_max) / 2.0 / img_w
        y_center = (y_min + y_max) / 2.0 / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        # 单类别法兰，class_id 固定为 0
        labels.append([0, x_center, y_center, w, h])

    return labels


def coco_to_yolo(coco_json_path: Path, output_label_dir: Path) -> None:
    """
    将 COCO JSON 转换为多张 YOLO txt。
    要求 JSON 中 image 文件名与图片目录中的文件名一致。
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_ids = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}
    img_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    img_id_to_size = {img["id"]: (img["width"], img["height"]) for img in coco["images"]}

    annotations_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    for img_id, anns in annotations_by_image.items():
        img_name = img_id_to_name.get(img_id)
        if not img_name:
            continue
        img_w, img_h = img_id_to_size[img_id]
        stem = Path(img_name).stem
        txt_path = output_label_dir / f"{stem}.txt"
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            cls_id = cat_ids.get(ann["category_id"], 0)
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
        txt_path.write_text("\n".join(lines), encoding="utf-8")


def write_yolo_txt(txt_path: Path, labels: List[List[float]]) -> None:
    """将 [[class_id, x, y, w, h], ...] 写入 YOLO txt。"""
    lines = [f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}" for l in labels]
    txt_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# 交互式画框标注
# -----------------------------------------------------------------------------
class BBoxAnnotator:
    """极简 OpenCV 画框工具：鼠标左键拖拽画框，空格保存，q 跳过，Esc 退出。"""

    def __init__(self, window_name: str = "Flange BBox Annotator"):
        self.window_name = window_name
        self.drawing = False
        self.ix = self.iy = 0
        self.bboxes: List[Tuple[int, int, int, int]] = []
        self.current_img: Optional[np.ndarray] = None
        self.canvas: Optional[np.ndarray] = None

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.canvas = self.current_img.copy()
            cv2.rectangle(self.canvas, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x0, y0 = min(self.ix, x), min(self.iy, y)
            x1, y1 = max(self.ix, x), max(self.iy, y)
            self.bboxes.append((x0, y0, x1, y1))
            self.canvas = self.current_img.copy()
            for bbox in self.bboxes:
                cv2.rectangle(self.canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    def annotate_image(self, image_path: Path) -> List[Tuple[int, int, int, int]]:
        self.current_img = cv2.imread(str(image_path))
        if self.current_img is None:
            print(f"[警告] 无法读取图片: {image_path}")
            return []
        self.canvas = self.current_img.copy()
        self.bboxes = []

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print(f"\n标注图片: {image_path.name}")
        print("操作: 左键拖拽画框 | d 删除上一个框 | 空格/Enter 保存 | q 跳过 | Esc 退出")

        while True:
            cv2.imshow(self.window_name, self.canvas)
            key = cv2.waitKey(50) & 0xFF
            if key == 27:  # Esc
                self.bboxes = []
                cv2.destroyWindow(self.window_name)
                raise KeyboardInterrupt("用户退出标注")
            elif key in (13, 32):  # Enter / Space
                break
            elif key == ord("q"):
                self.bboxes = []
                break
            elif key == ord("d") and self.bboxes:
                self.bboxes.pop()
                self.canvas = self.current_img.copy()
                for bbox in self.bboxes:
                    cv2.rectangle(self.canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        cv2.destroyWindow(self.window_name)
        return self.bboxes


def run_interactive_annotation(image_paths: List[Path], label_dir: Path) -> None:
    """对未标注图片逐个进行交互式画框。"""
    annotator = BBoxAnnotator()
    for img_path in image_paths:
        txt_path = label_dir / f"{img_path.stem}.txt"
        if txt_path.exists():
            continue
        try:
            bboxes = annotator.annotate_image(img_path)
        except KeyboardInterrupt:
            print("\n退出标注。")
            break

        if not bboxes:
            continue

        h, w = cv2.imread(str(img_path)).shape[:2]
        labels = []
        for x0, y0, x1, y1 in bboxes:
            x_center = (x0 + x1) / 2.0 / w
            y_center = (y0 + y1) / 2.0 / h
            bw = (x1 - x0) / w
            bh = (y1 - y0) / h
            labels.append([0, x_center, y_center, bw, bh])
        write_yolo_txt(txt_path, labels)
        print(f"已保存标注: {txt_path}")


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def build_dataset(
    images_dir: str,
    output_dir: str,
    xml_dir: Optional[str] = None,
    coco_json: Optional[str] = None,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    annotate: bool = False,
    class_name: str = DEFAULT_CLASS_NAME,
    seed: int = 42,
) -> None:
    """
    构建 YOLOv8 法兰数据集。
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")

    # 1. 准备目录结构
    train_img_dir = ensure_dir(output_dir / "images" / "train")
    val_img_dir = ensure_dir(output_dir / "images" / "val")
    train_lbl_dir = ensure_dir(output_dir / "labels" / "train")
    val_lbl_dir = ensure_dir(output_dir / "labels" / "val")

    image_paths = list_image_files(images_dir)
    if not image_paths:
        raise ValueError(f"在 {images_dir} 中未找到图片文件")

    print(f"共找到 {len(image_paths)} 张图片")

    # 2. 划分训练集/验证集
    random.seed(seed)
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * train_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    print(f"训练集: {len(train_paths)} 张, 验证集: {len(val_paths)} 张")

    # 3. 复制图片并处理标签
    def process_split(src_paths: List[Path], img_dst: Path, lbl_dst: Path):
        for img_path in src_paths:
            # 复制图片
            dst_img = img_dst / img_path.name
            shutil.copy2(img_path, dst_img)

            stem = img_path.stem
            txt_path = lbl_dst / f"{stem}.txt"

            # 优先使用 XML 转换
            if xml_dir:
                xml_path = Path(xml_dir) / f"{stem}.xml"
                if xml_path.exists():
                    img = cv2.imread(str(img_path))
                    h, w = img.shape[:2]
                    labels = xml_to_yolo(xml_path, w, h)
                    write_yolo_txt(txt_path, labels)
                    continue

            # 其次查找同目录下的 YOLO txt
            existing_txt = img_path.parent / f"{stem}.txt"
            if existing_txt.exists():
                shutil.copy2(existing_txt, txt_path)
                continue

            # 否则留空（后续可交互标注）
            txt_path.write_text("", encoding="utf-8")

    process_split(train_paths, train_img_dir, train_lbl_dir)
    process_split(val_paths, val_img_dir, val_lbl_dir)

    # 4. 如果是 COCO JSON，覆盖生成全部标签
    if coco_json:
        print("正在从 COCO JSON 转换标签...")
        coco_to_yolo(Path(coco_json), output_dir / "labels" / "train")
        coco_to_yolo(Path(coco_json), output_dir / "labels" / "val")

    # 5. 交互式标注（仅对训练集中空标签的图片）
    if annotate:
        empty_train = [
            p for p in train_paths
            if not (train_lbl_dir / f"{p.stem}.txt").exists()
            or (train_lbl_dir / f"{p.stem}.txt").read_text(encoding="utf-8").strip() == ""
        ]
        print(f"训练集中有 {len(empty_train)} 张图片待标注")
        run_interactive_annotation(empty_train, train_lbl_dir)

    # 6. 生成 data.yaml
    yaml_path = output_dir / "data.yaml"
    yaml_content = f"""path: {output_dir.resolve().as_posix()}  # 数据集根目录（绝对路径）
train: images/train
val: images/val

nc: 1
names:
  0: {class_name}
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"已生成: {yaml_path}")

    # 7. 校验
    validate_dataset(output_dir)

    print("\n===== 训练命令示例 =====")
    print(f"yolo detect train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640 batch=8")
    print("可选模型: yolov8n.pt / yolov8s.pt / yolov8m.pt")


def validate_dataset(output_dir: Path) -> None:
    """校验图片与标签是否一一对应，并统计标签数量。"""
    splits = [("train", output_dir / "images" / "train", output_dir / "labels" / "train"),
              ("val", output_dir / "images" / "val", output_dir / "labels" / "val")]

    for name, img_dir, lbl_dir in splits:
        if not img_dir.exists():
            continue
        img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in DEFAULT_IMAGE_EXTS}
        lbl_stems = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"}

        missing_labels = img_stems - lbl_stems
        missing_images = lbl_stems - img_stems
        empty_labels = [s for s in lbl_stems if (lbl_dir / f"{s}.txt").read_text(encoding="utf-8").strip() == ""]

        print(f"\n[{name}] 校验结果:")
        print(f"  图片数: {len(img_stems)}, 标签数: {len(lbl_stems)}")
        if missing_labels:
            print(f"  [警告] 缺少标签的图片: {len(missing_labels)} 张")
        if missing_images:
            print(f"  [警告] 缺少对应图片的标签: {len(missing_images)} 个")
        if empty_labels:
            print(f"  [警告] 空标签文件: {len(empty_labels)} 个")
        if not missing_labels and not missing_images and not empty_labels:
            print("  校验通过")


# -----------------------------------------------------------------------------
# 命令行入口
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="准备法兰 YOLOv8 训练数据集")
    parser.add_argument("--images_dir", required=True, help="原始图片目录")
    parser.add_argument("--output_dir", required=True, help="输出数据集目录")
    parser.add_argument("--xml_dir", default=None, help="LabelImg XML 标注目录（可选）")
    parser.add_argument("--coco_json", default=None, help="COCO JSON 标注文件（可选）")
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="训练集比例")
    parser.add_argument("--annotate", action="store_true", help="对无标签图片进行交互式画框")
    parser.add_argument("--class_name", default=DEFAULT_CLASS_NAME, help="类别名称")
    parser.add_argument("--seed", type=int, default=42, help="随机划分种子")

    args = parser.parse_args()
    build_dataset(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        xml_dir=args.xml_dir,
        coco_json=args.coco_json,
        train_ratio=args.train_ratio,
        annotate=args.annotate,
        class_name=args.class_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
    