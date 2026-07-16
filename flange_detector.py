"""
法兰检测器（YOLOv8 专用）

本文件为新增模块，不修改现有代码。可与项目中的 ROS2Client_Camera / Box_Transfer_Map 配合使用，
也支持独立运行（传入图片路径或 numpy 数组）。

主要功能：
    - 加载法兰专用 YOLOv8 模型
    - 对单张图片做推理，返回检测框、置信度、中心点
    - 可选：基于 4 个螺栓孔拟合法兰精确中心和朝向角
    - 可选：从 ROS2Client_Camera 直接取图检测
    - 可选：简单角度估计（基于检测框 minAreaRect）
    - 可选：可视化绘制结果

用法示例：
    # 1. 独立检测图片（带螺栓孔位姿）
    from flange_detector import FlangeDetector
    det = FlangeDetector(model_path="yolo_flange.pt")
    result = det.detect_with_pose("flange_001.jpg")
    best = result.best_bbox()
    print(best["pose_center"], best["pose_angle_deg"])

    # 2. 与现有相机类配合
    from box_camera import ROS2Client_Camera
    camera = ROS2Client_Camera()
    camera.start_camera()
    result = det.detect_with_pose_from_camera(camera)
    camera.stop_camera()

    # 3. 取最佳目标并转 base 系坐标
    from box_transfer_map import Box_Transfer_Map
    transfer = Box_Transfer_Map()
    best = result.best_bbox()
    if best is not None and best.get("pose_center"):
        u, v = best["pose_center"]  # 用基于螺栓孔的精确中心
        depth = ...  # 从 depth 图取 (u,v) 处深度，单位 mm
        head_pos = camera.get_normalize_robot_pose()
        pts_base = transfer.calc_box_pos_in_base(
            [{"u": u, "v": v, "depth": depth}], head_pos
        )
        print("法兰在 base 系坐标:", pts_base[0])
"""

import math
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------------------------------------------------------
# 检测结果数据结构
# -----------------------------------------------------------------------------
class FlangeDetectionResult:
    """单次检测的封装结果。"""

    def __init__(
        self,
        image: Optional[np.ndarray] = None,
        bboxes: Optional[List[Dict[str, Any]]] = None,
        inference_ms: float = 0.0,
    ):
        self.image = image                      # 原始输入图片（BGR，可能为 None）
        # bbox 字段示例：
        # {
        #   "bbox": (x1,y1,x2,y2), "conf": float, "class_id": int,
        #   "center": (cx,cy), "area": int, "angle_deg": float|None,
        #   "holes": [(hx,hy), ...], "hole_count": int,
        #   "pose_center": (px,py)|None, "pose_angle_deg": float|None
        # }
        self.bboxes = bboxes or []
        self.inference_ms = inference_ms        # 推理耗时

    def __len__(self) -> int:
        return len(self.bboxes)

    def is_empty(self) -> bool:
        return len(self.bboxes) == 0

    def best_bbox(
        self,
        strategy: str = "largest",
        image_center: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        从所有检测结果中选出最佳目标。

        Args:
            strategy: 选择策略
                - "largest": 面积最大
                - "center":  距离图像中心最近
                - "conf":    置信度最高
            image_center: 图像中心点，strategy="center" 时使用；为 None 则使用 self.image 的中心。
        """
        if not self.bboxes:
            return None

        if strategy == "largest":
            return max(self.bboxes, key=lambda b: b["area"])
        elif strategy == "conf":
            return max(self.bboxes, key=lambda b: b["conf"])
        elif strategy == "center":
            if image_center is None and self.image is not None:
                h, w = self.image.shape[:2]
                image_center = (w // 2, h // 2)
            if image_center is None:
                raise ValueError("strategy='center' 需要提供 image_center 或输入图片")
            return min(
                self.bboxes,
                key=lambda b: (b["center"][0] - image_center[0]) ** 2
                + (b["center"][1] - image_center[1]) ** 2,
            )
        else:
            raise ValueError(f"未知选择策略: {strategy}")


# -----------------------------------------------------------------------------
# 法兰检测器
# -----------------------------------------------------------------------------
class FlangeDetector:
    """
    基于 YOLOv8 的法兰检测器。
    模型路径可替换，与项目现有 YoloJaka 解耦，避免改动旧代码。
    """

    def __init__(
        self,
        model_path: str = "yolo_flange.pt",
        conf: float = 0.5,
        iou: float = 0.45,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Args:
            model_path: YOLOv8 模型路径（.pt 或导出后的 .onnx/.engine 等）
            conf: 置信度阈值
            iou:  NMS IoU 阈值
            device: 推理设备，None 则自动选择（优先 cuda）
            verbose: 是否打印 ultralytics 内部日志
        """
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device or ("cuda:0" if self._cuda_available() else "cpu")
        self.verbose = verbose
        self.model = None
        self._load_model()

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _load_model(self) -> None:
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"法兰模型不存在: {self.model_path}\n"
                f"请先训练模型或放置预训练权重到该路径。"
            )
        self.model = YOLO(self.model_path)
        # 预热一次，避免首次推理耗时异常
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            self.model.predict(dummy, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        except Exception as e:
            print(f"[FlangeDetector] 模型预热失败（不影响后续使用）: {e}")

    # ------------------------------------------------------------------
    # 核心检测接口
    # ------------------------------------------------------------------
    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        estimate_angle: bool = True,
    ) -> FlangeDetectionResult:
        """
        对单张图片进行法兰检测。

        Args:
            image: 图片路径或 BGR numpy 数组
            conf:  临时覆盖置信度阈值
            iou:   临时覆盖 NMS 阈值
            estimate_angle: 是否基于检测框估计朝向角（minAreaRect）

        Returns:
            FlangeDetectionResult
        """
        _conf = self.conf if conf is None else conf
        _iou = self.iou if iou is None else iou

        if isinstance(image, (str, Path)):
            img_array = cv2.imread(str(image))
            if img_array is None:
                raise ValueError(f"无法读取图片: {image}")
        elif isinstance(image, np.ndarray):
            img_array = image.copy()
        else:
            raise TypeError("image 必须是路径字符串或 numpy 数组")

        t0 = time.time()
        results = self.model.predict(
            img_array,
            conf=_conf,
            iou=_iou,
            device=self.device,
            verbose=self.verbose,
        )
        inference_ms = (time.time() - t0) * 1000.0

        bboxes = []
        result = results[0]
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy()

            for box, conf_score, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                angle = None
                if estimate_angle:
                    angle = self._estimate_angle_from_bbox(img_array, (x1, y1, x2, y2))

                bboxes.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(conf_score),
                    "class_id": int(cls_id),
                    "center": (cx, cy),
                    "area": (x2 - x1) * (y2 - y1),
                    "angle_deg": angle,  # 可能为 None
                })

        return FlangeDetectionResult(image=img_array, bboxes=bboxes, inference_ms=inference_ms)

    def detect_from_camera(
        self,
        camera,  # ROS2Client_Camera 实例，使用 forward ref 避免强依赖
        use_rgbd_pair: bool = True,
        max_delta_ms: float = 80.0,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> FlangeDetectionResult:
        """
        从项目现有的 ROS2Client_Camera 取图并检测。

        Args:
            camera: ROS2Client_Camera 实例
            use_rgbd_pair: 是否获取时间对齐的 RGB-D pair；False 则只取 color
            max_delta_ms: RGB-D pair 最大允许时间差
            conf/iou: 临时覆盖阈值
        """
        if use_rgbd_pair:
            pair = camera.get_latest_rgbd_pair(max_delta_ms=max_delta_ms)
            if pair is None:
                raise RuntimeError("无法获取 RGB-D pair，相机可能尚未就绪")
            color_frame = pair["color"]
        else:
            color_frame = camera.get_latest_color_image()

        if color_frame is None:
            raise RuntimeError("无法获取彩色图像")

        # ROS2 帧通常是 BGR，直接传入即可
        return self.detect(color_frame, conf=conf, iou=iou)

    def detect_with_pose(
        self,
        image: Union[str, Path, np.ndarray],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        min_holes: int = 3,
        max_holes: int = 8,
    ) -> FlangeDetectionResult:
        """
        检测法兰并在每个 bbox 内进一步检测螺栓孔，输出更精确的中心和朝向角。

        Args:
            image: 图片路径或 BGR numpy 数组
            conf/iou: 临时覆盖阈值
            min_holes: 至少检测到几个孔才计算位姿
            max_holes: 最多保留几个候选孔

        Returns:
            FlangeDetectionResult，其中每个 bbox 额外包含：
                - "holes": 检测到的螺栓孔中心列表（全图像素坐标）
                - "hole_count": 螺栓孔数量
                - "pose_center": 基于螺栓孔拟合的法兰中心（全图像素坐标）
                - "pose_angle_deg": 基于螺栓孔的朝向角（度，-90~90）
        """
        result = self.detect(image, conf=conf, iou=iou, estimate_angle=False)
        if result.is_empty():
            return result

        for item in result.bboxes:
            x1, y1, x2, y2 = item["bbox"]
            holes = self._detect_holes_in_bbox(
                result.image, (x1, y1, x2, y2), min_holes=min_holes, max_holes=max_holes
            )
            item["holes"] = holes
            item["hole_count"] = len(holes)

            if len(holes) >= min_holes:
                center, angle = self._compute_pose_from_holes(holes)
                item["pose_center"] = center
                item["pose_angle_deg"] = angle
            else:
                item["pose_center"] = None
                item["pose_angle_deg"] = None

        return result

    def detect_with_pose_from_camera(
        self,
        camera,
        use_rgbd_pair: bool = True,
        max_delta_ms: float = 80.0,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        min_holes: int = 3,
        max_holes: int = 8,
    ) -> FlangeDetectionResult:
        """
        从 ROS2Client_Camera 取图，并做法兰 + 螺栓孔位姿检测。
        参数同 detect_with_pose / detect_from_camera。
        """
        if use_rgbd_pair:
            pair = camera.get_latest_rgbd_pair(max_delta_ms=max_delta_ms)
            if pair is None:
                raise RuntimeError("无法获取 RGB-D pair，相机可能尚未就绪")
            color_frame = pair["color"]
        else:
            color_frame = camera.get_latest_color_image()

        if color_frame is None:
            raise RuntimeError("无法获取彩色图像")

        return self.detect_with_pose(
            color_frame,
            conf=conf,
            iou=iou,
            min_holes=min_holes,
            max_holes=max_holes,
        )

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_holes_in_bbox(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        min_holes: int = 3,
        max_holes: int = 8,
    ) -> List[Tuple[int, int]]:
        """
        在法兰检测框内检测螺栓孔，返回全图像素坐标列表。

        策略：
            1. 裁剪 ROI 并转灰度
            2. 高斯模糊 + OTSU 二值化
            3. 形态学闭运算连接碎片
            4. 轮廓检测，按圆度/面积比筛选出孔
            5. 按面积排序，取前 max_holes 个
        """
        x1, y1, x2, y2 = bbox
        # 稍微扩大 ROI，避免孔刚好在边缘被截断
        margin = 5
        h_img, w_img = image.shape[:2]
        x1m = max(0, x1 - margin)
        y1m = max(0, y1 - margin)
        x2m = min(w_img, x2 + margin)
        y2m = min(h_img, y2 + margin)

        roi = image[y1m:y2m, x1m:x2m]
        if roi.size == 0:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 假设孔比背景暗；若现场孔是亮的，可改为 THRESH_BINARY
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 形态学闭运算，填补孔内细小缺口
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = roi.shape[0] * roi.shape[1]

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20 or area > roi_area * 0.25:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-6:
                continue

            circularity = 4 * math.pi * area / (perimeter * perimeter)
            # 圆度接近 1；适当放宽到 0.5 以容忍透视变形
            if circularity < 0.5:
                continue

            # 计算轮廓中心
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"]) + x1m
            cy = int(M["m01"] / M["m00"]) + y1m
            candidates.append((area, circularity, (cx, cy)))

        # 按面积从大到小排序，取前 max_holes 个（孔通常比噪点大）
        candidates.sort(key=lambda x: x[0], reverse=True)
        holes = [c[2] for c in candidates[:max_holes]]

        # 如果孔太多，尝试只保留与最大孔面积相近的（避免误检）
        if len(holes) > min_holes and candidates:
            max_area = candidates[0][0]
            filtered = [c[2] for c in candidates if c[0] > max_area * 0.15]
            if len(filtered) >= min_holes:
                holes = filtered[:max_holes]

        return holes

    @staticmethod
    def _compute_pose_from_holes(
        holes: List[Tuple[int, int]],
    ) -> Tuple[Tuple[int, int], Optional[float]]:
        """
        根据螺栓孔中心计算法兰中心和朝向角。

        中心：所有孔中心的均值。
        角度：对孔中心做 minAreaRect，返回长边与水平轴夹角（度，-90~90）。
        """
        pts = np.array(holes, dtype=np.float32)
        center = (int(round(pts[:, 0].mean())), int(round(pts[:, 1].mean())))

        if len(holes) < 2:
            return center, None

        rect = cv2.minAreaRect(pts)
        angle = rect[2]
        if angle < -90:
            angle += 180
        elif angle > 90:
            angle -= 180

        return center, float(angle)

    @staticmethod
    def _estimate_angle_from_bbox(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        use_contour: bool = True,
    ) -> Optional[float]:
        """
        粗略估计法兰在图像中的朝向角（单位：度，-90~90）。
        当前实现基于检测框的 minAreaRect；如需更精确角度，建议后续改为圆孔拟合。
        """
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        if use_contour:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest)
                angle = rect[2]
                # 将角度统一到 -90~90 范围，长边方向
                if angle < -90:
                    angle += 180
                elif angle > 90:
                    angle -= 180
                return float(angle)

        # 退化到 bbox 长宽比判断
        w = x2 - x1
        h = y2 - y1
        if h == 0:
            return 0.0
        return 0.0 if w >= h else 90.0

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        result: FlangeDetectionResult,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_conf: bool = True,
        show_angle: bool = True,
        show_holes: bool = True,
        show_pose: bool = True,
    ) -> np.ndarray:
        """在图片上绘制检测结果，返回绘制后的 BGR 图片。"""
        canvas = image.copy()
        for idx, item in enumerate(result.bboxes):
            x1, y1, x2, y2 = item["bbox"]
            cx, cy = item["center"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
            cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)

            # 绘制螺栓孔
            if show_holes and "holes" in item:
                for hx, hy in item["holes"]:
                    cv2.circle(canvas, (hx, hy), 5, (255, 0, 0), -1)

            # 绘制基于孔的精确中心和朝向
            if show_pose and item.get("pose_center") is not None:
                px, py = item["pose_center"]
                cv2.circle(canvas, (px, py), 6, (0, 255, 255), -1)
                angle_rad = math.radians(item.get("pose_angle_deg", 0) or 0)
                line_len = 60
                ex = int(px + line_len * math.cos(angle_rad))
                ey = int(py + line_len * math.sin(angle_rad))
                cv2.line(canvas, (px, py), (ex, ey), (0, 255, 255), 2)

            texts = [f"#{idx} flange"]
            if show_conf:
                texts.append(f"conf={item['conf']:.2f}")
            if show_angle and item.get("angle_deg") is not None:
                texts.append(f"bbox_angle={item['angle_deg']:.1f}")
            if show_pose and item.get("pose_angle_deg") is not None:
                texts.append(f"pose_angle={item['pose_angle_deg']:.1f}")
            text = " | ".join(texts)
            cv2.putText(
                canvas, text, (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness,
            )
        return canvas

    def save_result_image(
        self,
        result: FlangeDetectionResult,
        output_path: str,
        **draw_kwargs,
    ) -> None:
        """绘制并保存结果图。"""
        if result.image is None:
            raise ValueError("result 中未包含原始图片，无法绘制")
        drawn = self.draw_detections(result.image, result, **draw_kwargs)
        cv2.imwrite(output_path, drawn)


# -----------------------------------------------------------------------------
# 简单测试入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python flange_detector.py <图片路径> [模型路径]")
        sys.exit(1)

    img_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "yolo_flange.pt"

    detector = FlangeDetector(model_path=model_path)
    result = detector.detect_with_pose(img_path)

    print(f"推理耗时: {result.inference_ms:.1f} ms")
    print(f"检测到 {len(result)} 个法兰")
    for item in result.bboxes:
        print(f"  bbox={item['bbox']}, conf={item['conf']:.3f}")
        print(f"  粗略中心={item['center']}, 孔数={item.get('hole_count', 0)}")
        print(f"  精确中心={item.get('pose_center')}, 朝向角={item.get('pose_angle_deg')}")

    if not result.is_empty():
        out_path = "flange_detect_result.jpg"
        detector.save_result_image(result, out_path, show_pose=True, show_holes=True)
        print(f"结果图已保存: {out_path}")
