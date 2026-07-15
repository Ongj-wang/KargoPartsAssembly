"""
ROS2Client — bridges ROS2 topics/actions/services to the agent framework.

Camera topics subscribed:
  /camera_head/color/image_raw   (sensor_msgs/Image, BGR8 or RGB8)
  /camera_head/depth/image_raw   (sensor_msgs/Image, 16UC1 depth in mm)

The node spins in a dedicated daemon thread; get_latest_color_image() and
get_latest_depth_image() return the most recent frame as a numpy array.
"""
import sys
sys.path.append('/home/user/jaka_toolbox/install/jaka_toolbox_interfaces/local/lib/python3.10/dist-packages')

import json
import math
import threading
import time
import logging
from collections import deque
import numpy as np
import urllib.request
import rclpy
from typing import Dict, Any, Optional, List

# from agv_node import Agv_Node_Planner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  ROS2Client
# ---------------------------------------------------------------------------
class ROS2Client_Camera:
    """
    ROS2 protocol bridge layer.
    Maintains a background spin thread and stores the latest camera frames.
    """

    def __init__(self):
        self._color_frame = None   # numpy ndarray (H, W, 3) uint8  BGR
        self._depth_frame = None   # numpy ndarray (H, W) uint16  mm
        self.current_agv_map = {} # 当前地图信息
        self._color_lock = threading.Lock()
        self._depth_lock = threading.Lock()
        # 缓冲最近若干帧用于时间对齐（stamp 单位 ns）
        self._color_buffer = deque(maxlen=30)
        self._depth_buffer = deque(maxlen=30)
        self.node = None
        self._ros_ready = False
        self._spin_started = False
        self._spin_thread = None
        self._running=False
        self.agv_host="192.168.16.100"
        self.agv_port=5002
        # self.agv_node=Agv_Node_Planner()
        # self.agv_node.init_node()

        self._shutdown_lock = threading.Lock()  # 添加锁保护关闭过程

        # 服务客户端缓存
        self._clients: Dict[str, Any] = {}
        self._client_lock = threading.Lock()

    def start_camera(self):
        self._init_ros()
        self._running=True
        self.run_spin_thread()

    def stop_camera(self):
        """安全停止相机和ROS2节点"""
        with self._shutdown_lock:
            # print("[ROS2Client] Stopping camera...")
            
            # 1. 先停止运行标志
            self._running = False
            self._ros_ready = False
            
            # 2. 等待spin线程结束
            if self._spin_thread and self._spin_thread.is_alive():
                # print("[ROS2Client] Waiting for spin thread to finish...")
                self._spin_thread.join(timeout=2.0)
            
            # 3. 清理所有服务客户端
            with self._client_lock:
                for service_name in list(self._clients.keys()):
                    try:
                        if self.node:
                            self.node.destroy_client(self._clients[service_name]['client'])
                    except Exception as e:
                        pass
                        logger.debug(f"销毁client {service_name} 时出错: {e}")
                self._clients.clear()
            
            # 4. 销毁节点
            if self.node:
                try:
                    self.node.destroy_node()
                except Exception as e:
                    logger.debug(f"销毁节点时出错: {e}")
                self.node = None
            
            # 5. 最后才shutdown rclpy
            if rclpy.ok():
                try:
                    rclpy.shutdown()
                except Exception as e:
                    logger.debug(f"rclpy shutdown时出错: {e}")
            
            # print("[ROS2Client] Camera stopped successfully")

    # ------------------------------------------------------------------
    def _init_ros(self):
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from nav_msgs.msg import Odometry

            if not rclpy.ok():
                rclpy.init()

            class _AgentBridgeNode(Node):
                def __init__(inner_self):
                    super().__init__("agent_bridge_node")
                    inner_self.create_subscription(
                        Image,
                        "/camera_head/color/image_raw",
                        self._color_callback,
                        10,
                    )
                    inner_self.create_subscription(
                        Image,
                        "/camera_head/depth/image_raw",
                        self._depth_callback,
                        10,
                    )
                    inner_self.create_subscription(
                        Odometry,
                        "/JAGV_O_01/global_nav_odom",
                        self._agv_pos_callback,
                        10
                    )
                    # inner_self.get_logger().info("[ROS2Client] Camera subscriptions active.")

            self.node = _AgentBridgeNode()
            self._ros_ready = True
            # print("[ROS2Client] Node initialized. Subscribing to camera topics...")

        except Exception as e:
            print(f"[ROS2Client] ROS2 init failed (running in stub mode): {e}")
            self._ros_ready = False

    # ------------------------------------------------------------------
    def _ros_image_to_numpy(self, msg):
        dtype = np.uint8
        channels = 1
        encoding = msg.encoding.lower()
        if encoding in ("16uc1", "16sc1"):
            dtype = np.uint16
            channels = 1
        elif encoding in ("32fc1",):
            dtype = np.float32
            channels = 1
        elif encoding in ("bgr8", "rgb8"):
            channels = 3
        elif encoding in ("bgra8", "rgba8"):
            channels = 4

        # .copy() makes the array writable and owned (frombuffer is read-only)
        arr = np.frombuffer(msg.data, dtype=dtype).copy()
        if channels > 1:
            arr = arr.reshape(msg.height, msg.width, channels)
        else:
            arr = arr.reshape(msg.height, msg.width)
        return arr

    def _msg_stamp_ns(self, msg) -> int:
        try:
            return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        except Exception:
            return int(time.time() * 1e9)

    def _color_callback(self, msg):
        import numpy as np
        frame = self._ros_image_to_numpy(msg)
        # Convert RGB→BGR for OpenCV convention
        if msg.encoding.lower() == "rgb8":
            frame = frame[:, :, ::-1]
        frame = np.ascontiguousarray(frame)
        stamp_ns = self._msg_stamp_ns(msg)
        with self._color_lock:
            self._color_frame = frame
            self._color_buffer.append((stamp_ns, frame))

    def _depth_callback(self, msg):
        frame = self._ros_image_to_numpy(msg)
        stamp_ns = self._msg_stamp_ns(msg)
        with self._depth_lock:
            self._depth_frame = frame
            self._depth_buffer.append((stamp_ns, frame))

    def _agv_pos_callback(self, msg):
        # 1. 获取位置 (x, y, z)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # 2. 获取姿态四元数
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w

        # 3. 将四元数转换为偏航角 - 弧度制
        # 这是通用的 3D 四元数转 Z 轴欧拉角的公式，兼容底盘有微小倾斜的情况
        rz = math.atan2(2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz))
        
        # 4. 更新成员变量 (不加锁的话在 Python 中赋值是原子的，基本够用)
        self.current_agv_map["x"] = x
        self.current_agv_map["y"] = y
        self.current_agv_map["rz"] = rz

    def navigate_to(self,node: int, speed: int)->bool:
        """
        点到点导航
        Args:
            node:  目标节点编号
            speed: 运行速度
        Returns:
            True  - 调用成功
            False - 调用失败
        """
        url = f"http://{self.agv_host}:{self.agv_port}/MapPage/SetMapNavigation"
        
        body_dict = {"Node": str(node), "Speed": speed,"StopMode":0}
        body_bytes = json.dumps(body_dict).encode("utf-8")
        
        req = urllib.request.Request(url, data=body_bytes, method="POST")
        req.add_header("Content-Type", "application/json")
        # posnow=self.get_web_agv_pose()
        # if (posnow and posnow[3]==node):
        #     print(f"[导航] 已在目标节点 {node}，无需导航")
        #     return True
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                if resp.status != 200:
                    print(f"[导航失败] HTTP 状态码: {resp.status}")
                    return False
                ret=json.loads(resp.read().decode('utf-8'))
                
                if(ret["code"]!=0):
                    if(ret["data"]!="AGV agv_auto_move service not available"):
                        msg=ret["data"]
                        print(f"[导航失败] {msg}")
                        return False
        except Exception as e:
            print(f"[导航失败] {e}")
            return False
        cnt =0
        while True:
            time.sleep(0.1)
            cnt+=1
            if cnt>600:
                break
            ret=self.get_web_agv_pose()
            if (ret and ret[3]==node):
                break
        if cnt>600:
            print("[导航失败] 运动超时！")
            return False
        return True

    def navigate_to_nodes(self,node_list:List[int],speed:int)->bool:
        res=True
        for node in node_list:
            print(f"[导航] 导航到站点:{node}")
            ret=self.navigate_to(node,speed)
            if not ret:
                res=False
                break
        return res
    
    def navigate_to_node_auto(self,tar_node:int,speed:int)->bool:
        res=True
        posnow=self.get_web_agv_pose()
        if not posnow:
            print("[导航] AGV通讯异常!")
            return False
        
        #在当前节点，无需导航
        cur_node=posnow[3]
        print(f"[导航] 当前站点号{cur_node}")
        if cur_node==tar_node:
            return True
        #站点为0
        if cur_node==0:
            print("[导航] 当前无站点号，需要重定位!")
            return False
        #导航路径规划
        node_list=self.agv_node.get_node_planner(cur_node,tar_node)
        if not node_list:
            print("[导航] 路径规划错误!")
            return False
        print(f"[导航] 路径规划:{node_list}")
        for node in node_list[1:]:
            print(f"[导航] 导航到站点:{node}")
            ret=self.navigate_to(node,speed)
            if not ret:
                res=False
                break
        return res
    

    def check_agv_at_pos(self, node: int):
        posnow=self.get_web_agv_pose()
        if (posnow and posnow[3]==node):
            # print(f"到位检测 已在目标节点 {node}")
            return True

    def get_web_agv_pose(self) -> Optional[List[float]]:
        """
        获取 AGV 位姿，返回 [x, y, rz] 列表，失败返回 None
        """
        url = f"http://{self.agv_host}:{self.agv_port}/HomePage/GetStateInfo"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status != 200:
                    return None
                
                body = json.loads(resp.read().decode('utf-8'))
                
                # 判断接口是否报错
                if body.get("code") != 0:
                    return None
                
                # 将 data 数组扁平化为字典：{"MapName": "xxx", "PosX": "1.2", ...}
                data_dict = {
                    item["name"]: item["value"] 
                    for item in body.get("data", []) 
                    if "name" in item and "value" in item
                }
                
                x = float(data_dict.get("PosX", 0))/1000
                y = float(data_dict.get("PosY", 0))/1000
                rz = float(data_dict.get("PosAng", 0))
                node=int(data_dict.get("PosNode", 0))

                rz = math.radians(rz)
                return [x, y, rz ,node]
                
        except Exception as e:
            print(f"获取AGV位姿异常: {e}")
            return None
        
    # ------------------------------------------------------------------
    def run_spin_thread(self):
        """Start background spin thread. Safe to call multiple times (idempotent)."""
        if not self._ros_ready:
            print("[ROS2Client] Stub mode — spin thread not started.")
            return
        if self._spin_started and self._spin_thread is not None and self._spin_thread.is_alive():
            return self._spin_thread

        def _spin():
            import rclpy
            while rclpy.ok():
                try:
                    rclpy.spin_once(self.node, timeout_sec=0.1)
                except Exception as e:
                    logger.warning(f"[ROS2Client] spin_once error: {e}")
                    time.sleep(0.1)
                if not self._running:
                    print("[ROS2Client] Stopping camera...")
                    break

        t = threading.Thread(target=_spin, daemon=True, name="ros2_spin")
        t.start()
        self._spin_thread = t
        self._spin_started = True
        print("[ROS2Client] Spin thread started.")
        return t

    # ------------------------------------------------------------------
    def get_latest_color_image(self):
        """Return latest color frame as numpy (H,W,3) BGR, or None if not yet received."""
        with self._color_lock:
            return self._color_frame.copy() if self._color_frame is not None else None

    def get_latest_depth_image(self):
        """Return latest depth frame as numpy (H,W) uint16 mm, or None if not yet received."""
        with self._depth_lock:
            return self._depth_frame.copy() if self._depth_frame is not None else None

    def get_latest_rgbd_pair(self, max_delta_ms: float = 80.0):
        """
        获取时间最接近的一对 RGB/Depth 帧。

        Returns:
            dict | None: {
                "color": np.ndarray,
                "depth": np.ndarray,
                "delta_ms": float,
                "color_stamp_ns": int,
                "depth_stamp_ns": int,
            }
        """
        with self._color_lock:
            color_buf = list(self._color_buffer)
        with self._depth_lock:
            depth_buf = list(self._depth_buffer)

        if not color_buf or not depth_buf:
            return None

        color_stamp, color = color_buf[-1]
        # 以最新 color 为基准，找时间最近 depth
        depth_stamp, depth = min(depth_buf, key=lambda x: abs(x[0] - color_stamp))
        delta_ms = abs(color_stamp - depth_stamp) / 1e6

        if delta_ms > max_delta_ms:
            logger.warning(
                f"[ROS2Client] RGB-Depth 时间差过大: {delta_ms:.1f} ms (阈值 {max_delta_ms:.1f} ms)"
            )

        return {
            "color": color.copy(),
            "depth": depth.copy(),
            "delta_ms": float(delta_ms),
            "color_stamp_ns": int(color_stamp),
            "depth_stamp_ns": int(depth_stamp),
        }

    def get_latest_agv_position(self):
        """返回当前 AGV 位姿 [x, y, rz]"""
        # 检查字典是否包含必要的键，如果数据还没准备好，返回 None
        if not all(k in self.current_agv_map for k in ["x", "y", "rz"]):
            # 可以选择打印日志提示数据未就绪
            logger.warning("AGV position data not ready yet.")
            return None  # 或者返回 [0.0, 0.0, 0.0]

        # 数据就绪，返回列表
        return [
            self.current_agv_map["x"],
            self.current_agv_map["y"],
            self.current_agv_map["rz"]
        ]

    # ------------------------------------------------------------------
    def execute_action(self, action_name: str, params: dict) -> dict:
        """Dispatch a ROS2 action (stub — extend with rclpy ActionClient)."""
        print(f"[ROS2Client] execute_action: {action_name}  params={params}")
        return {"status": "success", "action": action_name}

    def call_service(self, service_name: str, params: dict) -> dict:
        """Dispatch a ROS2 service call (stub — extend with rclpy ServiceClient)."""
        print(f"[ROS2Client] call_service: {service_name}  params={params}")
        return {"status": "success"}

    def get_latest_topic_data(self, topic_name: str):
        """Generic topic accessor — returns the latest frame for known topics."""
        if topic_name == "/camera_head001/color/image_raw":
            return self.get_latest_color_image()
        if topic_name == "/camera_head001/depth/image_raw":
            return self.get_latest_depth_image()
        return None

    # ------------------------------------------------------------------
    # 服务调用相关方法
    # ------------------------------------------------------------------
    
    def _get_or_create_client(self, service_type, service_name: str):
        """
        获取或创建服务客户端
        
        Args:
            service_type: 服务类型（如 PoseQuery）
            service_name: 服务名称
        
        Returns:
            服务客户端实例
        """
        if not self._ros_ready or not self.node:
            logger.error("ROS2未就绪，无法创建服务客户端")
            return None
        
        with self._client_lock:
            if service_name not in self._clients:
                try:
                    client = self.node.create_client(service_type, service_name)
                    self._clients[service_name] = {
                        'client': client,
                        'type': service_type,
                        'name': service_name
                    }
                    logger.info(f"创建服务客户端: {service_name}")
                except Exception as e:
                    logger.error(f"创建服务客户端失败 {service_name}: {e}")
                    return None
            return self._clients[service_name]['client']
    
    def call_service_sync(self, service_type, service_name: str, request, timeout_sec: float = 5.0):
        """
        同步调用服务
        
        Args:
            service_type: 服务类型
            service_name: 服务名称
            request: 请求对象
            timeout_sec: 超时时间（秒）
        
        Returns:
            响应对象，失败返回None
        """
        client = self._get_or_create_client(service_type, service_name)
        
        if client is None:
            logger.error(f"无法获取服务客户端: {service_name}")
            return None
        
        # 等待服务可用
        if not client.wait_for_service(timeout_sec=timeout_sec):
            logger.error(f"服务 {service_name} 不可用（超时 {timeout_sec}秒）")
            return None
        
        try:
            future = client.call_async(request)
            # 手动等待完成
            start_time = time.time()
            while rclpy.ok() and not future.done():
                time.sleep(0.01)
                if time.time() - start_time > timeout_sec:
                    logger.error(f"调用服务 {service_name} 超时")
                    future.cancel()
                    return None
            
            if future.done() and future.result() is not None:
                return future.result()
            else:
                logger.error(f"服务 {service_name} 调用失败")
                return None
                
        except Exception as e:
            logger.error(f"调用服务 {service_name} 异常: {e}")
            return None
    
     # ========== 具体服务接口 ==========
    
    def query_robot_pose(self, id: int = 0, ref_joint_positions: Optional[List[float]] = None, 
                         timeout_sec: float = 5.0) -> Optional[Dict]:
        """
        查询机器人位姿
        
        Args:
            id: 坐标系ID (0=当前位姿, 1/2=用户坐标系等)
            ref_joint_positions: 参考关节位置（7个浮点数），None或空列表表示使用当前关节位置
            timeout_sec: 超时时间
        
        Returns:
            包含位姿信息的字典，失败返回None
        """
        try:
            # 导入服务类型
            from jaka_toolbox_interfaces.srv import PoseQuery
            
            # 创建请求
            request = PoseQuery.Request()
            request.id = id
            request.ref_joint_positions = ref_joint_positions or []
            
            # 调用服务
            response = self.call_service_sync(PoseQuery, '/query_pose_ext', request, timeout_sec)
            print(f"self.query_robot_pose:{response}\n")
            if response and response.success:
                return {
                    'success': True,
                    'message': response.message,
                    'cartesian_pose': list(response.cartesian_pose),
                    'joint_positions': list(response.joint_positions),
                    'use_mm': response.use_mm,
                    'use_rad': response.use_rad
                }
            elif response:
                logger.warning(f"查询位姿失败: {response.message}")
                return {'success': False, 'message': response.message}
            else:
                logger.error("查询位姿失败：无响应")
                return None
            
        except ImportError as e:
            logger.error(f"无法导入服务类型: {e}")
            return None
        except Exception as e:
            logger.error(f"查询位姿异常: {e}")
            return None

    def get_normalize_robot_pose(self):
        query_robot_pose=self.query_robot_pose(4)["cartesian_pose"]
        xyz_raw = query_robot_pose[:3]
        rpy_raw = query_robot_pose[3:]
        unit_label = "mm→m" if max(abs(v) for v in xyz_raw) > 10.0 else "m"
        if unit_label == "mm→m":
            pose_m = [v / 1000.0 for v in xyz_raw] + rpy_raw
        else:
            pose_m = query_robot_pose
        return pose_m

    def pallet_pick_box(self,p1,p2,type=1,timeout_sec: float = 120.0)->bool:
        """
        查询机器人位姿
        
        Args:
            id: 坐标系ID (0=当前位姿, 1/2=用户坐标系等)
            ref_joint_positions: 参考关节位置（7个浮点数），None或空列表表示使用当前关节位置
            timeout_sec: 超时时间
        
        Returns:
            包含位姿信息的字典,失败返回None
        """
        try:
            # 导入服务类型
            from jaka_toolbox_interfaces.srv import PalletPickBox
                        # 创建请求
            request = PalletPickBox.Request()
            request.p1 = p1
            request.p2 = p2
            request.type = type
            
            # 调用服务
            response = self.call_service_sync(PalletPickBox, '/pallet_pick_box', request, timeout_sec)
            print(f"self.pallet_pick_box:{response}\n")
            if response:
                if response.ret_code!=0:
                    logger.error(" pallet_pick_box失败")
                    return False
                return True
            else:
                logger.error(" pallet_pick_box失败：无响应")
                return False
            
        except ImportError as e:
            logger.error(f"无法导入服务类型: {e}")
            return False
        except Exception as e:
            logger.error(f" pallet_pick_box异常: {e}")
            return False


    def pallet_place_box(self,p1,p2,type=1,timeout_sec: float = 120.0)->bool:
        """
        料架放箱子
        
        Args:
            p1:坐标点1
            P2:坐标点2
            type: 类型(暂未使用)
            timeout_sec: 超时时间
        
        Returns:
            Kargo执行动作结果
        """
        try:
            # 导入服务类型
            from jaka_toolbox_interfaces.srv import PalletPlaceBox
                        # 创建请求
            request = PalletPlaceBox.Request()
            request.p1 = p1
            request.p2=p2
            request.type=type
            
            # 调用服务
            response = self.call_service_sync(PalletPlaceBox, '/pallet_place_box', request, timeout_sec)
            if response:
                if (response.ret_code!=0):
                    print(f"料架放箱失败:{response}")
                    return False
            else:
                print("料架放箱失败:无响应")
                return False
            
        except ImportError as e:
            print(f"料架放箱失败: {e}")
            return False
        except Exception as e:
            print(f"料架放箱失败: {e}")
            return False
        return True

    def pallet_pick_from_shelf(self,p1,p2,type=6,timeout_sec: float = 120.0)->bool:
        """
        料架取箱
        
        Args:
            timeout_sec: 超时时间
        
        Returns:
            动作是否成功
        """
        try:
            # 导入服务类型
            from jaka_toolbox_interfaces.srv import PalletPickFromShelf
            # 创建请求
            request = PalletPickFromShelf.Request()
            request.p1 = p1
            request.p2=p2
            request.type=type
            
            # 调用服务
            response = self.call_service_sync(PalletPickFromShelf, '/pallet_pick_from_shelf', request, timeout_sec)
            print("料架取箱:",response)
            
        except ImportError as e:
            print("料架取箱失败:无法导入服务类型")
            return False
        except Exception as e:
            print(f"料架取箱失败:{e}")
            return False
        return True


    def pallet_box_to_panel(self,p1,p2,type=1,timeout_sec: float = 120.0)->bool:

        try:
            # 导入服务类型
            from jaka_toolbox_interfaces.srv import PalletPlaceToPanel
                        # 创建请求
            request = PalletPlaceToPanel.Request()
            request.p1 = p1
            request.p2=p2
            request.type=type
            
            # 调用服务
            response = self.call_service_sync(PalletPlaceToPanel, '/pallet_place_to_panel', request, timeout_sec)
            if response:
                if (response.ret_code!=0):
                    print(f"托盘放箱失败:{response}")
                    return False
            else:
                print("托盘放箱失败:无响应")
                return False
            
        except ImportError as e:
            print("托盘放箱失败:无法导入服务类型")
            return False
        except Exception as e:
            print(f"托盘放箱失败:{e}")
            return False
        return True
    def pallet_place_to_empty_panel(self,p1,p2,type=1,timeout_sec: float = 120.0)->bool:

        try:
            # 导入服务类型
            from jaka_toolbox_interfaces.srv import PalletPlaceToEmptyPanel
                        # 创建请求
            request = PalletPlaceToEmptyPanel.Request()
            request.p1 = p1
            request.p2=p2
            request.arm_type=type
            
            # 调用服务
            response = self.call_service_sync(PalletPlaceToEmptyPanel, '/pallet_place_to_empty_panel', request, timeout_sec)
            if response:
                if (response.ret_code!=0):
                    print(f"托盘放箱失败:{response}")
                    return False
            else:
                print("托盘放箱失败:无响应")
                return False
            
        except ImportError as e:
            print("托盘放箱失败:无法导入服务类型")
            return False
        except Exception as e:
            print(f"托盘放箱失败:{e}")
            return False
        return True

    def pallet_store_box(self,cartesian_pose,timeout_sec: float = 120.0)->bool:
        """
        托盘放箱
        
        Args:
            timeout_sec: 超时时间
        
        Returns:
            动作是否成功
        """
        try:
            # 导入服务类型
            from jaka_toolbox_interfaces.srv import LinearMove
                        # 创建请求
            request = LinearMove.Request()
            request.motion_unit_type=0
            request.motion_unit_id=-1
            request.move_mode=1#0绝对，1相对
            request.cartesian_pose=cartesian_pose
            request.velocity=2000.0
            request.acceleration=4000.0
            
            # 调用服务
            response = self.call_service_sync(LinearMove, '/linear_move_arm', request, timeout_sec)
            print("相对直线运动：",response)
            # if response:
            #     if (response.ret_code!=0):
            #         print(f"托盘放箱失败:{response}")
            #         return False
            # else:
            #     print("托盘放箱失败:无响应")
            #     return False
            
        except ImportError as e:
            print("托盘放箱失败:无法导入服务类型")
            return False
        except Exception as e:
            print(f"托盘放箱失败:{e}")
            return False
        return True

    def sample_sensor_zero(self,mode,timeout_sec: float = 120.0)->bool:
        """
        力零位校准
        
        Args:
            timeout_sec: 超时时间
        
        Returns:
            动作是否成功
        """
        try:
            # 导入服务类型
            from jaka_toolbox_interfaces.srv import ForceControl
                        # 创建请求
            request = ForceControl.Request()
            request.mode=mode
            
            # 调用服务
            response = self.call_service_sync(ForceControl, '/force_control', request, timeout_sec)
            print("力零位校准",response)
            # if response:
            #     if (response.ret_code!=0):
            #         print(f"托盘放箱失败:{response}")
            #         return False
            # else:
            #     print("托盘放箱失败:无响应")
            #     return False
            
        except ImportError as e:
            print("托盘放箱失败:无法导入服务类型")
            return False
        except Exception as e:
            print(f"托盘放箱失败:{e}")
            return False
        return True

    def Joint_Move(self,motion_unit_type,motion_unit_id,move_mode,joint_positions,timeout_sec: float = 120.0)->bool:
        """
        托盘放箱
        
        Args:
            timeout_sec: 超时时间
        
        Returns:
            动作是否成功
        """
        try:
            # 导入服务类型
            from jaka_toolbox_interfaces.srv import JointMove
                        # 创建请求
            request = JointMove.Request()
            request.motion_unit_type=motion_unit_type
            request.motion_unit_id=motion_unit_id
            request.move_mode=move_mode#0绝对，1相对
            request.joint_positions=joint_positions
            request.joint_velocity=2000.0
            request.joint_acceleration=4000.0
            request.joint_jerk=4000.0
            
            # 调用服务
            response = self.call_service_sync(JointMove, '/joint_move_arm', request, timeout_sec)
            print("关节运动：",response)
            # if response:
            #     if (response.ret_code!=0):
            #         print(f"托盘放箱失败:{response}")
            #         return False
            # else:
            #     print("托盘放箱失败:无响应")
            #     return False
            
        except ImportError as e:
            print("托盘放箱失败:无法导入服务类型")
            return False
        except Exception as e:
            print(f"托盘放箱失败:{e}")
            return False
        return True


    def pallet_place_prepare(self,lift_height,waist_angle,head0_angle,head1_angle,timeout_sec: float = 120.0)->bool:
        """
        托盘放箱
        
        Args:
            timeout_sec: 超时时间
        
        Returns:
            动作是否成功
        """
        try:
            # 导入服务类型
            from jaka_toolbox_interfaces.srv import PlaceBoxPrepare
                        # 创建请求
            request = PlaceBoxPrepare.Request()
            request.lift_height=lift_height
            request.waist_angle=waist_angle
            request.head0_angle=head0_angle
            request.head1_angle=head1_angle
            # 调用服务
            response = self.call_service_sync(PlaceBoxPrepare, '/place_box_prepare', request, timeout_sec)
            print("准备动作",response)
            # if response:
            #     if (response.ret_code!=0):
            #         print(f"托盘放箱失败:{response}")
            #         return False
            # else:
            #     print("托盘放箱失败:无响应")
            #     return False
            
        except ImportError as e:
            print("托盘放箱失败:无法导入服务类型")
            return False
        except Exception as e:
            print(f"托盘放箱失败:{e}")
            return False
        return True
    
    def dual_arm_move_with_ext(self, 
                             left_pos=None, 
                             right_pos=None, 
                             ext_pos=None, 
                             left_move_type: int = 0, 
                             right_move_type: int = 0, 
                             left_move_mode: int = 0, 
                             right_move_mode: int = 0, 
                             command: str = "", 
                             timeout_sec: float = 120.0) -> bool:
        """
        托盘放箱准备（双臂+外部轴统一运动）
        
        Args:
            left_pos: 左臂目标 [j1~j7] 或 [x,y,z,rx,ry,rz], 默认全0
            right_pos: 右臂目标 [j1~j7] 或 [x,y,z,rx,ry,rz], 默认全0
            ext_pos: 外部轴目标 [lift, waist, head0, head1], 默认全0
            left_move_type: 左臂运动类型 0=JOINT_MOVE, 1=LINEAR_MOVE
            right_move_type: 右臂运动类型 0=JOINT_MOVE, 1=LINEAR_MOVE
            left_move_mode: 左臂运动模式 0=ABS, 1=INCR
            right_move_mode: 右臂运动模式 0=ABS, 1=INCR
            command: 附加指令字符串
            timeout_sec: 超时时间
        
        Returns:
            动作是否成功
        """
        try:
            # 导入服务类型
            from jaka_toolbox_interfaces.srv import DualArmMoveWithExt
            
            # === 辅助函数：处理列表参数，确保长度和类型正确 ===
            def format_pos(pos_list, expected_len, default_val=0.0):
                if pos_list is None:
                    return [default_val] * expected_len
                if len(pos_list) != expected_len:
                    raise ValueError(f"位置参数长度错误，期望 {expected_len}，实际传入 {len(pos_list)}")
                # 确保列表内元素都是 float 类型
                return [float(p) for p in pos_list]

            # 创建请求
            request = DualArmMoveWithExt.Request()
            
            # 处理并赋值位置参数 (ROS2 会自动将 Python list 转换为 float64[])
            request.left_pos = format_pos(left_pos, 7)
            request.right_pos = format_pos(right_pos, 7)
            request.ext_pos = format_pos(ext_pos, 4)
            
            # 赋值运动参数
            request.left_move_type = left_move_type
            request.right_move_type = right_move_type
            request.left_move_mode = left_move_mode
            request.right_move_mode = right_move_mode
            request.command = command

            # 调用服务 (请确认服务名是否正确)
            response = self.call_service_sync(DualArmMoveWithExt, '/dual_arm_move_with_ext', request, timeout_sec)
            print("准备动作响应:", response)
            
            # 建议取消以下注释以判断真实执行结果
            # if response:
            #     if response.ret_code != 0:
            #         print(f"托盘放箱准备失败: {response.ret_msg}")
            #         return False
            # else:
            #     print("托盘放箱准备失败: 无响应")
            #     return False
            
        except ValueError as ve:
            print(f"多轴同时运动失败: 参数错误 - {ve}")
            return False
        except ImportError as e:
            print(f"多轴同时运动失败: 无法导入服务类型 - {e}")
            return False
        except Exception as e:
            print(f"多轴同时运动失败: {e}")
            return False
            
        return True

    # ------------------------------------------------------------------
    # BoxPlacement 力控码垛 Action 接口
    # ------------------------------------------------------------------

    def set_tool_id(self, left_id: int = 11, right_id: int = 12,
                    timeout_sec: float = 5.0) -> bool:
        """
        设置工具坐标系 ID（BoxPlacement 前置步骤）
        """
        try:
            from jaka_toolbox_interfaces.srv import ToolIdSetting
            request = ToolIdSetting.Request()
            request.left_id = left_id
            request.right_id = right_id
            response = self.call_service_sync(
                ToolIdSetting, '/set_tool_id', request, timeout_sec)
            if response and response.success == 1:
                print(f"工具ID设置成功: {response.message}")
                return True
            else:
                print(f"工具ID设置失败: {response.message if response else '无响应'}")
                return False
        except Exception as e:
            print(f"工具ID设置异常: {e}")
            return False

    def box_placement(self,
                      left_claw_id: int = 63991,
                      right_claw_id: int = 59687,
                      search_axis: str = "x",
                      search_direction: int = 1,
                      push_axis: str = "y",
                      push_direction: int = 1,
                      downward_axis: str = "z",
                      downward_direction: int = -1,
                      start_corner: str = "lower_right",
                      release_after_push: bool = True,
                      timeout_sec: float = 120.0) -> bool:
        """
        调用 BoxPlacement Action 进行力控寻孔码垛。

        前置条件：双臂已夹取箱子并移动到准备位置（由 pallet_box_to_panel 完成）。

        Args:
            left_claw_id: 左夹爪 ID
            right_claw_id: 右夹爪 ID
            search_axis: 搜孔轴 ("x"/"y"/"z", TCP 坐标系)
            search_direction: 搜孔方向 (1 或 -1)
            push_axis: 推箱轴 ("x"/"y"/"z", TCP 坐标系)
            push_direction: 推箱方向 (1 或 -1)
            downward_axis: 重力轴 ("x"/"y"/"z", 基座坐标系)
            downward_direction: 重力方向 (1 或 -1)
            start_corner: 起始角 ("lower_left" 或 "lower_right")
            release_after_push: 推箱后是否释放夹爪
            timeout_sec: 超时时间

        Returns:
            True=成功, False=失败
        """
        from rclpy.action import ActionClient
        from geometry_msgs.msg import Pose, Point, Quaternion
        from scipy.spatial.transform import Rotation as R
        from jaka_toolbox_interfaces.action import BoxPlacement

        def xyzrpy_to_pose(xyzrpy, use_degree):
            x, y, z, roll, pitch, yaw = xyzrpy
            rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=use_degree)
            quat = rot.as_quat()
            pose_msg = Pose()
            pose_msg.position = Point(x=float(x), y=float(y), z=float(z))
            pose_msg.orientation = Quaternion(
                x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            return pose_msg

        if not self._ros_ready or not self.node:
            print("BoxPlacement失败: ROS2未就绪")
            return False

        action_client = None
        try:
            action_client = ActionClient(
                self.node, BoxPlacement, '/box_placement')

            # 等待 Action Server 上线
            wait_elapsed = 0.0
            while not action_client.wait_for_server(timeout_sec=1.0):
                wait_elapsed += 1.0
                if wait_elapsed >= timeout_sec:
                    print("BoxPlacement失败: Action Server 不可用")
                    return False

            # 构建 Goal
            goal_msg = BoxPlacement.Goal()
            goal_msg.lefttcp_to_flange = xyzrpy_to_pose(
                [0.06, 0.0, 0.04, 90.0, -90.0, 0.0], True)
            goal_msg.righttcp_to_flange = xyzrpy_to_pose(
                [0.06, 0.0, 0.04, 90.0, -90.0, 0.0], True)
            goal_msg.left_claw_id = left_claw_id
            goal_msg.right_claw_id = right_claw_id
            goal_msg.search_axis = search_axis
            goal_msg.search_direction = search_direction
            goal_msg.push_axis = push_axis
            goal_msg.push_direction = push_direction
            goal_msg.downward_axis = downward_axis
            goal_msg.downward_direction = downward_direction
            goal_msg.start_corner = start_corner
            goal_msg.release_after_push = release_after_push

            print("[BoxPlacement] 发送 Goal...")

            # 1. 发送 Goal，等待接受
            send_goal_future = action_client.send_goal_async(goal_msg)
            start_time = time.time()
            while rclpy.ok() and not send_goal_future.done():
                time.sleep(0.01)
                if time.time() - start_time > timeout_sec:
                    print("BoxPlacement失败: 等待Goal接受超时")
                    return False

            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                print("BoxPlacement失败: Goal被拒绝")
                return False

            print("[BoxPlacement] Goal已接受，等待结果...")

            # 2. 等待最终结果
            get_result_future = goal_handle.get_result_async()
            start_time = time.time()
            while rclpy.ok() and not get_result_future.done():
                time.sleep(0.01)
                if time.time() - start_time > timeout_sec:
                    print("BoxPlacement失败: 等待结果超时")
                    return False

            result = get_result_future.result().result
            if result.success == 1:
                print(f"[BoxPlacement] 成功: {result.message}")
                return True
            else:
                print(f"[BoxPlacement] 失败: {result.message}")
                return False

        except Exception as e:
            print(f"BoxPlacement异常: {e}")
            return False
        finally:
            if action_client is not None:
                action_client.destroy()


if __name__ == '__main__':
    import time
    import cv2
    import os
    import sys
    import select  # Python 内置模块，用于监听终端输入，无需 root 权限

    # 目标保存路径
    save_dir = 'box_pallet/my_dataset/tempimages'
    
    # 确保保存路径存在
    os.makedirs(save_dir, exist_ok=True)
    
    demo = ROS2Client_Camera()
    demo.start_camera()
    
    print("相机已启动，正在预热...")
    time.sleep(3)
    print("预热完成！请在终端按下【任意键】进行拍照，按【q】键退出程序。")
    
    # 将终端设置为无缓冲模式（无需按回车即可立即响应按键）
    # 注意：这会改变终端的行为，因此需要在退出时恢复原样
    import tty
    import termios
    
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        while True:
            # 监听终端输入，超时设置为0.1秒，避免CPU空转占用过高
            # select.select(读取列表, 写入列表, 异常列表, 超时时间)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                # 读取按下的键
                key = sys.stdin.read(1)
                
                # 按下 q 键退出循环
                if key.lower() == 'q':
                    print("检测到 'q' 按键，退出程序。")
                    break
                
                # 获取最新图像
                img_color = demo.get_latest_color_image()
                
                if img_color is not None:
                    # 使用时间戳生成不重复的文件名（精确到毫秒）
                    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                    ms = int((time.time() % 1) * 1000)  # 获取毫秒部分
                    filename = f"capture_{timestamp}_{ms}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    
                    # 保存图片
                    cv2.imwrite(filepath, img_color)
                    print(f"拍照成功！图片已保存为: {filepath}")
                else:
                    print("获取图像失败，请检查相机状态。")
                    
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，退出程序。")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 恢复终端原始设置（非常重要，否则退出后终端将无法正常显示输入的字符）
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        # 安全关闭相机
        print("正在关闭相机...")
        demo.stop_camera()
        print("相机已关闭。")