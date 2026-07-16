"""
工件装配主流程脚本
按照 readme.md 中的 12 步流程搭建，不修改原有代码。

流程概述：
  1.  双臂运动到安全位置
  2.  头部相机拍照定位工件位置和摆放角度
  3.  左臂运动到抓取前的过渡位置
  4.  左臂打开夹爪
  5.  左臂运动到工件位置
  6.  左臂关闭夹爪，等待1秒
  7.  左臂上抬200mm，然后返回抓取前过渡位置
  8.  左臂移动到安装底座上方
  9.  左臂向下运动尝试安装（力控寻孔循环）
  10. 左臂撤离工件，回到安全位置
  11. 右臂取回工件，放置到另一侧
  12. 右臂回到安全位置
"""

import time
import math
import logging

import jkrc

from conf import (
    ROBOT_HOST, ROBOT_USER, ROBOT_PASSWORD,
    INIT_GRIPPER, OPEN_GRIPPER, CLOSE_GRIPPER,
    LeftArmPose, RightArmPose,
)
from box_camera_copy import ROS2Client_Camera
from flange_detector import FlangeDetector

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("assembly")

# ---------------------------------------------------------------------------
# 常量 / 可调参数
# ---------------------------------------------------------------------------
# 运动速度（mm/s 或 °/s，取决于接口单位）
MOVE_SPEED = 500.0
MOVE_ACCEL = 1000.0

# 力控寻孔参数
INSERT_LIFT_MM = 20.0           # 安装受阻后上抬距离 mm
INSERT_ROTATE_DEG = 5.0         # 安装受阻后 RZ 顺时针旋转角度
INSERT_MAX_RETRIES = 36         # 最大重试次数（360/5 = 72，取一半足够）
INSERT_DOWN_SPEED = 50.0        # 安装下探速度 mm/s
INSERT_DOWN_DISTANCE = 200.0    # 单次下探最大距离 mm

# JAKA 内置力控参数（方案 2）
# robot_id：0=左臂，1=右臂
FC_ROBOT_ID = 0                 # 左臂
FC_SENSOR_ID = 0                # 力传感器 ID（视现场配置而定）
FC_AXIS = 2                     # 力控方向：0=X, 1=Y, 2=Z（TCP 坐标系）
FC_TARGET_FORCE = -15.0         # 目标接触力 N（负值=沿轴负方向压入）
FC_REBOUND = 0.5                # 回弹系数（0~1，越大回弹越快）
FC_CONSTANT = 0.0               # 力常数（通常 0，由控制器计算）
FC_NORMAL_TRACK = 0             # 法向跟踪：0=关闭，1=开启
FC_VEL_LIMIT = (50.0, 50.0, 50.0, 30.0, 30.0, 30.0)   # 力控速度上限 (mm/s, deg/s)
FC_TOL = (5.0, 5.0, 5.0, 2.0, 2.0, 2.0)               # 力/力矩容差
# 安装到位判定
FC_INSERT_DEPTH_MM = 30.0       # 下探深度达到此值且力未超限 → 视为安装成功

PICK_LIFT_MM = 200.0            # 抓取后上抬距离 mm

# 夹爪通道
GRIPPER_CHANNEL = 1


# ===========================================================================
#  工具函数
# ===========================================================================

def init_robot(host: str, user: str, password: str) -> jkrc.RC:
    """连接、登录、上电、使能机器人，并初始化夹爪。"""
    log.info("连接机器人 %s ...", host)
    robot = jkrc.RC(host)
    robot.login(True, user, password)
    robot.power_on()
    robot.enable_robot()
    log.info("机器人已连接并使能")

    # 初始化夹爪
    log.info("初始化夹爪 ...")
    robot.send_tio_rs_command(GRIPPER_CHANNEL, INIT_GRIPPER, 1)
    robot.send_tio_rs_command(GRIPPER_CHANNEL, INIT_GRIPPER, 0)
    time.sleep(0.5)
    log.info("夹爪初始化完成")
    return robot


def open_left_gripper(robot: jkrc.RC):
    """打开夹爪。"""
    log.info("打开夹爪")
    # robot.send_tio_rs_command(GRIPPER_CHANNEL, OPEN_GRIPPER, 1)
    robot.send_tio_rs_command(GRIPPER_CHANNEL, OPEN_GRIPPER, 0)
    time.sleep(0.5)


def close_left_gripper(robot: jkrc.RC):
    """关闭夹爪。"""
    log.info("关闭夹爪")
    # robot.send_tio_rs_command(GRIPPER_CHANNEL, CLOSE_GRIPPER, 1)
    robot.send_tio_rs_command(GRIPPER_CHANNEL, CLOSE_GRIPPER, 0)
    time.sleep(1.0)  # readme 要求等待 1 秒


def open_right_gripper(robot: jkrc.RC):
    """打开右臂夹爪。"""
    log.info("打开右臂夹爪")
    robot.send_tio_rs_command(RIGHT_GRIPPER_CHANNEL, OPEN_GRIPPER, 1)
    # robot.send_tio_rs_command(RIGHT_GRIPPER_CHANNEL, OPEN_GRIPPER, 0)
    time.sleep(0.5)


def close_right_gripper(robot: jkrc.RC):
    """关闭右臂夹爪。"""
    log.info("关闭右臂夹爪")
    robot.send_tio_rs_command(RIGHT_GRIPPER_CHANNEL, CLOSE_GRIPPER, 1)
    # robot.send_tio_rs_command(RIGHT_GRIPPER_CHANNEL, CLOSE_GRIPPER, 0)
    time.sleep(1.0)


def move_left_arm_joint(robot: jkrc.RC, joint_pos, speed: float = MOVE_SPEED, is_block: bool = True):
    """左臂关节运动到目标关节角（使用 robot_run_multi_movj）。"""
    log.info("左臂关节运动 -> %s", joint_pos)
    robot.robot_run_multi_movj(
        robot_id=0,                             # 0=左臂
        move_mode=[jkrc.MoveMode.ABS]*2,
        joint_pos=[tuple(joint_pos),(0)*6],
        is_block=is_block,
        vel=[speed],
        acc=[MOVE_ACCEL],
        tol=[0.0],
    )


def move_left_arm_linear(robot: jkrc.RC, pose, speed: float = MOVE_SPEED, is_block: bool = True):
    """左臂直线运动到目标笛卡尔位姿（使用 robot_run_multi_movl）。"""
    log.info("左臂直线运动 -> %s", pose)
    robot.robot_run_multi_movl(
        robot_id=0,                             # 0=左臂
        move_mode=[jkrc.MoveMode.ABS]*2,
        end_pos=[tuple(pose) , (0,)*6],
        is_block=is_block,
        vel=[speed],
        acc=[MOVE_ACCEL],
        tol=[0.0],
    )


def move_left_arm_linear_relative(robot: jkrc.RC, delta_pose, speed: float = MOVE_SPEED, is_block: bool = True):
    """左臂相对直线运动（增量模式，使用 robot_run_multi_movl）。"""
    log.info("左臂相对直线运动 delta=%s", delta_pose)
    robot.robot_run_multi_movl(
        robot_id=0,                             # 0=左臂
        move_mode=[jkrc.MoveMode.INCR]*2,
        end_pos=[tuple(delta_pose) , (0,)*6],
        is_block=is_block,
        vel=[speed],
        acc=[MOVE_ACCEL],
        tol=[0.0],
    )


def move_right_arm_joint(robot: jkrc.RC, joint_pos, speed: float = MOVE_SPEED, is_block: bool = True):
    """右臂关节运动到目标关节角（使用 robot_run_multi_movj）。"""
    log.info("右臂关节运动 -> %s", joint_pos)
    robot.robot_run_multi_movj(
        robot_id=1,                             # 1=右臂
        move_mode=[jkrc.MoveMode.ABS]*2,
        joint_pos=[(0)*6,tuple(joint_pos)],
        is_block=is_block,
        vel=[speed],
        acc=[MOVE_ACCEL],
        tol=[0.0],
    )


def move_right_arm_linear(robot: jkrc.RC, pose, speed: float = MOVE_SPEED, is_block: bool = True):
    """右臂直线运动到目标笛卡尔位姿（使用 robot_run_multi_movl）。"""
    log.info("右臂直线运动 -> %s", pose)
    robot.robot_run_multi_movl(
        robot_id=1,                             # 1=右臂
        move_mode=[jkrc.MoveMode.ABS]*2,
        end_pos=[(0)*6, tuple(pose)],
        is_block=is_block,
        vel=[speed],
        acc=[MOVE_ACCEL],
        tol=[0.0],
    )


def move_right_arm_linear_relative(robot: jkrc.RC, delta_pose, speed: float = MOVE_SPEED, is_block: bool = True):
    """右臂相对直线运动（增量模式，使用 robot_run_multi_movl）。"""
    log.info("右臂相对直线运动 delta=%s", delta_pose)
    robot.robot_run_multi_movl(
        robot_id=1,                             # 1=右臂
        move_mode=[jkrc.MoveMode.INCR]*2,
        end_pos=[(0)*6,tuple(delta_pose)],
        is_block=is_block,
        vel=[speed],
        acc=[MOVE_ACCEL],
        tol=[0.0],
    )


def get_right_tcp(robot: jkrc.RC):
    """获取右臂当前 TCP 位姿。"""
    ret = robot.get_tcp_position()
    # ret 通常为 (status, left_pose, right_pose)
    if len(ret) == 3:
        return ret[2]  # 右臂 TCP
    return ret[1]


def get_left_tcp(robot: jkrc.RC):
    """获取左臂当前 TCP 位姿。"""
    ret = robot.get_tcp_position()
    # ret 通常为 (status, pose) 或 (status, left_pose, right_pose)
    if len(ret) == 3:
        return ret[1]  # 左臂 TCP
    return ret[1]


def get_ft_sensor_data(robot: jkrc.RC, sensor_id: int = 0, data_type: int = 0):
    """
    读取力/力矩传感器数据。
    返回 (fx, fy, fz, tx, ty, tz) 或 None。
    """
    try:
        ret = robot.get_ft_sensor_data(sensor_id, data_type)
        if ret and ret[0] == 0:
            return ret[1]
    except Exception as e:
        log.warning("读取力传感器数据异常: %s", e)
    return None


def check_force_exceeded(robot: jkrc.RC, threshold: float = 10.0) -> bool:
    """检测力传感器 Z 轴力是否超过阈值（工具函数，仅供调试使用）。"""
    data = get_ft_sensor_data(robot)
    if data is None:
        return False
    fz = abs(data[2]) if len(data) > 2 else 0.0
    if fz > threshold:
        log.warning("力传感器 Fz=%.2f 超过阈值 %.2f", fz, threshold)
        return True
    return False


# ===========================================================================
#  视觉定位
# ===========================================================================

def vision_locate_workpiece(camera: ROS2Client_Camera,
                            detector: FlangeDetector):
    """
    步骤 2：头部相机拍照，定位工件位置和摆放角度。

    Returns:
        dict | None: 包含 pixel_center, angle_deg, depth 等信息，失败返回 None
    """
    log.info("===== 步骤 2：视觉定位工件 =====")

    # 等待相机出图
    time.sleep(1.0)

    result = detector.detect_with_pose_from_camera(camera)
    if result.is_empty():
        log.error("未检测到工件！")
        return None

    best = result.best_bbox(strategy="largest")
    log.info("检测到工件: center=%s, angle=%.1f°, conf=%.3f",
             best.get("pose_center") or best["center"],
             best.get("pose_angle_deg") or 0.0,
             best["conf"])

    return best


# ===========================================================================
#  主流程
# ===========================================================================

def run_assembly():
    """执行完整的工件装配流程。"""

    # ------ 初始化 ------
    robot = init_robot(ROBOT_HOST, ROBOT_USER, ROBOT_PASSWORD)

    camera = ROS2Client_Camera()
    camera.start_camera()
    log.info("相机已启动")

    detector = FlangeDetector()  # 使用默认模型路径

    try:
        # ==============================================================
        # 步骤 1：双臂运动到安全位置
        # ==============================================================
        log.info("===== 步骤 1：双臂运动到安全位置 =====")
        if LeftArmPose.Home:
            move_left_arm_joint(robot, LeftArmPose.Home)
        else:
            log.warning("左臂安全位置未配置，跳过")

        # 右臂运动到安全位置
        if RightArmPose.Home:
            move_right_arm_joint(robot, RightArmPose.Home)
        else:
            log.warning("右臂安全位置未配置，跳过")

        # ==============================================================
        # 步骤 2：头部相机拍照定位工件位置和摆放角度
        # ==============================================================
        workpiece_info = vision_locate_workpiece(camera, detector)
        if workpiece_info is None:
            log.error("视觉定位失败，终止流程")
            return

        # TODO: 将像素坐标转换为 base 系坐标（需要深度图 + 相机标定）
        # pick_pose_in_base = pixel_to_base(workpiece_info, camera)
        pick_pose_in_base = None  # 待实现
        log.info("工件 base 系坐标: %s", pick_pose_in_base)

        # ==============================================================
        # 步骤 3：左臂运动到抓取前的过渡位置
        # ==============================================================
        log.info("===== 步骤 3：左臂运动到抓取前过渡位置 =====")
        if LeftArmPose.PickReadyPose:
            move_left_arm_joint(robot, LeftArmPose.PickReadyPose)
        else:
            log.warning("左臂抓取过渡位置未配置，跳过")

        # ==============================================================
        # 步骤 4：左臂打开夹爪
        # ==============================================================
        log.info("===== 步骤 4：打开左臂夹爪 =====")
        open_left_gripper(robot)

        # ==============================================================
        # 步骤 5：左臂运动到工件位置
        # ==============================================================
        log.info("===== 步骤 5：左臂运动到工件位置 =====")
        if pick_pose_in_base is not None:
            move_left_arm_linear(robot, pick_pose_in_base)
        else:
            log.warning("工件坐标未获取，跳过运动")

        # ==============================================================
        # 步骤 6：左臂关闭夹爪，等待 1 秒
        # ==============================================================
        log.info("===== 步骤 6：关闭夹爪 =====")
        close_left_gripper(robot)  # 内部已 sleep 1 秒

        # ==============================================================
        # 步骤 7：左臂上抬 200mm，然后返回抓取前过渡位置
        # ==============================================================
        log.info("===== 步骤 7：上抬 200mm 并返回过渡位置 =====")
        # 相对上抬 200mm（Z 轴正方向）
        move_left_arm_linear_relative(robot, [0, 0, PICK_LIFT_MM, 0, 0, 0])

        if LeftArmPose.PickReadyPose:
            move_left_arm_joint(robot, LeftArmPose.PickReadyPose)

        # ==============================================================
        # 步骤 8：左臂移动到安装底座上方
        # ==============================================================
        log.info("===== 步骤 8：移动到安装底座上方 =====")
        if LeftArmPose.PutReadyPose:
            move_left_arm_joint(robot, LeftArmPose.PutReadyPose)
        else:
            log.warning("安装底座上方位置未配置，跳过")

        # ==============================================================
        # 步骤 9：力控安装（下探 + 检测力 + 旋转重试）
        # ==============================================================
        log.info("===== 步骤 9：力控安装 =====")
        force_insert(robot, max_retries=INSERT_MAX_RETRIES)

        # ==============================================================
        # 步骤 10：左臂撤离工件，回到安全位置
        # ==============================================================
        log.info("===== 步骤 10：左臂撤离并回安全位置 =====")
        # 上抬撤离
        move_left_arm_linear_relative(robot, [0, 0, PICK_LIFT_MM, 0, 0, 0])
        # 打开左臂夹爪释放
        open_left_gripper(robot)
        if LeftArmPose.Home:
            move_left_arm_joint(robot, LeftArmPose.Home)

        # ==============================================================
        # 步骤 11：右臂来到工件位置，取回工件，放置到另一侧
        # ==============================================================
        log.info("===== 步骤 11：右臂取回工件并放置 =====")
        right_arm_pick_and_place(robot)

        # ==============================================================
        # 步骤 12：右臂回到安全位置
        # ==============================================================
        log.info("===== 步骤 12：右臂回到安全位置 =====")
        if RightArmPose.Home:
            move_right_arm_joint(robot, RightArmPose.Home)
        else:
            log.warning("右臂安全位置未配置，跳过")

        log.info("========== 装配流程完成 ==========")

    except Exception as e:
        log.exception("装配流程异常: %s", e)
    finally:
        # 清理
        camera.stop_camera()
        robot.logout()
        log.info("相机已关闭，机器人已登出")


# ===========================================================================
#  步骤 9 详细实现：力控安装（方案 2 — JAKA 内置力控）
# ===========================================================================
#
# 原理：
#   利用 JAKA SDK 的 set_ft_ctrl_config() 开启控制器级别的力控闭环。
#   控制器以 1ms 周期实时调节 Z 轴运动，当接触力达到目标力时自动减速/停止，
#   无需上位机轮询传感器，响应快且不会过冲。
#
#   若安装受阻（下探深度不够 / 运动超时），则关闭力控 → 上抬 → 旋转 → 重试。
#   若下探深度达到 FC_INSERT_DEPTH_MM 则认为安装成功。
# ===========================================================================

def _enable_force_control(robot: jkrc.RC):
    """配置并启用 JAKA 内置力控模式。"""
    log.info("启用 JAKA 内置力控  axis=%d  target_force=%.1fN",
             FC_AXIS, FC_TARGET_FORCE)

    # 1. 力传感器零位校准（消除重力偏移）
    robot.robot_zero_ftsensor_with_robot(FC_ROBOT_ID, FC_SENSOR_ID)
    time.sleep(0.3)

    # 2. 配置力控参数
    #    axis:    0=X  1=Y  2=Z  3=Rx  4=Ry  5=Rz（TCP 坐标系）
    #    opt:     0=禁用  1=启用
    #    ft_user: 目标力（N）或目标力矩（Nm）
    #    ft_rebound: 回弹系数
    #    ft_constant: 力常数
    #    ft_normal_track: 法向跟踪
    robot.set_ft_ctrl_config(
        FC_AXIS,                # axis
        1,                      # opt = 启用
        FC_TARGET_FORCE,        # ft_user
        FC_REBOUND,             # ft_rebound
        FC_CONSTANT,            # ft_constant
        FC_NORMAL_TRACK,        # ft_normal_track
    )

    # 3. 设置力控速度上限（防止冲撞）
    robot.set_force_ctrl_vel_limit(*FC_VEL_LIMIT, robot_id=FC_ROBOT_ID)

    # 4. 设置力控容差
    robot.set_cst_force_ctrl_tol(*FC_TOL, robot_id=FC_ROBOT_ID)

    # 5. 设置力控坐标系为 TCP 坐标系
    robot.set_ft_ctrl_mode(0)   # 0=TCP 坐标系

    # 6. 启用力控
    robot.robot_enable_force_control(FC_ROBOT_ID)
    time.sleep(0.2)
    log.info("力控已启用")


def _disable_force_control(robot: jkrc.RC):
    """关闭 JAKA 内置力控模式。"""
    log.info("关闭力控")
    try:
        robot.robot_disable_force_control(FC_ROBOT_ID)
    except Exception as e:
        log.warning("关闭力控异常（可能已自动关闭）: %s", e)


def force_insert(robot: jkrc.RC, max_retries: int = INSERT_MAX_RETRIES):
    """
    使用 JAKA 内置力控进行工件安装。

    流程：
      1. 启用力控（Z 轴目标力下压）
      2. 发出向下运动指令（控制器自动做力闭环）
      3. 运动结束后判断下探深度：
         - 达到 FC_INSERT_DEPTH_MM → 安装成功
         - 未达到 → 受阻，关闭力控，上抬 + 旋转后重试
      4. 循环直到成功或达到最大重试次数
    """
    for attempt in range(1, max_retries + 1):
        log.info("安装尝试 %d/%d", attempt, max_retries)

        # --- 记录运动前 Z 坐标 ---
        tcp_before = get_left_tcp(robot)
        if tcp_before is None:
            log.error("无法获取当前 TCP 位姿，终止安装")
            return False
        z_before = tcp_before[2]

        # --- 启用力控 ---
        _enable_force_control(robot)

        # --- 向下运动（控制器自动力闭环）---
        # 目标：在当前 Z 基础上下降 INSERT_DOWN_DISTANCE
        # 力控模式下，若接触力达到目标力，控制器会自动停止该方向运动
        target_pose = list(tcp_before)
        target_pose[2] = z_before - INSERT_DOWN_DISTANCE
        log.info("力控下探  起始Z=%.1f  目标Z=%.1f  速度=%.1f",
                 z_before, target_pose[2], INSERT_DOWN_SPEED)

        try:
            robot.robot_run_multi_movl(
                robot_id=0,                     # 0=左臂
                move_mode=[jkrc.MoveMode.ABS],
                end_pos=[tuple(target_pose)],
                is_block=True,
                vel=[INSERT_DOWN_SPEED],
                acc=[MOVE_ACCEL],
                tol=[0.0],
            )
        except Exception as e:
            log.warning("力控运动异常: %s", e)

        # --- 关闭力控 ---
        _disable_force_control(robot)

        # --- 判断安装深度 ---
        tcp_after = get_left_tcp(robot)
        if tcp_after is None:
            log.error("无法获取运动后 TCP 位姿")
            return False
        z_after = tcp_after[2]
        insert_depth = z_before - z_after  # 正值=下行距离

        log.info("下探深度: %.1f mm（阈值 %.1f mm）",
                 insert_depth, FC_INSERT_DEPTH_MM)

        if insert_depth >= FC_INSERT_DEPTH_MM:
            log.info("安装成功！（第 %d 次尝试，深度 %.1f mm）",
                     attempt, insert_depth)
            return True

        # --- 受阻：上抬 + 旋转重试 ---
        log.info("安装受阻（深度不足），上抬 %.1fmm + RZ 旋转 %.1f°",
                 INSERT_LIFT_MM, INSERT_ROTATE_DEG)

        # 上抬
        move_left_arm_linear_relative(
            robot, [0, 0, INSERT_LIFT_MM, 0, 0, 0], speed=MOVE_SPEED)

        # RZ 顺时针旋转
        move_left_arm_linear_relative(
            robot, [0, 0, 0, 0, 0, INSERT_ROTATE_DEG], speed=MOVE_SPEED)

        # 回到安装底座上方（补偿上抬距离）
        move_left_arm_linear_relative(
            robot, [0, 0, -INSERT_LIFT_MM, 0, 0, 0], speed=MOVE_SPEED)

    # 兜底：确保力控已关闭
    _disable_force_control(robot)
    log.warning("达到最大重试次数 %d，安装未成功", max_retries)
    return False


# ===========================================================================
#  步骤 11 详细实现：右臂取放
# ===========================================================================

def right_arm_pick_and_place(robot: jkrc.RC):
    """
    右臂来到工件位置，取回工件，并放置到另一侧。

    流程：
      1. 右臂运动到抓取前过渡位置
      2. 打开右臂夹爪
      3. 右臂直线运动到工件抓取位置
      4. 关闭右臂夹爪，等待 1 秒
      5. 右臂上抬 200mm
      6. 右臂运动到放置位置上方
      7. 右臂下降放置工件
      8. 打开右臂夹爪释放
    """
    # 1. 右臂运动到抓取前过渡位置
    log.info("右臂运动到抓取前过渡位置")
    if RightArmPose.PickReadyPose:
        move_right_arm_joint(robot, RightArmPose.PickReadyPose)
    else:
        log.warning("右臂抓取过渡位置未配置，跳过")
        return

    # 2. 打开右臂夹爪
    open_right_gripper(robot)

    # 3. 右臂直线运动到工件抓取位置
    log.info("右臂运动到工件抓取位置")
    if RightArmPose.PickPose:
        move_right_arm_linear(robot, RightArmPose.PickPose)
    else:
        log.warning("右臂抓取位置未配置，跳过")
        return

    # 4. 关闭右臂夹爪，等待 1 秒
    close_right_gripper(robot)  # 内部已 sleep 1 秒

    # 5. 右臂上抬 200mm
    log.info("右臂上抬 200mm")
    move_right_arm_linear_relative(robot, [0, 0, PICK_LIFT_MM, 0, 0, 0])

    # 6. 右臂运动到放置位置上方
    log.info("右臂运动到放置位置")
    if RightArmPose.PutReadyPose:
        move_right_arm_joint(robot, RightArmPose.PutReadyPose)
    else:
        log.warning("右臂放置过渡位置未配置，跳过")
        return

    # 7. 右臂下降放置工件
    log.info("右臂下降放置")
    if RightArmPose.PutPose:
        move_right_arm_linear(robot, RightArmPose.PutPose)
    else:
        log.warning("右臂放置位置未配置，跳过")
        return

    # 8. 打开右臂夹爪释放
    open_right_gripper(robot)

    # 上抬撤离
    move_right_arm_linear_relative(robot, [0, 0, PICK_LIFT_MM, 0, 0, 0])


# ===========================================================================
#  入口
# ===========================================================================

if __name__ == "__main__":
    run_assembly()
