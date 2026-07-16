# 夹爪信号配置
INIT_GRIPPER = bytearray([0x1, 0x6, 0x1, 0x0, 0x0, 0x1, 0x49, 0xf6])
OPEN_GRIPPER = bytearray([0x1, 0x6, 0x1, 0x3, 0x3, 0xe8, 0x78, 0x88])
CLOSE_GRIPPER = bytearray([0x01,0x06,0x01,0x03,0x00, 0x00, 0x78, 0x36])

# 机器人登录配置
ROBOT_HOST = "192.168.1.50"
ROBOT_USER = "jaka_sdk"
ROBOT_PASSWORD = "Jaka123@"

# 左臂点位
class LeftArmPose:
    Home = [] # 安全位置

    PickReadyPose = [] # 拍照位置
    PutReadyPose = [] # 放置位置

    

# 右臂点位
class RightArmPose:
    Home = []           # 安全位置
    PickReadyPose = []  # 抓取前过渡位置
    PickPose = []       # 工件抓取位置
    PutReadyPose = []   # 放置前过渡位置
    PutPose = []        # 工件放置位置