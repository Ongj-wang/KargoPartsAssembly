import time

import jkrc

from conf import *
RobotIP = "192.168.1.50"
Robot = jkrc.RC(RobotIP)
ret = Robot.login(True, "jaka_sdk", "Jaka123@")
print("login:",ret)

# Robot.power_on()
# Robot.enable_robot()

# ret = Robot.send_tio_rs_command(1,INIT_GRIPPER,1)
# time.sleep(0.5)
Robot.send_tio_rs_command(1,OPEN_GRIPPER,0)
# print(ret)
# Robot.logout()

# ret = Robot.get_robot_status()
# print("left: ",ret)
# ret = Robot.get_tcp_position()

# print("Right:",ret)

Robot.logout()