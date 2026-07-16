import time

import jkrc

from conf import *



Robot = jkrc.RC(ROBOT_HOST)
Robot.login(True, ROBOT_USER, ROBOT_PASSWORD)


Robot.power_on()
Robot.enable_robot()

ret = Robot.send_tio_rs_command(1,INIT_GRIPPER,1)
Robot.send_tio_rs_command(1,INIT_GRIPPER,0)
# time.sleep(0.5)

# print(ret)
# Robot.logout()

# ret = Robot.get_robot_status()
# print("left: ",ret)
# ret = Robot.get_tcp_position()

# print("Right:",ret)

Robot.logout()