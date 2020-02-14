#!/usr/bin/env python

""" robotiq_interface.py
    script used to define a simple and easy interface with Robotiq 2f-85
    author: Michael Anres Lin (michaelv03@gmail.com)
    date: 11/14/2019
"""

import os
import sys
import time
import signal
import roslib; roslib.load_manifest('robotiq_2f_gripper_control')
import rospy
import subprocess
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output  as outputMsg


class RobotiqInterface:
    """ An interface class for Robotiq 2F-85 """

    def __init__(self):
        # run the robotiq controller process
        os.system("sudo chmod 777 /dev/ttyUSB0")
        self.proc = subprocess.Popen(['exec rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py /dev/ttyUSB0'], stdout=subprocess.PIPE, \
                                       shell=True)

        time.sleep(2)
        print("Robotiq Gripper Started")

        self.gripper_pub = rospy.Publisher('Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output, queue_size = 3)

        # perform a gripper reset
        print("============ Gripper about to reset Press `Enter` to continue  ...")
        raw_input()

        self.command = outputMsg.Robotiq2FGripper_robot_output();
        self.command.rACT = 0
        self.gripper_pub.publish(self.command)
        time.sleep(1)
        self.command = outputMsg.Robotiq2FGripper_robot_output();
        self.command.rACT = 1
        self.command.rGTO = 1
        self.command.rSP  = 180
        self.command.rFR  = 150
        self.gripper_pub.publish(self.command)



    def __del__(self):
        self.proc.kill()

    def goto_gripper_pos(self, pos, wait=True):
        if (wait):
            print("============ Gripper about to move. Press `Enter` to continue  ...")
            raw_input()
        if (pos < 0) or (pos > 255):
            print("Robotiq Gripper bad position input")
        self.command.rPR = pos
        self.gripper_pub.publish(self.command)


