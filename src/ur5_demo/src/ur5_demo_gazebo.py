#!/usr/bin/env python

""" ur5_calib.py
    Script used to moving the robot to calibrate wrt to realsense camera.
    This assumes that we have already launched the ur5 bringup node
    author: Michael Andres Lin (michaelv03@gmail.com)
    date: 10/31/2019
"""

import os
import sys
import time
import rospy
import numpy as np
import copy
import moveit_commander

from ur5_interface import UR5Interface
from robotiq_interface import RobotiqInterface

### Global definitions

INTER_COMMAND_DELAY = 4

### end global definitions

def test_move_home():
    """
    Function to demonstrate moving the ur5 to home pose
    """
    # Initialize the ros node
    rospy.init_node("test_move_home", anonymous=True, disable_signals=True)

    # Instantiage the UR5 interface.
    ur5 = UR5Interface()

    # MoveIt! works well if joint limits are smaller (within -pi, pi)
    if not ur5.check_joint_limits():
        raise Exception('Bad joint limits! try running roslaunch with option "limited:=true"')

    # go to home and print the joint values
    ur5.goto_home_pose()
    print(ur5.get_joint_values())


def test_robotiq_gripper():
    """
    Function to demonstrate robotiq gripper usage
    """
    # Initialize the ros node
    rospy.init_node("test_robotiq_gripper", anonymous=True, disable_signals=True)

    # Instantiage the Robotiq gripper interface.
    gripper = RobotiqInterface()

    # Command the gripper to go to fully open
    gripper.goto_gripper_pos(0)

    # Command the gripper to go to fully close
    gripper.goto_gripper_pos(255)

    # The robotiq position command ranges from 0 to 255
    gripper.goto_gripper_pos(126)

def test_robotiq_gripper_gazebo():
    # Initialize the ros node
    rospy.init_node("test_robotiq_gripper_gazebo", anonymous=True, disable_signals=True)

    # Instantiage the Robotiq gripper interface.
    grp = moveit_commander.MoveGroupCommander("gripper")

    # Command the gripper to go to fully close
    grp.set_named_target('close')
    grp.go(wait=True)

    # Command the gripper to go to fully open
    grp.set_named_target('open')
    grp.go(wait=True)

def test_move_ur5():
    rospy.init_node("test_move_ur5", anonymous=True, disable_signals=True)

    ur5 = UR5Interface()

    # MoveIt! works well if joint limits are smaller (within -pi, pi)
    if not ur5.check_joint_limits():
        raise Exception('Bad joint limits! try running roslaunch with option "limited:=true"')


    current_pose = ur5.get_pose()
    print("============ Current pose: %s" % current_pose)
    

    while(1):
        ### go to P1
        ur5.goto_home_pose()
        home_pose = ur5.get_pose()

        ### go to P2
        P2_pose = copy.deepcopy(home_pose)
        P2_pose.position.x += 0.1
        ur5.goto_pose_target(P2_pose)

        ### go to P3
        P3_pose = copy.deepcopy(home_pose)
        P3_pose.position.z += 0.1
        ur5.goto_pose_target(P3_pose)

def test_move_ur5_continuous():
    rospy.init_node("test_move_ur5_continuous", anonymous=True, disable_signals=True)

    ur5 = UR5Interface()

    # MoveIt! works well if joint limits are smaller (within -pi, pi)
    if not ur5.check_joint_limits():
        raise Exception('Bad joint limits! try running roslaunch with option "limited:=true"')


    ur5.goto_home_pose()
    # predefine all the poses that we want to go to
    home_pose = ur5.get_pose()

    P2_pose = copy.deepcopy(home_pose)
    P2_pose.position.x += 0.1

    P3_pose = copy.deepcopy(home_pose)
    P3_pose.position.z += 0.1

   # print("============ Current pose: %s" % current_pose)

    # The following commands are just to stall the script
    print("============ Press `Enter` to continue the movement ...")
    raw_input()

    # loop through the waypoints
    while(1):
        ### go to P1
        ur5.goto_home_pose(wait=False)
        time.sleep(INTER_COMMAND_DELAY)

        ### go to P2
        ur5.goto_pose_target(P2_pose, wait=False)
        time.sleep(INTER_COMMAND_DELAY)

        ### go to P3
        ur5.goto_pose_target(P3_pose, wait=False)
        time.sleep(INTER_COMMAND_DELAY)


curr_demo = 3
if __name__ == '__main__': 
    if (curr_demo == 1):
        test_move_home()
    elif (curr_demo == 2):
        test_robotiq_gripper()
    elif (curr_demo == 3):
        test_move_ur5()
    elif (curr_demo == 4):
        test_move_ur5_continuous()
    elif (curr_demo == 5):
        test_robotiq_gripper_gazebo()
