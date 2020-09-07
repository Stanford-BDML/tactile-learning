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
from gazebo_msgs.srv import GetJointProperties
from gazebo_msgs.srv import ApplyJointEffort
from gazebo_msgs.srv import JointRequest
from std_msgs.msg import Float64
import math

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

def test():
    # Initialize the ros node
    rospy.init_node("test", anonymous=True, disable_signals=True)

    # Instantiage the UR5 interface.
    ur5 = UR5Interface()
    grp = moveit_commander.MoveGroupCommander("gripper")

    print(ur5.get_rpy())
    print(ur5.get_pose())
    print(ur5.get_joint_values())
#    print(ur5.get_pose_array())    

def open_by_control():
    # Initialize the ros node
    rospy.init_node("open_by_control", anonymous=True, disable_signals=True)

    # Instantiage the UR5 and gripper interface.
    ur5 = UR5Interface()
    grp = moveit_commander.MoveGroupCommander("gripper")

    # MoveIt! works well if joint limits are smaller (within -pi, pi)
    if not ur5.check_joint_limits():
        raise Exception('Bad joint limits! try running roslaunch with option "limited:=true"')

    eelink_pose_before_grasp = [-0.00545284639771, 0.340081666162, 0.26178413889301, 1.570795, 0, 1.570795]
    ur5.goto_pose_target(eelink_pose_before_grasp)
    eelink_pose_grasp_position = [-0.00545284639771, 0.390081666162, 0.26178413889301, 1.570795, 0, 1.570795]
    ur5.goto_pose_target(eelink_pose_grasp_position)

    grp.set_named_target('close0.4')
    grp.go(wait=True)

    eelink_pose_after_rotate = [-0.00545284639771, 0.390081666162, 0.26178413889301, 3, 0, 1.570795]
    ur5.goto_pose_target(eelink_pose_after_rotate)
    eelink_pose_after_pull = [-0.086040, 0.294412, 0.260207, 3, 0, 1.202130]
    ur5.goto_pose_target(eelink_pose_after_pull)
    print(ur5.get_joint_values())

    grp.set_named_target('open')
    grp.go(wait=True)

def get_ref_callback(data):
    global angle_ref
    angle_ref = data.data

def test_robotiq_gripper_gazebo_force():
    # Initialize the ros node
    Ts = 0.001
    omega = 2*math.pi*2

    rospy.init_node("test_robotiq_gripper_gazebo_force", anonymous=True, disable_signals=True)
    rate = rospy.Rate(1.0/Ts)
    rospy.Subscriber('angle_ref', Float64, get_ref_callback)

    get_angle = rospy.ServiceProxy('/gazebo/get_joint_properties', GetJointProperties)
    clear_torque = rospy.ServiceProxy('/gazebo/clear_joint_forces', JointRequest)
    set_torque = rospy.ServiceProxy('/gazebo/apply_joint_effort', ApplyJointEffort)

    Kd = 3*omega**1
    Kp = 3*omega**2
    Ki = 1*omega**3

    torque = 0
    u_z = [0, 0]
    y_z = [0, 0]
    d_init = True

    angle_ref = 0
    
    while not rospy.is_shutdown():
       angle = get_angle(joint_name=joint_name).position[0]
       u_z[0] = angle_ref - angle
       y_z[0] = Ts*sum(u_z)/2 + y_z[1]

       if d_init:
         torque = Kp*u_z[0] + Ki*y_z[0]
         d_init = False
       else:
         torque = Kp*u_z[0] + Ki*y_z[0] + Kd*(u_z[0]-u_z[1])/Ts

       u_z[1] = u_z[0]
       y_z[1] = y_z[0]

       clear_torque(joint_name=joint_name)
       set_torque(joint_name=joint_name, effort=torque, duration=rospy.Duration(-1))
       rate.sleep()

    # Instantiage the Robotiq gripper interface.
#    grp = moveit_commander.MoveGroupCommander("gripper")

    # set torque to each joint
#    rospy.wait_for_service('/gazebo/apply_joint_effort')
#    set_torque("simple_gripper_right_follower_joint", torque, rospy.Duration.from_sec(0), rospy.Duration.from_sec(-1))
#    clear_torque("simple_gripper_right_driver_joint")


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

curr_demo = 8
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
    elif (curr_demo == 6):
        test_robotiq_gripper_gazebo_force()
    elif (curr_demo == 7):
        test()
    elif (curr_demo == 8):
        open_by_control()
