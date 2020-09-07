#!/usr/bin/env python

""" ur5_interface.py
    script used to define a simple and easy interface with UR5
    author: Michael Anres Lin (michaelv03@gmail.com)
    date: 10/31/2019
"""

import os
import sys
import time

from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
import roslib; roslib.load_manifest('ur_driver')
import rospy
import moveit_commander
import moveit_msgs.msg
import numpy as np
from std_msgs.msg import Header

import numpy as np
from scipy.spatial.transform import Rotation as R


### Global definitions

INTER_COMMAND_DELAY = 4

### end global definitions

class UR5Interface:
    """ An interface class for UR5 """

    joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                   'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    joint_values_home = [1.5450898256183896, -1.810624635807664, 2.258478488681325, 
                        -2.0176230710216156, -1.5706590472860515, 3.1148891041905493]

    joint_values_calib = [1.544891846222953, -1.8106006673578534, 2.258661677945332, 
                      -2.0195989461345256, -1.5703836018608142, 2.3299036774634354]

    joint_values_spec = [1.4034008128830147, -1.667990371279239, 2.0688124016401552, -0.40665629147605475, 1.3692966421147243, -1.5738457779422106]

    joint_values_before_grasp = [1.0936292324911197, -1.9619247342409523, 2.284918129311265, -0.3295925202225334, 1.0594761648023008, -1.5717792992464688]

    joint_values_grasp_position = [1.2168549330235976, -1.7469892316131368, 2.1438566966530352, -0.4029605278499071, 1.18284603228705, -1.5726680497015089]

    joint_values_rotate = [1.2271511722945796, -1.7471523185028532, 2.147928955121685, -0.41287096288384983, 1.1990159658338717, -0.25820784403716246]

    def __init__(self):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander("manipulator")

        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                        moveit_msgs.msg.DisplayTrajectory,
                                                        queue_size=20)

        self.goal_state_publisher = rospy.Publisher('/rviz/moveit/update_custom_goal_state',
                                                        moveit_msgs.msg.RobotState,
                                                        queue_size=20)

        # Walls are defined with respect to the coordinate frame of the robot base, with directions 
        # corresponding to standing behind the robot and facing into the table.

        rospy.sleep(0.6)
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'


    def get_pose(self):
        """ get robot end effector pose """
        return self.group.get_current_pose().pose

    def get_rpy(self):
        """ get robot end effector rpy """
        return self.group.get_current_rpy()

    def get_pose_array(self):
        """ get robot end effector pose as two arrays of position and
        orientation 
        """
        pose = self.group.get_current_pose().pose
        return np.array([pose.position.x, pose.position.y, 
                        pose.position.z]), \
               np.array([pose.orientation.x, pose.orientation.y,
                        pose.orientation.z, pose.orientation.w])

    def get_joint_values(self):
        """ get robot joint values """
        return self.group.get_current_joint_values()

    def goto_home_pose(self, wait=True):
        """ go to robot end effector home pose """
        self.goto_joint_target(self.joint_values_home, wait=wait)

    def goto_before_grasp_pose(self, wait=False):
        """ go to robot end effector before grasp """
        self.goto_joint_target(self.joint_values_before_grasp, wait=wait)

    def goto_grasp_position_pose(self, wait=False):
        """ go to robot end effector grasp position pose """
        self.goto_joint_target(self.joint_values_grasp_position, wait=wait)

    def goto_rotate_pose(self, wait=False):
        """ go to robot end effector rotate pose """
        self.goto_joint_target(self.joint_values_rotate, wait=wait)

    def goto_spec_pose(self, wait=True):
        """ go to robot spec pose """
        self.goto_joint_target(self.joint_values_spec, wait=wait)

    def goto_calib_home_pose(self):
        """ go to robot end effector home pose for calibration.
            This is basically the same as the home pose but the 
            frame is rotated by 45 deg in the x axis (axis
            defined in the ee frame and x pointing along rotation axis.
        """
        self.goto_joint_target(self.joint_values_calib, wait=True)

    def goto_pose_target(self, pose, wait=False):
        """ go to robot end effector pose target """
        self.group.set_pose_target(pose)
        # simulate in rviz then ask user for feedback
        plan = self.group.plan()
        self.display_trajectory(plan)
        if (wait == True):
            print("============ Press `Enter` to execute the movement ...")
            raw_input()
        self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

    def goto_pose_array_target(self, pos, ori, wait=True):
        assert len(pos) == 3,\
              "Error: pos array must be length 3"
        assert len(ori) == 4,\
              "Error: orientation (quat) array must be length 4"
        new_pose = Pose()
        new_pose.position.x = pos[0]
        new_pose.position.y = pos[1]
        new_pose.position.z = pos[2]
        new_pose.orientation.x = ori[0]
        new_pose.orientation.y = ori[1]
        new_pose.orientation.z = ori[2]
        new_pose.orientation.w = ori[3]
        self.goto_pose_target(new_pose, wait=wait)

    def goto_joint_target(self, joint_vals, wait=True):
        """ go to robot end effector joint target """
        self.group.set_joint_value_target(joint_vals)
        # simulate in rviz then ask user for feedback
        plan = self.group.plan()
        self.display_trajectory(plan)
        if (wait == True):
            print("============ Press `Enter` to execute the movement ...")
            raw_input()
        self.group.execute(plan, wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

    def stop(self):
        self.group.stop()
        self.group.clear_pose_targets()

    def display_trajectory(self, plan):
        """ displays planned trajectory in rviz """
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        robot = self.robot
        display_trajectory_publisher = self.display_trajectory_publisher

        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        display_trajectory_publisher.publish(display_trajectory)
        # not working for goal state yet
        robot_goal_state = moveit_msgs.msg.RobotState()
        robot_goal_state.joint_state.position = plan.joint_trajectory.points[-1].positions
        self.goal_state_publisher.publish(robot_goal_state)
