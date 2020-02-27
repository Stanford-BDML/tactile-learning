# Repository for UR5 + Robotiq Demo
Demo code for controlling the UR5 and Robotic 2F gripper in BDML with MoveIt!

## Requirements:
    + ROS melodic (runs well on Ubuntu 18.04)
    + ROS MoveIt!

What's in side the package:
This repository contains a whole catkin workspace. Within it there are three
ROS packages.
1. Robotiq's ROS-industrial package (cloned on 2019) 
2. Universal Robots's ROS-industrial package (cloned on 2019) 
3. UR5 rospy interface

## Compilation:
1. First time compiling this ROS workspace do:
```console
source /opt/ros/melodic/setup.bash
cd [path/to/demo/project/root]
catkin build
```

## Running this code:

1. source env
```console 
source [path/to/demo/project/root]/devel/setup.bash
```

2. Launch ur5 connection, MoveIt! and RVIZ
```console 
roslaunch ur5_demo demo.launch
```
This will (i) launch the ur5_bringup to open the TCP bridge to UR5, (ii) launch
the moveit ros node that will take in your joint or Cartesian commands and
perform the motion planning and (iii) run RVIZ so you can visualize motions of
the UR5 before you execute them.

3. Run demo script
```console 
rosrun ur5_demo ur5_demo.py
```
This will try to move the UR5 to its home position. It will first plan the path
without executing it and it will wait for user to press Enter to proceed with
execution. At this time you can look at the RVIZ interface to see what the path
will be. Tip: To re-play to path in case you missed it, uncheck the motion planning
item on the left pane and check it again.


## How to adapt this to your code?
Please look at the ur5_demo.py and how you initialize a ROS node with the ur5
interface. There are a few examples of how to commend it to go to a cartesian
pose or a joint pose.

