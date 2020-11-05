# Repository for UR5 + Robotiq Demo and RL for dooropening task in Gazebo
Demo code for controlling the UR5 and Robotic 2F gripper in BDML with MoveIt!

## Requirements:
    + ROS melodic (runs well on Ubuntu 18.04)
    + ROS MoveIt!
    + ROS Gazebo
    + Gazebo Grasp Plugin
    + OpenAI ROS

What's in side the package:
This repository contains a whole catkin workspace. Within it there are three
ROS packages.
1. Robotiq's ROS-industrial package (cloned on 2019) 
2. Universal Robots's ROS-industrial package (cloned on 2019) 
3. UR5 rospy interface
4. gazebo_ros_pkgs
5. gazebo-pkgs
 ```console
 cd [project dir/src] 
 git clone https://github.com/JenniferBuehler/gazebo-pkgs.git
 catkin build
 *delete "gazebo_state_plugins" and "gazebo_test_tools"
6. openai_ros
 ```console
 cd [project dir/src] 
 git clone https://bitbucket.org/theconstructcore/openai_ros.git
 cd ..
 catkin build
 source ./devel/setup.bash
 rosdep install openai_ros
 pip install empty
 catkin build
 ```

## Make virtual env:
1. Make Virtual Env in python2
```console 
cd [project dir] 
virtualenv venv 
```

2. Activate/Deactivate 
```console 
source venv/bin/activate 
```
```console 
([newenvname])$ deactivate
```

## Activate the virtual env and install the packages by pip install:
1. numpy==(1.16.6)
2. matplotlib==(2.2.5)
3. sklearn-utils==(0.0.15)
4. tensorflow==1.14.0       *the version is important
5. tensorflow-gpu==1.14.0   *the version is important
6. pyyaml==(5.3.1)
7. rospkg==(1.2.8)
8. gym==(0.16.0)


## Compilation:
1. First time compiling this ROS workspace do:
```console
source /opt/ros/melodic/setup.bash
cd [path/to/demo/project/root]
catkin build
```

## Running this code for RL door opening task simulation:
1. source env
```console 
source [path/to/demo/project/root]/devel/setup.bash
```

2. Launch ur5 connection and Gazebo
```console 
roslaunch ur5_demo openai_ur_robotiq_gazebo.launch
```
This will (i) launch the ur5_bringup to open the TCP bridge to UR5, and (ii) launch the Gazebo simulator.

3. Run RL script
```console 
python ppo_gae_main.py
```
This will try to move the UR5 and the robotiq gripper to open the door.


## Running this code for position control simulation:

1. source env
```console 
source [path/to/demo/project/root]/devel/setup.bash
```

2. Launch ur5 connection, MoveIt!, RVIZ, and Gazebo
```console 
roslaunch ur5_demo demo_ur_robotiq_gazebo.launch
```
This will (i) launch the ur5_bringup to open the TCP bridge to UR5, (ii) launch
the moveit ros node that will take in your joint or Cartesian commands and
perform the motion planning, (iii) run RVIZ so you can visualize motions of
the UR5 before you execute them, and (iv) launch the Gazebo simulator.

3. Run demo script
```console 
rosrun ur5_demo ur5_demo_gazebo.py
```
This will try to move the UR5 to its home position. It will first plan the path
without executing it and it will wait for user to press Enter to proceed with
execution. At this time you can look at the RVIZ interface to see what the path
will be. Tip: To re-play to path in case you missed it, uncheck the motion planning
item on the left pane and check it again.

