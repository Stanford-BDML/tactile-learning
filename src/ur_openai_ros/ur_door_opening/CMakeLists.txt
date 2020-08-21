cmake_minimum_required(VERSION 2.8.3)
project(ur_door_opening)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package()
 
include_directories(
  ${catkin_INCLUDE_DIRS}
)


install(PROGRAMS
scripts/reaching_main.py
scripts/reinforcement_main.py
scripts/gazebo_execution.py

  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


