# experiments_framework
This repository concentrates the experiments related to the research thesis titled ***Top View Aided Navigation in Orchards*** which was submitted to the Senate of the Technion - Israel Institute of Technology in May 2019.

**Author:** Omer Shalev

**Advisor:** Assistant Professor Amir Degani

## Installation
The experiments were tested on Ubuntu 16.04 with ROS Kinetic.

Follow the instructions below to install all dependencies:
```
# Create catkin workspace
mkdir -p ~/orchards_ws/src
cd ~/orchards_ws
catkin_init_workspace

# Clone dependencies
git clone https://github.com/omershalev/air_ground_orchard_navigation
git clone https://github.com/omershalev/nelder_mead
git clone https://github.com/omershalev/astar
git clone https://github.com/omershalev/navigation -b kinetic-devel

# Clone this repository
git clone https://github.com/omershalev/experiments_framework

# Build
catkin_make
```
