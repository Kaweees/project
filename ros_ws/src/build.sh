#!/bin/bash

# Builds a ROS2 package and its nodes
# Usage:
#    build.sh {package_name} {node_name1} {node_name2} ...

#
#  Private Impl
#

# If less than 2 arguments are provided, print usage and exit
if [ $# -lt 2 ]; then
  echo "Usage: $0 <package_name> <node_name1> <node_name2> ..."
  exit 1
fi

package_name=$1
# shift  # Remove first argument (package_name)

# Build the package
colcon build --packages-select $package_name

# Source the project based on SHELL
# if [ "$SHELL" = "/bin/zsh" ]; then
#   source ./install/setup.zsh
# else
#   source ./install/setup.bash
# fi
source ./install/setup.zsh

# Run each node
for node in "$@"; do
  ros2 run $package_name "$node" &
done

# Wait for all background processes to complete
wait
