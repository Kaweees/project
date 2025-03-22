from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("ros_gz_example_bringup"),
                        "launch",
                        "diff_drive.launch.py",
                    )
                )
            ),
            Node(
                package="ros_gz_example_application",
                executable="teleop_keyboard",
                name="teleop_keyboard",
            ),
            Node(
                package="rqt_graph",
                executable="rqt_graph",
                name="rqt_graph",
                output="screen",
            ),
            Node(
                package="final",
                executable="final",
                name="final",
                output="screen",
            ),
        ]
    )
