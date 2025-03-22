import math
import copy
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Vector3, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class Final(Node):
    def __init__(self):
        super().__init__('final')

        self.publisher_ = self.create_publisher(Odometry, "my_odom", 10)
        self.subscriber_= self.create_subscription(LaserScan, "diff_drive/scan", self.scan_callback, 10)

        self.prev_points = []

    def scan_callback(self, scan):
        print(scan)

def main(args=None):
    rclpy.init(args=args)

    final = Final()

    rclpy.spin(final)

    final.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

