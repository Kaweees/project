#!/usr/bin/env python3

import math
import copy
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Vector3, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class Final(Node):
    def __init__(self):
        super().__init__("final")

        self.publisher_ = self.create_publisher(Odometry, "my_odom", 10)
        self.subscriber_ = self.create_subscription(
            LaserScan, "diff_drive/scan", self.scan_callback, 10
        )

        self.prev_points = []

    def distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def scan_callback(self, scan):
        if len(self.prev_points) == 0:
            points = self.cart_points(scan)
            self.prev_points = points
            return

        points = self.cart_points(scan)
        points = [p for p in points if p is not None]

        closest = self.closest_points(points)

        print(self.center_of_mass(closest[1])[0] - self.center_of_mass(closest[0])[0])
        print(self.center_of_mass(closest[1])[1] - self.center_of_mass(closest[0])[1])

        self.prev_points = points

    # Generates cartesian points from scan data
    def cart_points(self, scan):
        points = []

        angle = scan.angle_min
        for r in scan.ranges:
            if (
                math.isnan(r)
                or math.isinf(r)
                or r < scan.range_min
                or r > scan.range_max
            ):
                points.append(None)
            else:
                points.append((r * math.cos(angle), r * math.sin(angle)))
            angle += scan.angle_increment

        return points

    def closest_points(self, points):
        curr = []
        prev = []
        for p1 in points:
            min_distance = 1000000000
            min_p2 = None
            for p2 in self.prev_points:
                distance = self.distance(p1, p2)
                if distance < min_distance:
                    min_distance = distance
                    min_p2 = p2

            curr.append(p1)
            prev.append(min_p2)

        return (curr, prev)

    def center_of_mass(self, points):
        return tuple(sum(col) / len(col) for col in zip(*points))


def main(args=None):
    rclpy.init(args=args)

    final = Final()

    rclpy.spin(final)

    final.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
