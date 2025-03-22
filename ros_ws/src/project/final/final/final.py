#!/usr/bin/env python3

import math
import copy
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, Vector3, Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

import numpy as np
from sklearn.neighbors import NearestNeighbors


class Final(Node):
    def __init__(self):
        super().__init__("final")

        self.publisher_ = self.create_publisher(Odometry, "my_odom", 10)
        self.subscriber_ = self.create_subscription(
            LaserScan, "diff_drive/scan", self.scan_callback, 10
        )

        self.x = 0
        self.y = 0
        self.theta = 0

        self.prev_points = []

    def quaternion_from_euler(self, pitch, roll, yaw):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)

        w = cy * cr * cp + sy * sr * sp
        x = cy * sr * cp - sy * cr * sp
        y = cy * cr * sp + sy * sr * cp
        z = sy * cr * cp - cy * sr * sp

        return (w, x, y, z)

    def distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def scan_callback(self, scan: LaserScan):
        if len(self.prev_points) == 0:
            points = self.cart_points(scan)
            self.prev_points = points
            return

        points = self.cart_points(scan)
        points = [p for p in points if p is not None]
        A = np.array(self.prev_points)
        B = np.array(points)
        if len(A) == 0 or len(B) == 0:
            return

        if len(A) > len(B):
            A = A[: len(B)]
        elif len(A) < len(B):
            B = B[: len(A)]
        T, _, _ = self.icp(A, B)


        self.x += math.sqrt(T[0][2] ** 2 + T[1][2] ** 2) * math.cos(self.theta)
        self.y += math.sqrt(T[0][2] ** 2 + T[1][2] ** 2) * math.sin(self.theta)
        self.theta -= math.atan2(T[1][0], T[0][0])

        self.get_logger().info(f"x: {self.x}, y: {self.y}, theta: {self.theta}")

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        q = self.quaternion_from_euler(0, 0, self.theta)
        odom.pose.pose.orientation.w = q[0]
        odom.pose.pose.orientation.x = q[1]
        odom.pose.pose.orientation.y = q[2]
        odom.pose.pose.orientation.z = q[3]

        self.publisher_.publish(odom)

        self.prev_points = points

    # Generates cartesian points from scan data
    def cart_points(self, scan: LaserScan) -> list[tuple[float, float] | None]:
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

    def best_fit_transform(
        self, A: np.ndarray, B: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m - 1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R, centroid_A.T)

        # homogeneous transformation
        T = np.identity(m + 1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t

    def nearest_neighbor(
        self, src: np.ndarray, dst: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    def icp(
        self,
        A: np.ndarray,
        B: np.ndarray,
        init_pose: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m + 1, A.shape[0]))
        dst = np.ones((m + 1, B.shape[0]))
        src[:m, :] = np.copy(A.T)
        dst[:m, :] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(20):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.nearest_neighbor(src[:m, :].T, dst[:m, :].T)

            # compute the transformation between the current source and nearest destination points
            T, _, _ = self.best_fit_transform(src[:m, :].T, dst[:m, indices].T)

            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < 0.001:
                break
            prev_error = mean_error

        # calculate final transformation
        T, _, _ = self.best_fit_transform(A, src[:m, :].T)

        return T, distances, i


def main(args=None):
    rclpy.init(args=args)

    final = Final()

    rclpy.spin(final)

    final.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
