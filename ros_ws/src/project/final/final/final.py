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

        self.prev_points = []

    def distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def scan_callback(self, scan: LaserScan):
        if len(self.prev_points) == 0:
            points = self.cart_points(scan)
            self.prev_points = points
            return

        # points = self.cart_points(scan)
        # points = [p for p in points if p is not None]

        # closest = self.closest_points(points)

        # print(self.center_of_mass(closest[1])[0] - self.center_of_mass(closest[0])[0])
        # print(self.center_of_mass(closest[1])[1] - self.center_of_mass(closest[0])[1])

        points = self.cart_points(scan)
        points = [p for p in points if p is not None]
        A = np.array(self.prev_points)
        B = np.array(points)
        if len(A) != 0 and len(B) != 0:
            if len(A) > len(B):
                A = A[: len(B)]
            elif len(A) < len(B):
                B = B[: len(A)]
            T, _, _ = self.icp(A, B)
            print(T)

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

    def closest_points(
        self, points: list[tuple[float, float] | None]
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
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

    def center_of_mass(self, points: list[tuple[float, float]]) -> tuple[float, float]:
        return tuple(sum(col) / len(col) for col in zip(*points))

    def get_closest_point(
        self, points: list[tuple[float, float]]
    ) -> tuple[float, float]:
        return np.linalg.svd(points)

    def best_fit_transform(
        self, A: np.ndarray, B: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
        Input:
        A: Nxm numpy array of corresponding points
        B: Nxm numpy array of corresponding points
        Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
        """

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
        """
        Find the nearest (Euclidean) neighbor in dst for each point in src
        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        """

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
        max_iterations: int = 20,
        tolerance: float = 0.001,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
        Input:
            A: Nxm numpy array of source mD points
            B: Nxm numpy array of destination mD point
            init_pose: (m+1)x(m+1) homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation that maps A on to B
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        """

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

        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.nearest_neighbor(src[:m, :].T, dst[:m, :].T)

            # compute the transformation between the current source and nearest destination points
            T, _, _ = self.best_fit_transform(src[:m, :].T, dst[:m, indices].T)

            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
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
