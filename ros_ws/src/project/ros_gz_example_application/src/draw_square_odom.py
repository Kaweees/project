#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
from transforms3d.euler import quat2euler as euler_from_quaternion


class DrawSquareOdom(Node):
    # States for the FSM
    MOVING = 0
    TURNING = 1
    FINISHED = 2

    def __init__(self):
        super().__init__("draw_square_odom")

        # Create publisher for velocity commands
        self.vel_publisher = self.create_publisher(Twist, "/diff_drive/cmd_vel", 10)

        # Subscribe to odometry
        self.odom_subscription = self.create_subscription(
            Odometry, "/diff_drive/odometry", self.odom_callback, 10
        )

        # Initialize state variables
        self.current_state = self.MOVING
        self.side_count = 0
        self.current_yaw = 0.0
        self.target_yaw = 0.0
        self.side_length = 2.0  # meters
        self.start_position = None
        self.distance_traveled = 0.0

        # Control parameters
        self.linear_speed = 0.3  # m/s
        self.angular_speed = 0.5  # rad/s
        self.yaw_tolerance = 0.05  # radians
        self.distance_tolerance = 0.1  # meters

        # Create timer for control loop
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz control loop

    def odom_callback(self, msg):
        # Extract position
        position = msg.pose.pose.position

        # Convert quaternion to Euler angles
        orientation_q = msg.pose.pose.orientation
        orientation_list = [
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w,
        ]
        _, _, self.current_yaw = euler_from_quaternion(orientation_list)

        # Initialize start position if not set
        if self.start_position is None:
            self.start_position = position
            return

        # Calculate distance traveled when moving
        if self.current_state == self.MOVING:
            dx = position.x - self.start_position.x
            dy = position.y - self.start_position.y
            self.distance_traveled = math.sqrt(dx**2 + dy**2)

    def timer_callback(self):
        msg = Twist()

        if self.current_state == self.MOVING:
            if self.distance_traveled < self.side_length:
                # Move forward
                msg.linear.x = self.linear_speed
                self.get_logger().info(
                    f"Moving forward: {self.distance_traveled:.2f}/{self.side_length:.2f} meters"
                )
            else:
                # Prepare for turning
                self.current_state = self.TURNING
                self.target_yaw = self.current_yaw + math.pi / 2  # 90 degrees
                if self.target_yaw > math.pi:
                    self.target_yaw -= 2 * math.pi
                self.start_position = None
                self.distance_traveled = 0.0

        elif self.current_state == self.TURNING:
            # Calculate the error in yaw
            yaw_error = self.target_yaw - self.current_yaw
            if abs(yaw_error) > math.pi:
                yaw_error = yaw_error - (2 * math.pi if yaw_error > 0 else -2 * math.pi)

            if abs(yaw_error) > self.yaw_tolerance:
                # Turn using proportional control
                msg.angular.z = self.angular_speed * (yaw_error / abs(yaw_error))
                self.get_logger().info(
                    f"Turning: Current={math.degrees(self.current_yaw):.1f}°, Target={math.degrees(self.target_yaw):.1f}°"
                )
            else:
                # Finished turning, prepare for next side
                self.current_state = self.MOVING
                self.side_count += 1
                if self.side_count >= 4:
                    self.current_state = self.FINISHED
                    self.get_logger().info("Square completed!")

        self.vel_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DrawSquareOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
