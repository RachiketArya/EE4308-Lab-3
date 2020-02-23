#!/usr/bin/env python
# A0158305H

from numpy import *
from math import *
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from tf2_msgs.msg import TFMessage

# ================================= CONSTANTS ==========================================
# Time Increase Constant
T_INC = 0.2

# PID Constants
KP_R = 0.4
KI_R = 0.001
KD_R = 0.02
KP_PHI = 1
KI_PHI = 0.001
KD_PHI = 0.001

x_t = y_t = phi_t = x_p = y_p = 0


def subscribe_true(msg):
    # subscribes to the robot's true position in the simulator. This should not be used, for checking only.
    global x_t, y_t, phi_t, x_p, y_p
    x_t = msg.transforms[0].transform.translation.x
    y_t = msg.transforms[0].transform.translation.y
    phi_t = euler_from_quaternion([
        msg.transforms[0].transform.rotation.x,
        msg.transforms[0].transform.rotation.y,
        msg.transforms[0].transform.rotation.z,
        msg.transforms[0].transform.rotation.w,
    ])[2]


# Subscribing to the odom value
def subscribe_odom(data):
    global x_t, y_t, phi_t, x_p, y_p
    x_t = data.pose.pose.position.x
    y_t = data.pose.pose.position.y
    phi_t = data.pose.pose.orientation.z


def subscribe_target(data):
    global x_t, y_t, phi_t, x_p, y_p
    x_p = data.x
    y_p = data.y


###
DEBUGGING = True


# moving to the target position from current position
def main():
    global x_t, y_t, phi_t, x_p, y_p

    # Node initialisation
    rospy.init_node('motion_control', anonymous=False)

    # Publisher & Subscriber
    vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    if not DEBUGGING:
        rospy.Subscriber("odom", Odometry, subscribe_odom, queue_size=1)
    rospy.Subscriber("immediate_target", Point, subscribe_target, queue_size=1)

    if DEBUGGING:
        rospy.Subscriber('tf', TFMessage, subscribe_true, queue_size=1)

    # Waiting or initialisation of robot variables maybe needed
    x_t = y_t = phi_t = 0
    x_p = y_p = 2

    # initialising variables
    pos_error = sqrt((y_p - y_t)*(y_p - y_t) + (x_p - x_t)*(x_p - x_t))
    pos_error_sum = pos_error
    prev_pos_error = pos_error
    phi_error = atan2((y_p - y_t), (x_p - x_t)) - phi_t
    phi_error_sum = phi_error
    prev_phi_error = phi_error
    run = True

    cmd_vel_value = Twist()

    # ---------------------------------- BEGIN ----------------------------------------------
    t = rospy.get_time()

    while (not rospy.is_shutdown()):  # required to Keyboard interrupt nicely

        if rospy.get_time() >= t and run:

            # Calculating Linear Velocity value
            pos_error = sqrt((y_p - y_t)*(y_p - y_t) + (x_p - x_t)*(x_p - x_t))
            pt_r = KP_R*pos_error

            it_r = KI_R*pos_error_sum
            pos_error_sum += pos_error

            dt_r = KD_R*(pos_error - prev_pos_error)
            prev_pos_error = pos_error

            v_forward = pt_r + it_r + dt_r  # Final velocity to be published
            print 'x: ', x_t, ', y: ', y_t
            print(pos_error)

            # Calculating Angular Velocity value
            phi_error = atan2((y_p - y_t), (x_p - x_t)) - phi_t
            if phi_error >= pi:
                phi_error -= 2*pi
            elif phi_error < -pi:
                phi_error += 2*pi

            pt_phi = KP_PHI*phi_error

            it_phi = KI_PHI*phi_error_sum
            phi_error_sum += phi_error

            dt_phi = KD_PHI*(phi_error - prev_phi_error)
            prev_phi_error = phi_error

            omega = pt_phi + it_phi + dt_phi  # Final omega to be published
            print 'omega: ', phi_t
            print(phi_error)

            if (phi_error < 0.05 and pos_error < 0.1):
                print("reached target")
                un = False
                v_forward = 0
                omega = 0

            # Publish the velocity values
            cmd_vel_value.linear.x = v_forward
            cmd_vel_value.angular.z = omega
            vel_pub.publish(cmd_vel_value)

            # increment the time counter
            et = rospy.get_time() - t
            print(et <= 0.2, et)

            t += T_INC


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
