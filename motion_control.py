#!/usr/bin/env python
# A0158305H

from numpy import *
from math import *

pos_error_sum = 0
phi_error_sum = 0
prev_pos_error = 0
prev_phi_error = 0
#t = rospy.get_time()

kp_r = 1
kd_r = 1
ki_r = 1

kp_phi = 1
ki_phi = 1
kd_phi = 1

init_now = False # Is initialisation needed
print (pos_error_sum)


#moving to the target position from current position
def move_to(x_t, y_t, x_p, y_p, phi_t, init_now  = False):
    
    global pos_error_sum, phi_error_sum, prev_pos_error, prev_phi_error
    
    if init_now:
        initialise()  
    
    # Calculating Linear Velocity value
    pos_error = sqrt((y_p - y_t)*(y_p - y_t) + (x_p - x_t)*(x_p - x_t))
    pt_r = kp_r*pos_error

    it_r = ki_r*pos_error_sum   
    pos_error_sum += pos_error

    dt_r = kd_r*(pos_error - prev_pos_error)
    prev_pos_error = pos_error

    v_forward = pt_r + it_r + dt_r # Final velocity to be published
    print(v_forward)

    # Calculating Angular Velocity value
    phi_error = atan2((y_p - y_t), (x_p - x_t)) - phi_t
    if phi_error >= pi:
        phi_error -= 2*pi
    elif phi_error < -pi:
        phi_error += 2*pi

    pt_phi = kp_phi*phi_error

    it_phi = ki_phi*phi_error_sum   
    phi_error_sum += phi_error

    dt_phi = kd_phi*(phi_error - prev_phi_error)
    prev_phi_error = phi_error

    omega = pt_phi + it_phi + dt_phi # Final omega to be published
    print(omega)

    return(v_forward, omega)



def initialise():

    return init_now