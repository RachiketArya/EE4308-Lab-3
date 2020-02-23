#!/usr/bin/env python
# A0158305H
from numpy import *
import motion_control
from math import *

print (motion_control.initialise())
print (motion_control.move_to(0, 0, 5, 5, pi/2))

