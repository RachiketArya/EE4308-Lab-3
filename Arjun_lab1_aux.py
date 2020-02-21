#!/usr/bin/env python
# --- A0168924R_Arjun ---
import roslib, rospy, rospkg
from numpy import *
from geometry_msgs.msg import *
from sensor_msgs.msg import LaserScan, JointState #Imu
from nav_msgs.msg import Odometry
from std_msgs import *
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from tf2_msgs.msg import TFMessage
import cv2
import numpy
# ================================= CONSTANTS ==========================================        
# let's cache the SIN and POS so we don't keep recalculating it, which is slow
DEG2RAD = [i/180.0*pi for i in xrange(360)] # DEG2RAD[3] means 3 degrees in radians
SIN = [sin(DEG2RAD[i]) for i in xrange(360)] # SIN[32] means sin(32degrees)
COS = [cos(DEG2RAD[i]) for i in xrange(360)]

# ================================== DATA STRUCTS ===========================================
class OccupancyGrid: # Occupancy Grid
    def __init__(self, min_pos, max_pos, cell_size, initial_value): 
        """ Constructor for Occupancy Grid
        Parameters:
            min_pos (tuple of float64): The smallest world coordinate (x, y). This determines the lower corner of the rectangular grid
            max_pos (tuple of float64): The largest world coordinate (x, y). This determines the upper corner of the rectangular grid
            cell_size (float64): The size of the cell in the real world, in meters.
            initial_value (float64): The initial value that is assigned to all cells
        """
        di = int64(round((max_pos[0] - min_pos[0])/cell_size))
        dj = int64(round((max_pos[1] - min_pos[1])/cell_size))
        self.cell_size = cell_size
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.num_idx = (di+1, dj+1) # number of (rows, cols)
        self.m = ones((di, dj)) * initial_value # numpy ones
        # CV2 inits
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', di*5, dj*5) # so each cell is 5px*5px
    def idx2pos(self, idx):
        """ Converts indices (map indices) to position (world coordinates)
        Parameters:
            idx (tuple of float64 or tuple of int64): Index tuple (i, j)
        Returns:
            tuple of float64: (i, j)
        """
        w = self.cell_size
        mp = self.min_pos
        return (idx[0] * w + mp[0], idx[1] * w + mp[1])
    def pos2idx(self, pos, rounded=True): 
        """ Converts position (world coordinates) to indices (map indices)
        Parameters:
            pos (tuple of float64): Position tuple (x, y)
            rounded (bool): By default True. Set True to return a integer indices that can be used to access the array. Set False to return exact indices.
        Returns:
            tuple of int64 or tuple of float64: (i, j), exact indices or integer indices
        """
        w = self.cell_size
        mp = self.min_pos
        idx = ((pos[0] - mp[0])/w, (pos[1] - mp[1])/w)
        if rounded:
            return (int64(round(idx[0])), int64(round(idx[1])))
        return idx
    def idx_in_map(self, idx): # idx must be integer
        """ Checks if the given index is within map boundaries
        Parameters:
            idx (tuple of int64): Index tuple (i, j) to be checked
        Returns:
            bool: True if in map, False if outside map
        """
        i, j = idx
        return i >= 0 and i < self.num_idx[0] and j >= 0 and j < self.num_idx[1]
    def get_cell_idx(self, idx, is_int=False):
        """ Retrieves the value of the cell at a given index
        Parameters:
            idx (tuple of int64 or tuple of float64): Index tuple (i, j) of cell.
            is_int: By default False. If False, the tuple is converted to tuple of int64 by rounding. Set to True to skip this only if idx is tuple of int64
        Returns:
            None or float64: None if the idx is outside the map. float64 value of the cell if idx is in the map.
        """
        if not is_int:
            idx = (int64(round(idx[0])), int64(round(idx[1])))
        if self.idx_in_map(idx):
            return self.m[idx[0], idx[1]]
        return None
    def get_cell_pos(self, pos):
        """ Retrieves the value of the cell where the given position (world coordinates) is
        Parameters:
            pos (tuple of float64): position tuple (x, y).
        Returns:
            None or float64: None if pos is outside the map. float64 value of the cell if pos is in the map.
        """
        idx = self.pos2idx(pos)
        return self.get_cell_idx(idx, True)
    def set_cell_idx(self, idx, value, is_int=False):
        """ Sets the value of the cell at a given index
        Parameters:
            idx (tuple of int64 or tuple of float64): Index tuple (i, j) of cell.
            is_int: By default False. If False, the tuple is converted to tuple of int64 by rounding. Set to True to skip this only if idx is tuple of int64
        """
        if not is_int:
            idx = (int64(round(idx[0])), int64(round(idx[1])))
        if self.idx_in_map(idx):
            self.m[idx[0], idx[1]] = value
    def set_cell_pos(self, pos, value):
        """ Sets the value of the cell cell where the given position (world coordinates) is
        Parameters:
            pos (tuple of float64): position tuple (x, y).
            is_int: By default False. If False, the tuple is converted to tuple of int64 by rounding. Set to True to skip this only if idx is tuple of int64
        """
        idx = self.pos2idx(pos)
        return self.set_cell_idx(idx, value, True)
    def show_map(self, rbt_pos):
        """ Prints the occupancy grid and robot position on it as a picture in a resizable window
        Parameters:
            rbt_pos (tuple of float64): position tuple (x, y) of robot.
        """
        # normalise values in numpy matrix occ_grid.m to values between 0 and 255, + RGB.
        # img_mat = uint8(self.m * 255) # multiply by 255 from 0-1 to 0-255
        # img_mat = stack((img_mat, img_mat, img_mat), 2) # convert to RGB (3rd dim)
        # print(img_mat.shape) # (100, 100, 3)
        
        # color the robot position as a crosshair
        rbt_idx = self.pos2idx(rbt_pos) # robot index
        # img_mat[rbt_idx[0], rbt_idx[1], :] = (0, 255, 0) # green
        img_mat = self.m.copy()
        img_mat[rbt_idx] = 1 # white
        img_mat[rbt_idx] = 0
        img_mat[rbt_idx[0] - 1, rbt_idx[1]] = 1
        img_mat[rbt_idx[0] + 1, rbt_idx[1]] = 1
        img_mat[rbt_idx[0], rbt_idx[1] - 1] = 1
        img_mat[rbt_idx[0], rbt_idx[1] + 1] = 1
        
        # print to a window 'img'
        cv2.imshow('img', img_mat)
        cv2.waitKey(10);
        
        
class LOS:
    def __init__(self, map):
        self.pos2idx = map.pos2idx # based on the map (occ_grid) it return's the map index representation of the position pos
        # use self.pos2idx(pos, False) to return the exact index representation, including values that are less than 1.
        # use self.pos2idx(pos) to return the integer index representation, which is the rounded version of self.pos2idx(pos, False)
    def calculate(self, start_pos, end_pos):
        # sets up the LOS object to prepare return a list of indices on the map starting from start_pos (world coordinates) to end_pos (world)
        # start_pos is the robot position.
        # end_pos is the maximum range of the LIDAR, or an obstacle.
        # every index returned in the indices will be the index of a FREE cell
        # you can return the indices, or update the cells in here
        start_idx = float64(self.pos2idx(start_pos)) # <-- use start_idx. Don't use start_pos. 
        end_idx = float64(self.pos2idx(end_pos)) # <-- use end_idx. Don't use end_pos.
        indices = [] # init an empty list
        indices.append(int64(start_idx)) # append the starting index into the cell
        

        delX = end_idx[0] - start_idx[0]
        delY = end_idx[1] - start_idx[1]
        
        if abs(delX) > abs(delY):
            deltaL = delX
            deltaS = delY
            (l,s) = (start_idx[0], start_idx[1])
            (lf, sf) = (end_idx[0], end_idx[1])
            get_idx = lambda l,s : (l,s)
            
        else:
            deltaL = delY
            deltaS = delX
            (l,s) = (start_idx[1], start_idx[0])
            (lf, sf) = (end_idx[1], end_idx[0])
            get_idx = lambda l,s : (s,l)


        delS = sign(deltaS)
        delL = sign(deltaL)
        psi = deltaS/abs(deltaL)
        
        eta = 0
        
        while (l,s) != (lf,sf):
            # print((l,s), (lf,sf))
            l += delL
            eta += psi
            if abs(eta) >= 0.5:
                eta -= delS
                s += delS
                
            (i,j) = get_idx(l,s)
            indices.append((int64(i),int64(j))) 
         
        #del indices[0]   
        del indices[-1]     
        #indices.pop()
        
        return indices
        
class OdometryMM:
    def __init__(self, initial_pose, initial_wheels, axle_track, wheel_dia):
        self.x = initial_pose[0] # m, robot's x position in world
        self.y = initial_pose[1] # m, robot's y position in world
        self.o = initial_pose[2] # rad, robot's bearing in world
        self.wl = initial_wheels[0] # rad, robot's left angle
        self.wr = initial_wheels[1] # rad, robot's right angle
        self.L = axle_track # m, robot's axle track
        self.WR = wheel_dia/2.0 # m, robot's wheel RADIUS, not DIAMETER
        self.t = rospy.get_time() # s, time last calculated
    def calculate(self, wheels):
        # calculates the robot's new pose based on wheel encoder angles
        # INPUT: wheels: (left_wheel_angle, right_wheel_angle)
        # OUTPUT: a new pose (x, y, theta)
        
        # previous wheel angles stored in self.wl and self.wr, respectively. Remember to overwrite them
        # previous pose stored in self.x, self.y, self.o, respectively. Remember to overwrite them
        # previous time stored in self.t. Remember to overwrite it
        # axle track stored in self.L. Should not be overwritten.
        # wheel radius, NOT DIAMETER, stored in self.WR. Should not be overwritten.
        dt = rospy.get_time() - self.t # current time minus previous time
        dwl = wheels[0] - self.wl 
        dwr = wheels[1] - self.wr

        vt= 2*self.WR*(dwl + dwr)/(4*dt)
        deltaphi = 2*self.WR*(dwr - dwl)/(2*self.L) 
        w = self.WR*(dwr - dwl)/(self.L*dt) 

         
        if abs(w) < 1e-10:
        #   MM for move straight
            self.x = self.x + vt*dt*cos(self.o) 
            self.y = self.y + vt*dt*sin(self.o)
            self.o = self.o 
        else:
        #   MM for curve turns
            rt = self.L*(dwl + dwr)/(2*(dwr - dwl))
            self.x = self.x - rt*sin(self.o) + rt*sin(self.o + deltaphi)
            self.y = self.y + rt*cos(self.o) - rt*cos(self.o + deltaphi)
            self.o = self.o + deltaphi

        self.wl = wheels[0]
        self.wr = wheels[1]
        self.t = self.t + dt # update the current time. There's a reason why resampling the time is discouraged
        return (self.x, self.y, self.o)

class VelocityMM:
    def __init__(self, initial_pose, axle_track, wheel_dia):
        self.x = initial_pose[0] # m, robot's x position in world
        self.y = initial_pose[1] # m, robot's y position in world
        self.o = initial_pose[2] # rad, robot's bearing in world
        self.L = axle_track # m, robot's axle track
        self.WR = wheel_dia/2.0 # m, robot's wheel RADIUS, not DIAMETER
        self.t = rospy.get_time() # s, time last calculated
    def calculate(self, control):
        # calculates the robot's new pose based on wheel encoder angles
        # INPUT: control: (vt, wt)
        # OUTPUT: a new pose (x, y, theta)
        
        # previous pose stored in self.x, self.y, self.o, respectively. Remember to overwrite them
        # previous time stored in self.t. Remember to overwrite it
        # axle track stored in self.L. Should not be overwritten.
        # wheel radius, NOT DIAMETER, stored in self.WR. Should not be overwritten.
        dt = rospy.get_time() - self.t # current time minus previous time
        vt = control[0]
        wt = control[1]
        if abs(wt) < 1e-10:
        #   MM for move straight
            self.x = self.x + vt*dt*cos(self.o)
            self.y = self.y + vt*dt*sin(self.o)
            self.o = self.o
        else:
        #   MM for curve turns
            self.x = self.x - vt*sin(self.o)/wt + vt*sin(self.o + wt*dt)/wt 
            self.y = self.y + vt*cos(self.o)/wt - vt*cos(self.o + wt*dt)/wt
            self.o = self.o  + wt*dt
        self.t = self.t + dt # update the current time. There's a reason why resampling the time is discouraged
        return (self.x, self.y, self.o)
        
# =============================== SUBSCRIBERS =========================================  
def subscribe_true(msg):
    # subscribes to the robot's true position in the simulator. This should not be used, for checking only.
    global rbt_true
    msg_tf = msg.transforms[0].transform
    rbt_true = (\
        msg_tf.translation.x, \
        msg_tf.translation.y, \
        euler_from_quaternion([\
            msg_tf.rotation.x, \
            msg_tf.rotation.y, \
            msg_tf.rotation.z, \
            msg_tf.rotation.w, \
        ])[2]\
    )
    
def subscribe_scan(msg):
    # stores a 360 long tuple of LIDAR Range data into global variable rbt_scan. 
    # 0 deg facing forward. anticlockwise from top.
    global rbt_scan, write_scan, read_scan
    write_scan = True # acquire lock
    if read_scan: 
        write_scan = False # release lock
        return
    rbt_scan = msg.ranges
    write_scan = False # release lock

def subscribe_control(msg):
    global rbt_control
    rbt_control = (msg.linear.x, msg.angular.z)
    
def subscribe_wheels(msg):
    # returns the angles in which the wheels have been recorded to turn since the start
    global rbt_wheels
    right_wheel_angle = msg.position[0] # examine topic /joint_states #Edited
    left_wheel_angle = msg.position[1] # examine topic /joint_states #Edited
    rbt_wheels = (left_wheel_angle, right_wheel_angle)
    return rbt_wheels
    
def get_scan():
    # returns scan data after acquiring a lock on the scan data to make sure it is not overwritten by the subscribe_scan handler while using it.
    global write_scan, read_scan
    read_scan = True # lock
    while write_scan:
        pass
    scan = rbt_scan # create a copy of the tuple
    read_scan = False
    return scan
    
# ================================== PUBLISHERS ========================================

# =================================== TO DO ============================================
# Define the LIDAR maximum range
MAX_RNG = 3.5 #Edited after reading from /scan

# Define the inverse sensor model for mapping a range reading to world coordinates
def inverse_sensor_model(rng, deg, pose):
    # degree is the bearing in degrees # convert to radians
    # range is the current range data at degree
    # pose is the robot 3DOF pose, in tuple form, (x, y, o)
    x, y, o = pose
    xk = x + rng*cos(o+(deg*pi/180)) #Edited
    yk = y + rng*sin(o+(deg*pi/180)) #Edited
    

    #f.write("%.3f %.3f\n" %(xk, yk))
    
    
    return (xk, yk)

# ================================ BEGIN ===========================================
def main():
    # ---------------------------------- INITS ----------------------------------------------
    # init node
    rospy.init_node('main')
    
    # Set the labels below to refer to the global namespace (i.e., global variables)
    # global is required for writing to global variables. For reading, it is not necessary
    global rbt_scan, rbt_true, read_scan, write_scan, rbt_wheels, rbt_control
    
    # Initialise global vars with NaN values 
    # nan and inf are imported from numpy. If you use "import numpy as np", then nan is np.nan, and inf is np.inf.
    rbt_scan = [nan]*360 # a list of 360 nans
    rbt_true = [nan]*3
    read_scan = False
    write_scan = False
    rbt_control = None
    rbt_wheels = None

    # Subscribers
    rospy.Subscriber('scan', LaserScan, subscribe_scan, queue_size=1)
    rospy.Subscriber('tf', TFMessage, subscribe_true, queue_size=1)
    rospy.Subscriber('joint_states', JointState, subscribe_wheels, queue_size=1)
    rospy.Subscriber('cmd_vel', Twist, subscribe_control, queue_size=1)
    
    # Wait for Subscribers to receive data.
    while isnan(rbt_scan[0]) or isnan(rbt_true[0]) or rbt_wheels is None:
        pass
    
    # Data structures
    occ_grid = OccupancyGrid((-5,-5), (5,5), 0.1, 0.5) # OccupancyGrid((-5,-5), (5,5), ???, ???)
    los = LOS(occ_grid)
    motion_model = OdometryMM((0,0,0), (0,0), 0.16, 0.066) # VelocityMM((0,0,0), 0.16, 0.066)
    # ---------------------------------- BEGIN ----------------------------------------------
    t = rospy.get_time()
    while (not rospy.is_shutdown()): # required to Keyboard interrupt nicely
#        print('---')
#        print('True pos: ({:.3f} {:.3f} {:.3f})'.format(rbt_true[0], rbt_true[1], rbt_true[2]))
#        continue
        
        if (rospy.get_time() > t): # every 50 ms
            
            # get scan
            scan = get_scan()
            # calculate the robot position using the motion model
            rbt_pos = motion_model.calculate(rbt_wheels) 
            #rbt_true  motion_model.calculate(???) # rbt_wheels, rbt_control #Edited
            
            #f = open("src/pkg/scripts/inverse_sensor_model.txt", "w")
            
            def bayes_calc (v_new, p_prev):
                return (1-1/(1+numpy.exp(v_new + numpy.log(p_prev/(1-p_prev)))))
            
            # for each degree in the scan
            for i in xrange(360):
                # if you use log-odds Binary Bayes
                if scan[i] != inf: # range reading is < max range ==> occupied
                    end_pos = inverse_sensor_model(scan[i], i, rbt_pos)
                    # # set the obstacle cell as occupied
                    occ_grid.set_cell_pos(end_pos, bayes_calc(1, occ_grid.get_cell_pos(end_pos)))
                else: # range reading is inf ==> no obstacle found
                    end_pos = inverse_sensor_model(MAX_RNG, i, rbt_pos)
                    # # set the last cell as free
                    occ_grid.set_cell_pos(end_pos, bayes_calc(-0.5, occ_grid.get_cell_pos(end_pos)))
                # # set all cells between current cell and last cell as free
                for idx in los.calculate(rbt_pos, end_pos):
                    occ_grid.set_cell_idx(idx, bayes_calc(-0.5, occ_grid.get_cell_idx(idx))) #??? # occ_grid.set_cell_idx(idx, ???)
                    
                # if you don't use log-odds Binary Bayes
                #if scan[i] != inf: # range reading is < max range ==> occupied
                #    end_pos = inverse_sensor_model(scan[i], i, rbt_pos, f)
                #else: # range reading is inf ==> no obstacle found
                #    end_pos = inverse_sensor_model(MAX_RNG, i, rbt_pos, f)
                # update the in between areas
                #for idx in los.calculate(rbt_pos, end_pos):
                #    occ_grid.set_cell_idx(idx, 0)
                # update the boundaries
                #if scan[i] != inf: # range reading is < max range ==> occupied
                    # set the obstacle cell as occupied
                #    occ_grid.set_cell_pos(end_pos, 1)
                #else: # range reading is inf ==> no obstacle found
                    # set the last cell as free
                #    occ_grid.set_cell_pos(end_pos, 0)
                
            
            #f.close()
            
            
            # show the map as a picture
            occ_grid.show_map(rbt_pos)
            
            # increment the time counter
            t += 0.05
        
        
if __name__ == '__main__':      
    try: 
        main()
    except rospy.ROSInterruptException:
        pass
