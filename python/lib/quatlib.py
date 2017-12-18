"""
Quaternion handling library

@author: lukas huber
@date: 2017-11-03
"""
# import libraries
from math import asin, atan2, sin, cos, pi, copysign

import numpy as np
from numpy import linalg as LA

## Quaternion Funtions
def quatinv(q):
    #inverse quaternion 
    # return inv_q =
    return ( np.array([q[0], -q[1], -q[2], -q[3]]) / LA.norm(q) )


def quatmul(q1,q2):
    #quaternion multiplication
    s1 = q1[0]
    v1 = np.array(q1[1:4])
    
    s2 = q2[0]
    v2 = np.array(q2[1:4])

    # return q = ...
    a = s1*s2 - np.dot(v1,v2)
    b = np.cross(v1,v2)

    return np.hstack(([s1*s2 - np.dot(v1,v2)],np.cross(v1,v2) + s1*v2 + s2*v1))

def quatrot_inv(r, q):
    r_ = quatmul(quatinv(q), np.hstack(([0],r)) )
    r_ = quatmul(r_,q)
    
    return r_[1:4]


def quatrot(r, q):
    # rotation with quaternions
    if(len(r)==3): # 3d -- rotation of vector
        r_ = quatmul(q, np.hstack(([0],r)) )
        r_ = quatmul(r_, quatinv(q) )
        return r_[1:4]
    
    else: # 4d --  rotation of quaternion
        r_ = quatmul(q, r)
        r_ = quatmul(r_, quatinv(q))
        
        return r_

    
def quat2eul(q):
    # roll (x-axis rotation)
    sinr = +2.0 * (q[0]*q[1] + q[2]*q[3])
    cosr = +q[0]*q[0] + q[1]*q[1] - q[2]*q[2]+q[3]*q[3]
    roll = atan2(sinr, cosr)

    ## pitch (y-axis rotation)
    sinp = +2.0 * (q[0] * q[2] - q[3] * q[1])
    #sinp = +1.0 if sinp > +1.0 else sinp
    #sinp = -1.0 if sinp < -1.0 else sinp
    if (abs(sinp) >= 1):
        pitch = pi/2 # use 90 degrees if out of range
    else:
        pitch = asin(sinp)

    # yaw (z-axis rotation)
    siny = +2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy = +q[0]*q[0] + q[1]*q[1] - q[2]*q[2] - q[3]*q[3]
    yaw = atan2(siny, cosy)

    return [roll, pitch, yaw]


def eul2quat(eulAngles):
    # case ZYX
    pitch, roll, yaw  = eulAngles

    # Abbreviations for the various angular functions
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)

    # Calulate quaternion
    q = [cy * cr * cp + sy * sr * sp,
         cy * cr * sp - sy * sr * cp,
         cy * sr * cp + sy * cr * sp,
         sy * cr * cp - cy * sr * sp]
    
    #return np.array(q)
    return q
