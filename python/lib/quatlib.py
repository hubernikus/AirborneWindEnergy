# coding: utf-8

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
    return ( np.array([q[0], -q[1], -q[2], -q[3]]) / LA.norm(q) )


def quatmul(q1,q2):
    #quaternion multiplication
    s1 = q1[0]
    v1 = np.array(q1[1:4])
    
    s2 = q2[0]
    v2 = np.array(q2[1:4])

    return np.hstack(([s1*s2 - np.dot(v1,v2)],np.cross(v1,v2) + s1*v2 + s2*v1))


def quatrot_inv(r, q):
    r_ = quatmul(quatinv(q), np.hstack(([0],r)) )
    r_ = quatmul(r_,q)
    
    return r_[1:4]

def quatrot(r, q):
    # rotation with quaternions
    #if(len(r)==3): # 3d -- rotation of vector
        r_ = quatmul(q, np.hstack(([0],r)) )
        r_ = quatmul(r_, quatinv(q) )
        return r_[1:4]
    
#    else: # 4d --  rotation of quaternion ??? is this sensible?
#        r_ = quatmul(q, r)
#        r_ = quatmul(r_, quatinv(q))
#        return r_


def quat2eul_rotationRate(q, omega):
    v = np.array([q[1:3]])
    
    S_e = np.array([[0, -q[3], q[2]],
                    [q[3], 0, -q[1]],
                    [-q[2], q[1], q[0]]])
    
    Phi = np.hstack(( S_e+q[0]*eye(3), -q.T ))

    return 0.5*np.dot(Phi,omega)

def eul2quat_rotationrate(omega):
    # TODO
    return 0
    
#    
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

def slerp(p0, p1, t):
# spherical linear interpolation
    p0 = p0/norm(p0)
    p1 = p1/norm(p1)

    dotProd = np.dot(p0,p1)

    # If the dot product is negative, the quaternions
    # have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if dotProd < 0:
        p1 = -p1
        dotProd = -dotProd
        
    omega = arccos(dotProd)
    so = sin(omega)

    return sin((1.0-t)*omega) / so * p0 + sin(t*omega)/so * p1

# Quaternion slerp(Quaternion v0, Quaternion v1, double t) {
#     // Only unit quaternions are valid rotations.
#     // Normalize to avoid undefined behavior.
#     v0.normalize();
#     v1.normalize();

#     // Compute the cosine of the angle between the two vectors.
#     double dot = dot_product(v0, v1);

#     const double DOT_THRESHOLD = 0.9995;
#     if (fabs(dot) > DOT_THRESHOLD) {
#         // If the inputs are too close for comfort, linearly interpolate
#         // and normalize the result.

#         Quaternion result = v0 + t*(v1 – v0);
#         result.normalize();
#         return result;
#     }

#     // If the dot product is negative, the quaternions
#     // have opposite handed-ness and slerp won't take
#     // the shorter path. Fix by reversing one quaternion.
#     if (dot < 0.0f) {
#         v1 = -v1;
#         dot = -dot;
#     }  

#     Clamp(dot, -1, 1);           // Robustness: Stay within domain of acos()
#     double theta_0 = acos(dot);  // theta_0 = angle between input vectors
#     double theta = theta_0*t;    // theta = angle between v0 and result 

#     Quaternion v2 = v1 – v0*dot;
#     v2.normalize();              // { v0, v2 } is now an orthonormal basis

#     return v0*cos(theta) + v2*sin(theta);
# }
