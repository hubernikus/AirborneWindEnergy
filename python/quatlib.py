#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 23:20:36 2017

@author: lukas
"""
# import libraries
import numpy as np
from numpy import linalg as LA

def quatinv(q):
    #inverse quaternion 
    
    # return inv_q =
    return ( np.array([(1) -q(2) -q(3) -q(4)]) / LA.norm(q) )




def quatmul(q1,q2):
    #quaternion multiplication
    s1 = q1[0]
    v1 = q1[1:4]
    
    s2 = q2[0]
    v2 = q2[1:4]
    
    # return q = ...
    return np.array([s1*s2 - np.dot(v1,v2), np.cross(v1,v2) + s1*v2 + s2*v1])