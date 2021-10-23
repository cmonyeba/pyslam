import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pykitti
from frame import KeyFrame, matchFrames
from map import Map

# Change this to the directory where you store KITTI data
basedir = '/Users/sebaw/Documents/kitti/raw'

# Specify the dataset to load
date = '2011_09_26'
drive = '0035'

# Load the data. Optionally, specify the frame range to load.
dataset = pykitti.raw(basedir, date, drive)

class Slam:
    def __init__(self, W, H, K):
        self.cam_pos = [0,0,0]
        self.cam_xyz = []
        self.kps_xyz = []
        self.map = Map()
        self.scale = 5
        self.W, self.H = W, H
        #Intrinsic Matrix
        self.K = K
        self.alt = 0
        self.count = 0
        
    def processFrame(self, frame):
        
        #Verify that the frame shape is correct and create KeyFrame object
        assert frame.shape[:2] == (self.H, self.W)
        frame =  KeyFrame(frame, self.map, self.K)
        
        #Makes sure it is not the first frame
        #The most recent frame and previous frame are set as f1 and f2
        if frame.id == 0:
            return  
        f1 = self.map.frames[-1]
        f2 = self.map.frames[-2]
        
        #Calculate Matches and Extrinsic Matrix (R|t relation between the two frames)
        idx1, idx2, Extrinsic = matchFrames(f1, f2, self.K)
        
        f1.pts = [f1.kps[idx] for idx in idx1]
        f2.pts = [f2.kps[idx] for idx in idx2]
       
      
        
        f1.pose = np.dot(Extrinsic, f2.pose)
    
        P1 = np.dot(f1.K, np.array(f1.pose)[:3])
        P2 = np.dot(f2.K, np.array(f2.pose)[:3])
        
        #u = PX
        A = np.eye(4)
        
        pts1 = self.cvtPoint(np.array(f1.pts))
        pts2 = self.cvtPoint(np.array(f2.pts))
        Point = np.zeros((pts1.shape[0], 4))

        for i in range(len(pts1)):
            
            temp1 = np.cross(pts1[i], P1.T)
            temp2 = np.cross(pts2[i], P2.T)
            te1 = temp1.T
            te2 = temp2.T
            A[:2, :4] = te1[:2]
            A[2:4, :4] = te2[:2]
            _, _, V = np.linalg.svd(A)
            Point[i] = V[3]
        Point = Point / Point[:, 3:]
        
        # pts4d = cv2.triangulatePoints(P1, P2, np.array(f1.pts), np.array(f2.pts))
        # print(pts4d)
        for pt in Point:
            print(pt)
            self.kps_xyz.append(pt)
    def cvtPoint(self, pts):
        ret = []
        for pt in pts:
            temp = np.zeros(3)
            temp[:2] = pt
            temp[2] = 1
            ret.append(temp)
        return np.array(ret)
    
 