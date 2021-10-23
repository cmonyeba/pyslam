import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pykitti
from function import triangulate, cvtPoint
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
        self.cam_position = [0,0,0]
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
        
        
        pts1 = cvtPoint(np.array(f1.pts))
        pts2 = cvtPoint(np.array(f2.pts))
       

        pts4d = triangulate(P1, P2, pts1, pts2)
        pts4d = pts4d / pts4d[:, 3:]
        
        for pt in pts4d:
            # print(pt)
            # self.kps_xyz.append(pt)
            self.kps_xyz.append([pt[0] * self.scale + self.cam_position[0],
                        pt[1] * self.scale + self.cam_position[1],
                        pt[2] * self.scale + self.cam_position[2]])

        # print(self.kps_xyz)
        self.cam_xyz.append([self.cam_position[0] + f1.pose[:3, 3][0], self.cam_position[1] + f1.pose[:3, 3][1], self.cam_position[2] + f1.pose[:3, 3][2]])

        

        self.cam_position = [self.cam_position[0] + f1.pose[:3, 3][0], self.cam_position[1] + f1.pose[:3, 3][1], self.cam_position[2] + f1.pose[:3, 3][2]]
