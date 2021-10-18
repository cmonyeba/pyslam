import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Slam:
    def __init__(self) -> None:
        self.old_kps = []
        self.old_des = []
        self.cam_position = [0,0,0]
        self.cam_xyz = []
        self.kps_xyz = []
        
    def runSlam(self, frame):
        pts1, pts2 = self.match_features(frame)
        if pts1 and pts2:
            self.calculate_coords(pts1, pts2)
            

    def match_features(self, frame):
        #FLANN Features
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
        search_params = dict(checks=50)   
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        
        kps, des = self.extract_features(frame)
        pts1 = []
        pts2 = []
        # print(self.old_kps, kps)
        if len(self.old_kps) != 0:
            # flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(self.old_des, des, 2)
            
            # print(matches)

            # Need to draw only good matches, so create a mask
            # ratio test as per Lowe's paper
            for pair in matches:
                print(len(pair))
                if len(pair) != 0 and pair[0].distance < 0.7*pair[1].distance:
                    print('hi')
                    pass
            
        self.old_kps = kps
        self.old_des = des
        
        return pts1, pts2        
       
    def extract_features(self, frame):
        
        #ORB is a fusion of FAST keypoint detector and BRIEF descriptor
        #ORB is a good choice in low-powerdevices for panorama stitching etc.

        #create an instance of ORB(Oriented FAST and Rotated BRIEF)
        orb = cv2.ORB_create()
        
        #find all the features in the frame using (Shi-Tomasi Corner Detector)
        features = cv2.goodFeaturesToTrack(frame, maxCorners=50, qualityLevel=0.01, minDistance=5)
        features = features.astype(np.uint16)
        
        # print(features)
        #now from the features determine the key points
        #size is key
        # point diameter
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
    
        #compute descriptors from keypoints (and possibly update keypoints)
        kps, des = orb.compute(frame, kps)
        
        #return arrays for keypoints and associated descriptors
        return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]).astype(np.uint16), des
        #np.array([(kp.pt[0], kp.pt[1]) for kp in kps]).astype(np.uint16)
        
    def calculate_coords(self, pts1, pts2):
        x = 1920//2
        y = 1080//2
        
        scale = 5
        # focal lengths (assumes that the field of view is 60)
        fov = 60 * (math.pi / 180)
        f_x = x / math.tan(fov / 2)
        f_y = y / math.tan(fov / 2)
        
        # camera matrix
        K = np.array([[f_x, 0, x],
                        [0, f_y, y],
                        [0, 0, 1]])
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)
        points, R, t, mask = cv2.recoverPose(F, np.asmatrix(pts1), np.asmatrix(pts2), K, 500)
        R = np.asmatrix(R).I
        
        self.cam_xyz.append([self.cam_position[0] + t[0], self.cam_position[1] + t[1], self.cam_position[2] + t[2]])
        
        C = np.hstack((R,t))
        
        for i in range(len(self.old_kps)):
            pts2d = np.asmatrix([self.old_kps[i][0], self.old_kps[0][i], 1]).T
            P = np.asmatrix(K) * np.asmatrix(C)
            pts3d = np.asmatrix(P).I * pts2d
            self.kps_xyz.append[pts3d[0][0] * scale + self.cam_position[0],
                        pts3d[1][1] * scale + self.cam_position[1],
                        pts3d[2][0] * scale + self.cam_position[2]]
            
        self.cam_position = [self.cam_position[0] + t[0], self.cam_position[1] + t[1], self.cam_position[2] + t[2]]