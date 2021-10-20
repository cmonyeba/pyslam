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
                if len(pair) == 2:
                    if pair[0].distance < 0.9*pair[1].distance:
                        # print(self.old_kps[pair[0].queryIdx], kps[pair[0].trainIdx])
                        pts1.append(self.old_kps[pair[0].queryIdx])
                        pts2.append(kps[pair[0].trainIdx])
            
        self.old_kps = kps
        self.old_des = des
       
        
        return pts1, pts2        
       
    def extract_features(self, frame):
        
        #ORB is a fusion of FAST keypoint detector and BRIEF descriptor
        #ORB is a good choice in low-powerdevices for panorama stitching etc.

        #create an instance of ORB(Oriented FAST and Rotated BRIEF)
        orb = cv2.ORB_create()
        
        #find all the features in the frame using (Shi-Tomasi Corner Detector)
        features = cv2.goodFeaturesToTrack(frame, maxCorners=250, qualityLevel=0.01, minDistance=5)
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
        x = 2563/2
        y = 1440/2
        
        scale = 15
        # focal lengths (assumes that the field of view is 60)
        fov = 60 * (math.pi / 180)
        f_x = x / math.tan(fov / 2)
        f_y = y / math.tan(fov / 2)
        
        # camera matrix
        K = np.array([[f_x, 0, x],
                        [0, f_y, y],
                        [0, 0, 1]])
        
        #fundamental matrix between the two points
        F, mask = cv2.findFundamentalMat(np.asarray(pts1), np.asarray(pts2), cv2.FM_8POINT)
        # print(F)
        
        #intrinsic and extrinsic 
        #ethzurich
        #maplab, pangolin
        #kalman filter particle filter gps real ground truth
        
        #this is just getting the extrinsic matrix (returns the rotation matrix and translation vector)
        points, R, t, mask = cv2.recoverPose(F, np.asmatrix(pts1), np.asmatrix(pts2), K)
        # print(R)
        R = np.asmatrix(R).I
        
        self.cam_xyz.append([self.cam_position[0] + t[0], self.cam_position[1] + t[1], self.cam_position[2] + t[2]])
        
        #the extrinsic matrix
        C = np.hstack((R,t))
        
        for i in range(len(pts2)):
            pts2d = np.asmatrix([pts2[i][0], pts2[i][1], 1]).T
            #CAMERA MATRIX
            P = np.asmatrix(K) * np.asmatrix(C)
            #camera matrix with 2d points find 3d points
            pts3d = np.asmatrix(P).I * pts2d  
            # print(pts3d)
            self.kps_xyz.append([pts3d[0][0] * scale + self.cam_position[0],
                        pts3d[1][0] * scale + self.cam_position[1],
                        pts3d[2][0] * scale + self.cam_position[2]])
            
        self.cam_position = [self.cam_position[0] + t[0], self.cam_position[1] + t[1], self.cam_position[2] + t[2]]