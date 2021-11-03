from function import poseRt, calc_pose_matrices
import cv2
import numpy as np
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform
from skimage.measure import ransac

#Takes in an image array and returns kps and des.
def extractFeatures(frame):
        #ORB (Oriented FAST and Rotated BRIEF) is a fusion of FAST keypoint detector and BRIEF descriptor
        #Create an instance
       
        
        orb = cv2.ORB_create()
        #Find Features (Shi-Tomasi Corner Detector)
        features = cv2.goodFeaturesToTrack(frame, maxCorners=1000, qualityLevel=0.01, minDistance=7)
        #Convert to KeyPoint datatype to put into orb.compute
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
        #Compute descriptors (and possibly update keypoints)
        kps, des = orb.compute(frame, kps)
        #Return deconstructed kps and des
        return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

#Matches two frames and returns matching corresponding points
def matchFrames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    ret = []
    idx1 = []
    idx2 = []

    for m, n in matches:
        if m.distance < 0.75*n.distance:
            pt1 = f1.nkps[m.queryIdx]
            pt2 = f2.nkps[m.trainIdx]

            if np.linalg.norm((pt1-pt2)) < 0.1*np.linalg.norm([f1.w, f1.h]) \
            and m.distance < 32:
                if m.queryIdx not in idx1 and m.trainIdx not in idx2:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    ret.append((pt1, pt2))

    # avoid duplicates
    assert(len(set(idx1)) == len(idx1))
    assert(len(set(idx2)) == len(idx2))
    assert len(ret) >= 8

    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    model, inliers = ransac(
                        (ret[:, 0], ret[:, 1]),
                        FundamentalMatrixTransform,
                        min_samples = 8, residual_threshold  = 0.001,
                        max_trials = 100
                    )

    # Hartley & Zissserman approach.
    Rt = calc_pose_matrices(model)

    return idx1[inliers], idx2[inliers], Rt


class KeyFrame():
    def __init__(self, frame, map, K):

        #Intrinsic Matrices
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        #Extracting unsorted KPS and DES from frame
        self.kps, self.des = extractFeatures(frame)
        self.nkps = self.normalize(self.Kinv, self.kps)
        self.pts = [None]*len(self.nkps)

        #Homogenous 4x4 matrix for Camera Pose at each frame
        self.pose = np.eye(4)
        #Frame shape
        self.h, self.w = frame.shape[0:2]

        #Adding frame to Map and giving frame an ID
        self.id = map.addFrame(self)

    def normalize(self, Kinv, point):
        return np.dot(Kinv, self.add_ones(point).T).T[:, 0:2]

    def add_ones(self, x):
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis = 1)