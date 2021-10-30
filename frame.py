from os import lseek
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
        features = cv2.goodFeaturesToTrack(frame, maxCorners=250, qualityLevel=0.01, minDistance=5)
        #Convert to KeyPoint datatype to put into orb.compute
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
        #Compute descriptors (and possibly update keypoints)
        kps, des = orb.compute(frame, kps)
        #Return deconstructed kps and des
        return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

#Matches two frames and returns matching corresponding points
def matchFrames(f1, f2, K):

        #FLANN Parameters
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
        search_params = dict(checks=25)

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        ret = []
        idx1 = []
        idx2 = []
        idxRef1, idxRef2 = set(), set()

        matches = flann.knnMatch(f1.des, f2.des, 2)
        # ratio test as per Lowe's paper
        for pair in matches:
            try:
                m, n = pair
                if m.distance < 0.75*n.distance:
                    if m.distance < 20:
                        pt1 = f1.kps[m.queryIdx]
                        pt2 = f2.kps[m.trainIdx]
                        if m.queryIdx not in idxRef1 and m.trainIdx not in idxRef2:
                            idx1.append(m.queryIdx)
                            idx2.append(m.trainIdx)
                            idxRef1.add(m.queryIdx)
                            idxRef2.add(m.trainIdx)
                            ret.append((pt1, pt2))
            except ValueError:
                pass


        assert len(ret) > 8
        ret = np.array(ret)
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        # print(np.transpose(np.array(f1.K)))
        # print(np.array(f2.K))
        
        
        #Fundamental matrix using skimage library because cv2 sucks
        # F, mask  = cv2.findFundamentalMat(ret[: ,0], ret[: ,1], method=cv2.RANSAC, ransacReprojThreshold=0.001, confidence=0.99)

        model, inliers = ransac((ret[:, 0], ret[:, 1]),
                         EssentialMatrixTransform, min_samples=8,
                         residual_threshold=1, max_trials=1000)
        

        #Fundamental to Essential Matirx 
        #The Essential Matrix is essentially a calibrated Fundamental Matrix
        #E = (K')^T * F * K
        # E = np.transpose(np.matrix(f1.K)) * np.matrix(F) * np.matrix(f2.K)

        # ret, R, t, mask = cv2.recoverPose(E, ret[: ,0], ret[: ,1], f1.K)
        # R = np.asmatrix(R).I
        Rt = calc_pose_matrices(model)

        return idx1[inliers], idx2[inliers], Rt
    
def poseRt(R, t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret

def calc_pose_matrices(model):
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float)
    U, w, V = np.linalg.svd(model.params)

    if np.linalg.det(U) < 0:
        U *= -1.0

    if np.linalg.det(V) < 0:
        V *= -1.0

    R = np.dot(np.dot(U, W), V)

    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), V)
    t = U[:, 2]
    return poseRt(R, t)

class KeyFrame():
    def __init__(self, frame, map, K):
        self.K = K
        
        self.kps, self.des = extractFeatures(frame)
        self.pts = [None]*len(self.kps)
        self.id = map.addFrame(self)
        self.idx = [None]
        #Homogenous 4x4 matrix
        self.pose = np.eye(4)


