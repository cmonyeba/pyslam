from os import lseek
import cv2
import numpy as np

#Takes in an image array and returns kps and des.
def extractFeatures(frame):
        #ORB (Oriented FAST and Rotated BRIEF) is a fusion of FAST keypoint detector and BRIEF descriptor
        #Create an instance
        orb = cv2.ORB_create()
        #Find Features (Shi-Tomasi Corner Detector)
        features = cv2.goodFeaturesToTrack(frame, maxCorners=250, qualityLevel=0.01, minDistance=5)
        #Convert to KeyPoint datatype to put into orb.compute
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
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
        F, mask  = cv2.findFundamentalMat(ret[: ,0], ret[: ,0], method=cv2.RANSAC, ransacReprojThreshold=0.02, confidence=0.3, maxIters=100)
        #E = (K')^T * F * K
        E = np.transpose(np.array(f1.K) * F * np.array(f2.K))
      
        ret, R, t, mask = cv2.recoverPose(E, ret[: ,0], ret[: ,1], f1.K)
   
        hRt = np.eye(4)
        hRt[:3, :3] = R
        hRt[:3, 3] = np.transpose(t)
        # print(hRt)
        return idx1, idx2, hRt
    
class KeyFrame():
    def __init__(self, frame, map, K):
        self.K = K
        self.kps, self.des = extractFeatures(frame)
        self.id = map.addFrame(self)
        #Homogenous 4x4 matrix
        self.pose = np.eye(4)
        self.pts = None
        
        
    