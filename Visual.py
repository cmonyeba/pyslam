import cv2
import numpy as np

old_kps = []
old_des = []
def extract_features(img):
    #ORB is a fusion of FAST keypoint detector and BRIEF descriptor
    #ORB is a good choice in low-powerdevices for panorama stitching etc.

    #create an instance of ORB(Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create()
    
    #find all the features in the frame using (Shi-Tomasi Corner Detector)
    features = cv2.goodFeaturesToTrack(img, maxCorners=250, qualityLevel=0.01, minDistance=5)
    features = features.astype(np.uint16)
    
    # print(features)
    #now from the features determine the key points
    #size is key
    # point diameter
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
   
    #compute descriptors from keypoints (and possibly update keypoints)
    kps, des = orb.compute(img, kps)
    draw = cv2.drawKeypoints(img, kps, np.array([]), (0,255,0))
    #return arrays for keypoints and associated descriptors
    cv2.imshow('frame', draw)
    # return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]).astype(np.uint16), des
    #np.array([(kp.pt[0], kp.pt[1]) for kp in kps]).astype(np.uint16)
    
def track_features(old_des, des):
    #FLANN Features
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    search_params = dict(checks=50)   
    
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(old_des, des)


    old_idx = []
    new_idx = []
    
    for m in matches:  
        # print(m, n)  
        # print(m.trainIdx, m.queryIdx)
        # print(n.trainIdx, n.queryIdx)
        # print(m.distance , n.distance)
        if m.distance < 25:
            # print(m.distance)
            old_idx.append(m.trainIdx)
            new_idx.append(m.queryIdx)
            
    # matches = sorted(matches, key = lambda x:x.distance)

    return old_idx, new_idx