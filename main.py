import numpy as np
import cv2
from numpy import ma, uint8
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size

def extract_features(img):
    #ORB is a fusion of FAST keypoint detector and BRIEF descriptor
    #ORB is a good choice in low-powerdevices for panorama stitching etc.

    #create an instance of ORB(Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create()
    
    #find all the features in the frame using (Shi-Tomasi Corner Detector)
    features = cv2.goodFeaturesToTrack(img, maxCorners=5000, qualityLevel=0.01, minDistance=5)
    features = features.astype(np.uint16)
    
    # print(features)
    #now from the features determine the key points
    #size is key
    # point diameter
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
   
    #compute descriptors from keypoints (and possibly update keypoints)
    kps, des = orb.compute(img, kps)
    
    #return arrays for keypoints and associated descriptors
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]).astype(np.uint16), des
    #np.array([(kp.pt[0], kp.pt[1]) for kp in kps]).astype(np.uint16)

def track_features(old_des, des):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
    matches = bf.match(old_des, des)
    old_idx = []
    new_idx = []
    for m in matches:    
        # print(m.trainIdx, m.queryIdx, m.distance)
        old_idx.append(m.trainIdx)
        new_idx.append(m.queryIdx)
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good.append([m])
        
    # matches = sorted(matches, key = lambda x:x.distance)

    return old_idx, new_idx

W = 1920//2
L = 1080//2

cap = cv2.VideoCapture('test.mp4')

# Capture first frame
ret, old_frame = cap.read()

# Change frame to gray-scale for feature detection + resize frame
gray_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
gray_frame = cv2.resize(gray_frame, (W,L))
old_frame = cv2.resize(old_frame, (W,L))
# Extract key-points and descriptors from frame    
old_kps, old_des = extract_features(gray_frame)


while(True):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Change frame to gray-scale for feature detection + resize frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, (W,L))
    frame = cv2.resize(frame, (W,L))
    
    # Extract key-points and descriptors from frame    
    kps, des = extract_features(gray_frame)
    old_idx, new_idx = track_features(old_des, des)
    # print(old_idx, new_idx)
    # print(kps.shape, old_kps.shape, old_idx )
    
    # pt_kps = [cv2.KeyPoint(x=i[0], y=i[1], size=20) for i in kps]
    # pt_old_kps = [cv2.KeyPoint(x=i[0], y=i[1], size=20) for i in old_kps]
    # frame = cv2.drawKeypoints(frame, kps , outImage=np.array([]), color=(0,255,0))
    for old_kp, kp in zip(old_kps, kps):
        frame = cv2.circle(frame, (kp[0], kp[1]), radius=3, color=(0,255,0))
    for i, j in zip(old_idx, new_idx):
        # print(i,j)
        frame = cv2.line(frame, (old_kps[j][0], old_kps[j][1]), (kps[i][0], kps[i][1]), color = (255,0,0))
    # frame = cv2.line(frame, (old_kp[0], old_kp[1]), (kp[0],kp[1]), color=(0,0,255))
        # img = cv2.drawMatches(old_frame, pt_old_kps, frame, pt_kps, matches[:100],outImg=np.array([]), flags=2)
    # print(matches)
    # cv2.imshow("matches", img)
    cv2.imshow('frame',frame)

    old_frame = frame
    old_kps = kps
    old_des = des

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()