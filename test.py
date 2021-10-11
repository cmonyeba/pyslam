import numpy as np
import cv2
from numpy import ma, uint8
from matplotlib import pyplot as plt
from numpy.core.fromnumeric import size

W = 1920//2
L = 1080//2

cap = cv2.VideoCapture('test.mp4')


#takes in a frame and returns details regarding the frame
def extract_features(img):
    #ORB is a fusion of FAST keypoint detector and BRIEF descriptor
    #ORB is a good choice in low-powerdevices for panorama stitching etc.

    #create an instance of ORB(Oriented FAST and Rotated BRIEF)
    orb = cv2.ORB_create()
    
    #find all the features in the frame using (Shi-Tomasi Corner Detector)
    features = cv2.goodFeaturesToTrack(img, maxCorners=5000, qualityLevel=0.01, minDistance=5)
    #change type for features to usigned int 16
    features = features.astype(np.uint16)
    
    #now from the features determine the key points
    #size is keypoint diameter
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in features]
    
    #compute descriptors from keypoints (and possibly update keypoints)
    kps, des = orb.compute(img, kps)
    
    #return arrays for keypoints and associated descriptors
    return kps, des
    #np.array([(kp.pt[0], kp.pt[1]) for kp in kps])

# def match_frames(frame1, frame2):
#         bf = cv2.BFMatcher()
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (W,L))
    # frame = cv2.resize(frame, (W,L))
    
    kps, des = extract_features(frame)
   
    frame = cv2.drawKeypoints(frame, kps , outImage=np.array([]), color=(0,255,0))
        # print(x,y)
    cv2.imshow('frame',frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()