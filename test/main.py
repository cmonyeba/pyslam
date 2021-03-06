import numpy as np
import cv2
import slam
import math
import Visual
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


L = 1440//2
W = 2562//2

#capture video in cap
cap = cv2.VideoCapture('images/test2.mp4')

#create an SLAM instance
W = 2562
H = 1440
F = 500
K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
slam = slam.Slam(W, H, K)

# slam.runSlam(old_frame)

# print(slam.kpsxyz)

while(cap.isOpened()):
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    # print(frame.shape)
    if ret:
        #gray-scale frame and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # print(frame.shape)
        # print(frame)
        Visual.extract_features(frame)
        
        slam.processFrame(frame)
    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('no frame')
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

cam_xyz = np.array(slam.cam_xyz)
kps_xyz = np.array(slam.kps_xyz)
fig = plt.figure()
ax = Axes3D(fig)
# ax.view_init(0, 150)
ax.scatter(kps_xyz[:, [0]], kps_xyz[:, [1]], kps_xyz[:, [2]], s=1)
ax.scatter(cam_xyz[:, [0]], cam_xyz[:, [1]], cam_xyz[:, [2]], c='r', s=1)
plt.show() 