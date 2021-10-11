import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = 'com.jpg'
filename2 = 'comcom.jpg'
size = (4032//6, 3024//6)
img0 = cv2.imread(filename2)
img = cv2.imread(filename)
# print(img.shape)
img = cv2.resize(img, size) 
img2 = cv2.resize(img0, size)


# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img0,None)
kp2, des2 = orb.detectAndCompute(img,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img0,kp1,img,kp2,matches[0:10], outImg=np.array([]), flags=2)
plt.imshow(img3), plt.show()
# kp = orb.detect(img)
# #result is dilated for marking the corners, not important
# kp, des = orb.compute(img, kp)
# print(kp, des)


# img2 = cv2.drawKeypoints(img,kp, outImage=np.array([]), color=(0,255,0), flags=0)
# plt.imshow(img2),plt.show()

# Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[0,0,255]
# cv2.imshow('dst',img)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
