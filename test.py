"""Example of pykitti.raw usage."""
import itertools
import matplotlib.pyplot as plt
import Visual
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pykitti


# Change this to the directory where you store KITTI data
basedir = '/Users/sebaw/Documents/kitti/raw'

# Specify the dataset to load
date = '2011_09_26'
drive = '0035'

# Load the data. Optionally, specify the frame range to load.
dataset = pykitti.raw(basedir, date, drive)
                    #   frames=range(0, 120, 1))
# print(dataset.calib.P_rect_00)
print(dataset.calib.K_cam0)
for frame in dataset.cam0:
    print(type(frame))
   
    test = cv2.imread(frame)
    print(test)
    orb = cv2.ORB_create()
    
    kps, des = orb.detectAndCompute(frame)
    
    frame = cv2.drawKeypoints(np.array(frame), kps, np.array([]), (0,255,0))
    
    cv2.imshow('shpow', frame)
    cv2.waitKey(100)
    
# dataset.calib:         Calibration data are accessible as a named tuple
# dataset.timestamps:    Timestamps are parsed into a list of datetime objects
# dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
# dataset.camN:          Returns a generator that loads individual images from camera N
# dataset.get_camN(idx): Returns the image from camera N at idx
# dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
# dataset.get_gray(idx): Returns the monochrome stereo pair at idx
# dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
# dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
# dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
# dataset.get_velo(idx): Returns the velodyne scan at idx

# Grab some data
# first_gray = dataset.get_gray(0)
# print(np.array(first_gray[0]).shape)

    
    
    # cv2.waitKey(1000)
# first_rgb = dataset.get_rgb(0)

# # Do some stereo processing
# stereo = cv2.StereoBM_create()
# disp_gray = stereo.compute(np.array(first_gray[0]), np.array(first_gray[1]))
# disp_rgb = stereo.compute(
#     cv2.cvtColor(np.array(first_rgb[0]), cv2.COLOR_RGB2GRAY),
#     cv2.cvtColor(np.array(first_rgb[1]), cv2.COLOR_RGB2GRAY))

# # Display some data
# f, ax = plt.subplots(2, 2, figsize=(15, 5))
# ax[0, 0].imshow(first_gray[0], cmap='gray')
# ax[0, 0].set_title('Left Gray Image (cam0)')

# ax[0, 1].imshow(disp_gray, cmap='viridis')
# ax[0, 1].set_title('Gray Stereo Disparity')

# ax[1, 0].imshow(first_rgb[0])
# ax[1, 0].set_title('Left RGB Image (cam2)')

# ax[1, 1].imshow(disp_rgb, cmap='viridis')
# ax[1, 1].set_title('RGB Stereo Disparity')

# # plt.show()