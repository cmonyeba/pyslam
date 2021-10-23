"""Example of pykitti.raw usage."""
import itertools
import statistics as stat
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import Improved_Slam
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
# print(dataset.calib
# .P_rect_00)
# print(dir(dataset.oxts[0].packet))
# print(dataset.oxts[0].packet.alt)
K = dataset.calib.K_cam0
# print(K)
# print(dataset.oxts[0].T_w_imu)
# print(dataset.oxts[1].T_w_imu)
temp = np.array(dataset.oxts[0].T_w_imu[:3])
temp2 = np.array(dataset.oxts[1].T_w_imu[:3])

# print(temp)
# print(temp2)
# print(temp)
# print(dataset.oxts[2].T_w_imu)
cam_pos = [0,0,0]
cam_xyz = []
add = 0

for i in range(len(dataset.oxts)):
    # print(dataset.oxts[i].packet.alt)
    vf = dataset.oxts[i].packet.vf
    vl = dataset.oxts[i].packet.vl
    alt = dataset.oxts[i].packet.alt

    cam_xyz.append([cam_pos[0] + vf * 1, cam_pos[1] + vl * 1, alt])
    cam_pos = [cam_pos[0] + vf * 1, cam_pos[1] + vl * 1, alt]
    # print(cam_pos)
# print(dataset.cam0)
# print(dataset.oxts)
H = 375
W = 1242
slam = Improved_Slam.Slam(W, H, K)

for i, frame in enumerate(dataset.cam0):
    # print(type(np.array(frame)))
    # print(i)
    slam.processFrame(np.array(frame))

    Visual.extract_features(np.array(frame))
    # cv2.imshow('show', frame)
    # cv2.waitKey(100)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      
cam_xyz = np.array(slam.cam_xyz)
kps_xyz = np.array(slam.kps_xyz)
# print(kps_xyz)
# print(cam_xyz)
fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim3d(0, 600)
ax.set_ylim3d(0, 5)
ax.set_zlim3d(0, 30)
# ax.view_init(0, 150)
ax.scatter(cam_xyz[:, [0]], cam_xyz[:, [1]], cam_xyz[:, [2]], c='r', s=1)
ax.scatter(kps_xyz[:, [0]], kps_xyz[:, [1]], kps_xyz[:, [2]], s=1)
plt.show() 

# for i in range(len(dataset.oxts)):


   
    
    # print(dataset.oxts[i].packet.vf, dataset.oxts[i].packet.vl)
        
    # # print(dataset.oxts[i].packet.alt)
    # print(dataset.oxts[i].packet.ax, dataset.oxts[i].packet.ay, dataset.oxts[i].packet.az)
    # print(dataset.oxts[i].packet.af, dataset.oxts[i].packet.al, dataset.oxts[i].packet.au)
    # z = dataset.oxts[i].packet.alt
    # _, _, _, _, K, R, t = cv2.decomposeProjectionMatrix(np.array(dataset.oxts[i].T_w_imu[:3]))
    # t = t/np.linalg.norm(t)
    # print(t[0], t[1], t[1])

    # cam_xyz.append([cam_pos[0] + t[0], cam_pos[1] + t[1], cam_pos[2] + t[2]])
    # cam_pos = [cam_pos[0] + t[0], cam_pos[1] + t[1], cam_pos[2] + t[2]]
    
# cam_xyz = np.array(cam_xyz)
# # kps_xyz = np.array(slam.kps_xyz)
# fig = plt.figure()
# ax = Axes3D(fig)
# # ax.view_init(0, 150)
# # ax.scatter(kps_xyz[:, [0]], kps_xyz[:, [1]], kps_xyz[:, [2]], s=1)


    
#     # kps, des = orb.detectAndCompute(np.array())
    
#     # frame = cv2.drawKeypoints(np.array(frame), kps, np.array([]), (0,255,0))
    
    
   


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