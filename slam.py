import numpy as np
from visual import visualize
import cv2
import math
from display import Display3D
from point import Point
import pykitti
from function import calculateRotationMatrix, triangulate, cvtPoint, triangulate2
from frame import KeyFrame, matchFrames
from map import Map

# # Change this to the directory where you store KITTI data
# basedir = '/Users/sebaw/Documents/kitti/raw'

# # Specify the dataset to load
# date = '2011_09_26'
# drive = '0035'

# # Load the data. Optionally, specify the frame range to load.
# dataset = pykitti.raw(basedir, date, drive)

class Slam:
    def __init__(self, W, H, K):
        self.cam_position = [0,0,0]
        self.map = Map()
        self.scale = 5
        self.W, self.H = W, H
        #Intrinsic Matrix
        self.K = K
        # self.alt = 0
        # self.count = 0
        
    def processFrame(self, frame, Rt):
        
        #Verify that the frame shape is correct and create KeyFrame object
        assert frame.shape[:2] == (self.H, self.W)
        frame =  KeyFrame(frame, self.map, self.K)
        
        #Makes sure it is not the first frame
        #The most recent frame and previous frame are set as f1 and f2
        if frame.id == 0:
            return  
        f1 = self.map.frames[-1]
        f2 = self.map.frames[-2]
        
        #Calculate Matches and Extrinsic Matrix (R|t relation between the two frames)
        idx1, idx2, Extrinsic = matchFrames(f1, f2, self.K)
        f1.idx = idx1
        f2.idx = idx2
        f1.pts = [f1.kps[idx] for idx in idx1]
        f2.pts = [f2.kps[idx] for idx in idx2]
       
        
        f1.pose =  np.dot(Extrinsic, f2.pose)

        P1 = np.dot(f1.K, np.array(f1.pose)[:3])
        P2 = np.dot(f2.K, np.array(f2.pose)[:3])
        
        #u = PX
        
        
        pts1 = cvtPoint(np.array(f1.pts))
        pts2 = cvtPoint(np.array(f2.pts))
       

        pts4d2 = triangulate(P1, P2, pts1, pts2)
        pts4d = triangulate2(f1.pose, f2.pose, pts1, pts2)
        
        pts4d = pts4d / pts4d[:, 3:]
        good_points4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
        for idx, point in enumerate(pts4d):
            if not good_points4d[idx]:
                continue
            color = (0,0,255)
            p  = Point(point, self.map, color)
            # p.addObservation(f1, idx1[idx])
            # p.addObservation(f2, idx2[idx])
            # self.kps_xyz.append(pt)
            # self.kps_xyz.append([pt[0] * self.scale + self.cam_position[0],
            #             pt[1] * self.scale + self.cam_position[1],
            #             pt[2] * self.scale + self.cam_position[2]])

        # print(self.kps_xyz)
        # self.cam_xyz.append([self.cam_position[0] + f1.pose[:3, 3][0], self.cam_position[1] + f1.pose[:3, 3][1], self.cam_position[2] + f1.pose[:3, 3][2]])

        

        # self.cam_position = [self.cam_position[0] + f1.pose[:3, 3][0], self.cam_position[1] + f1.pose[:3, 3][1], self.cam_position[2] + f1.pose[:3, 3][2]]

if __name__ == '__main__':
    
    basedir = '/home/swueste1/kitti'

    # Specify the dataset to load
    date = '2011_09_26'
    drive = '0035'

    # Load the data. Optionally, specify the frame range to load.
    dataset = pykitti.raw(basedir, date, drive)
    # print(dataset)
    #capture video in cap
    cap = cv2.VideoCapture('images/drive.mp4')

    #create an SLAM instance
    W = 1242
    H = 375
    # K = np.array([[F,0,W//2],[0,F,H//2],[0,0,1]])
    K = dataset.calib.K_cam2
    # print(np.dot(np.matrix(K).T, np.matrix(K)))
    slam = Slam(W, H, K)
    disp3d = Display3D()

    # slam.runSlam(old_frame)

    # print(slam.kpsxyz)
    pos = [0,0,0]
    a = None
    # while(cap.isOpened()):
    for i, frame in enumerate(dataset.cam2):    
        # Capture frame-by-frame
        # ret, frame = cap.read()
        # print(frame.shape)
        # if ret:
            #gray-scale frame and resize
        gray = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)
        # print(np.array(frame).shape)
            # print(frame)
        vf = dataset.oxts[i].packet.vf
        vl = dataset.oxts[i].packet.vl
        alt = dataset.oxts[i].packet.alt
        # print(dataset.oxts[i].packet.lat, dataset.oxts[i].packet.lon)
        # print(dataset.oxts[i].packet.ax, dataset.oxts[i].packet.ay, dataset.oxts[i].packet.az)
        vec = np.array([vf,vl,alt])
        
        roll = dataset.oxts[i].packet.roll
        pitch = dataset.oxts[i].packet.pitch
        yaw = dataset.oxts[i].packet.yaw
        R = calculateRotationMatrix(yaw, pitch, roll)
        if a == None:
            a = alt
        # print(vec)
        Rt = np.eye(4)
        temp = np.transpose(R)
        p = np.dot(R, temp)
        
        Rt[2,3] = vf
        Rt[1,3] = vl
        Rt[0,3] = alt/a
        # print(Rt)
        # print(Rt)
        # Rt[2,3] = i
        # Rt[2,2] = 1
        # Rt[1,1] = 1
        # Rt[0,0] = 1
        
        C = np.dot(-np.transpose(Rt[:3,:3]), Rt[:3,3] )
        O = np.dot(np.transpose(R), np.array([0,0,1]))
        pose = np.eye(4)
        pose[:3,3] = C
        pose[:3,:3] = O
        pos = [ (pos[0] - vf) * 0.5  , (pos[1] - vl) * 0.5, alt/alt]
        slam.processFrame(np.array(gray), Rt)
        if i > 0:
            visualize(np.array(frame), slam.map.frames[-1], slam.map.frames[-2])
             
  

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # else:
        #     print('no frame')
        disp3d.paint(slam.map)    
    # When everything done, release the capture
    
    cap.release()
    cv2.destroyAllWindows()