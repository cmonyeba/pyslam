import numpy as np
from visual import visualize
import cv2
from display import Display3D
from point import Point
import pykitti
from function import calculateRotationMatrix, triangulate, cvtPoint, triangulate2
from frame import KeyFrame, matchFrames
from map import Map

class Slam:
    def __init__(self, W, H, K):
        self.map = Map(W, H)
        self.scale = 5
        self.W, self.H = W, H
        self.K = K
        
    def processFrame(self, frame):
        
        #Verify that the frame shape is correct and create KeyFrame object
        assert frame.shape[:2] == (self.H, self.W)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keyframe =  KeyFrame(gray, self.map, self.K)
        
        #Makes sure it is not the first frame
        #The most recent frame and previous frame are set as f1 and f2
        if keyframe.id == 0:
            return  
        f1 = self.map.frames[-1]
        f2 = self.map.frames[-2]
        #Calculate Matches and Extrinsic Matrix (R|t relation between the two frames)
        #Returns the KPS index for matching points
        idx1, idx2, Extrinsic = matchFrames(f1, f2)
         
        f1.pose = np.dot(Extrinsic, f2.pose)

        for i, idx in enumerate(idx2):
            if f2.pts[idx] is not None:
                f2.pts[idx].addObservation(f1, idx1[i])

       
        # print(f1.pose)
      


        # pts4d = triangulate(P1, P2, pts1, pts2)
        pts4d = triangulate2(f1.pose, f2.pose, f1.nkps[idx1], f2.nkps[idx2])
        pts4d = pts4d / pts4d[:, 3:]

        unmatched_points = np.array([f1.pts[i] is None for i in idx1])
        good_points4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points

        for idx, point in enumerate(pts4d):
            if not good_points4d[idx]:
                continue
            p = Point(point, self.map, (255, 255, 255))
            p.addObservation(f1, idx1[idx])
            p.addObservation(f2, idx2[idx])

        if keyframe.id != 0:
            ret = visualize(frame, f1.kps[idx1], f2.kps[idx2])
        
        if keyframe.id >=4:
            self.map.optimize()
            # ret = visualize(frame, f1.kps[idx1], f2.kps[idx2])
            # pass
            
        return ret
        
if __name__ == '__main__':
    
    basedir = '/home/sebaw/kitti'

    # Specify the dataset to load
    date = '2011_09_26'
    drive = '0001'

    # Load the data. Optionally, specify the frame range to load.38
    dataset = pykitti.raw(basedir, date, drive)
    # cap = cv2.VideoCapture(-1)
    #capture video in cap

    #create an SLAM instance
    W = 1242
    H = 375
    K = dataset.calib.K_cam2
    slam = Slam(W, H, K)
    disp3d = Display3D()

    a = None
    # while(cap.isOpened()):
    for i, frame in enumerate(dataset.cam2):    
        # Capture frame-by-frame
        # ret, frame = cap.read()
        # if ret:
            #gray-scale frame and resize
        
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
        Rt[:3,:3] = R
    
        
        C = np.dot(-np.transpose(Rt[:3,:3]), Rt[:3,3] )
        O = np.dot(np.transpose(R), np.array([0,0,1]))
        pose = np.eye(4)
        pose[:3,3] = C
        pose[:3,:3] = O
        
        frame = slam.processFrame(np.array(frame))
        
        if i != 0:    
            cv2.imshow('annotate', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            print('no frame')
        disp3d.paint(slam.map)    
    # When everything done, release the capture
    
    cap.release()
    cv2.destroyAllWindows()