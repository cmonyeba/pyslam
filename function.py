import numpy as np

def triangulate(P1, P2, pts1, pts2):
        A = np.eye(4)
        ret = np.zeros((pts1.shape[0], 4))
        for i in range(len(pts1)):
            temp1 = np.cross(pts1[i], P1.T)
            temp2 = np.cross(pts2[i], P2.T)
            te1 = temp1.T
            te2 = temp2.T
            A[:2, :4] = te1[:2]
            A[2:4, :4] = te2[:2]
            _, _, V = np.linalg.svd(A)
            ret[i] = V[3]
        return ret
    
def triangulate2(pose1, pose2, points1, points2):
    ret = np.zeros((points1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(points1, points2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret
    
def cvtPoint(pts):
    ret = []
    for pt in pts:
        temp = np.zeros(3)
        temp[:2] = pt
        temp[2] = 1
        ret.append(temp)
    return np.array(ret)

def calculateRotationMatrix(y, p, r):
    R_yaw = np.matrix([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R_pitch = np.matrix([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    R_roll = np.matrix([[1, 0 ,0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])

    R =  R_roll * R_yaw * R_pitch
    # R1 = np.dot(R_yaw, R_pitch)
    # R = np.dot(R1, R_roll)
    # print(np.linalg.eigvals(R))
    #The Determinate of the Rotation matrix = 1 then proper rotation?
    return R
    