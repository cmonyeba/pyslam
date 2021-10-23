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
    
def cvtPoint(pts):
    ret = []
    for pt in pts:
        temp = np.zeros(3)
        temp[:2] = pt
        temp[2] = 1
        ret.append(temp)
    return np.array(ret)
    