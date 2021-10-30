import cv2
import numpy as np

def visualize(frame, f1 , f2):
    img = frame
    for i in range(len(f1.idx)):
        cv2.circle(img, (round(f1.kps[i][0]),round(f1.kps[i][1])), radius = 2, color = (0,255,0))
        cv2.line(img, (round(f1.kps[f1.idx[i]][0]),round(f1.kps[f1.idx[i]][1])), (round(f2.kps[f2.idx[i]][0]),round(f2.kps[f2.idx[i]][1])), color = (0,0,255))
    cv2.imshow('img', img)
