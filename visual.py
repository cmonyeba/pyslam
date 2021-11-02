import cv2
import numpy as np

def visualize(frame, pts1, pts2):
    ret = frame
    for pt1, pt2 in zip(pts1, pts2):
        print(pt1,pt2)
        cv2.circle(ret, (round(pt1[0]),round(pt1[1])), radius = 2, color = (0,255,0))
        cv2.line(ret, (round(pt1[0]),round(pt1[1])),(round(pt2[0]), round(pt2[1])), color = (0,0,255))
    return ret