import numpy as np


class Map():
    def __init__(self) -> None:
        #Frames w/ poses
        self.frames = []
        self.maxFrame =  0
        #3D points 
        self.points  = []
        self.maxPoint = 0
        
    def addFrame(self, frame):
        #Sets return value as current amount of frames
        #Increments frame count and appends frame to frames list
        ret = self.maxFrame
        self.maxFrame += 1
        self.frames.append(frame)
        return ret
    
    def addPoint(self, point):
        #Sets return value as current amount of frames
        #Increments frame count and appends frame to frames list
        ret = self.maxPoint
        self.maxPoint += 1
        self.points.append(point)
        return ret