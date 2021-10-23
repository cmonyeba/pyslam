import numpy as np

# class Point():
#     def __init__(self) -> None:
#         self.wrldPt = None
#         # self.imgPts = []
#     def addPoint    

class Map():
    def __init__(self) -> None:
        self.frames = []
        self.maxFrame =  0 
        
    def addFrame(self, frame):
        #Sets return value as current amount of frames
        #Increments frame count and appends frame to frames list
        ret = self.maxFrame
        self.maxFrame += 1
        self.frames.append(frame)
        return ret