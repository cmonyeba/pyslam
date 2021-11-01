import numpy as np

class Point():
    def __init__(self, loc, map, color):
        self.frames = []
        self.idx = []
        self.loc = loc
        self.id = map.addPoint(self)
        self.color = np.copy(color)
    
    def addObservation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idx.append(idx)
 
        