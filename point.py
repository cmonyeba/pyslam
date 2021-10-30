import numpy as np

class Point():
    def __init__(self, point, map, color) -> None:
        self.frames = []
        self.loc = point
        self.idxs = []
        self.id = map.addPoint(self)
        self.color = np.copy(color)
    
    # def addObservation(self, frame, idx):
    #     # assert frame.pts[idx] is None
    #     assert frame not in self.frames
    #     frame.pts[idx] = self
        
    #     print(idx)
    #     self.frames.append(frame)
    #     self.idxs.append(idx)
 
        