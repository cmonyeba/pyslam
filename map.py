import numpy as np
import g2o
import pangolin
from function import poseRt

class Map():
    def __init__(self, W, H):
        self.points = []
        self.frames = []
        self.maxPoint = 0
        self.maxFrame = 0
 
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
    
    #BUNDLE ADJUSTMENT FROM KAUNILD and G2opy
    def optimize(self):
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)

        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

        for fdx, frame in enumerate(self.frames):
            pose = frame.pose

            sbacam = g2o.SBACam(g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3]))
            sbacam.set_cam(frame.K[0][0], frame.K[1][1], frame.K[0][2], frame.K[1][2], 1.0)

            v_se3 = g2o.VertexCam()
            v_se3.set_id(frame.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(frame.id <= 1)
            opt.add_vertex(v_se3)

        PT_ID_OFFSET = 0x10000

        for pdx, point in enumerate(self.points):
            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(point.id + PT_ID_OFFSET)
            pt.set_estimate(point.loc[0:3])
            pt.set_marginalized(True)
            pt.set_fixed(False)
            opt.add_vertex(pt)

            for f in point.frames:
                edge = g2o.EdgeProjectP2MC()
                edge.set_vertex(0, pt)
                edge.set_vertex(1, opt.vertex(f.id))
                uv = f.kps[f.pts.index(point)]
                edge.set_measurement(uv)
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)

        opt.set_verbose(True)
        opt.initialize_optimization()
        opt.optimize(50)

        for frame in self.frames:
            est = opt.vertex(frame.id).estimate()
            R = est.rotation().matrix()
            t = est.translation()
            frame.pose = poseRt(R, t)

        # put points back
        for point in self.points:
            est = opt.vertex(point.id + PT_ID_OFFSET).estimate()
            point.location = np.array(est)