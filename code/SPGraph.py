import networkx as nx
import math
import cv2
import numpy as np

class SPNode():
    
    def __init__(self):
        self.label = None
        self.pixels = []
        self.centroid = ()
        self.type = 'na'
        self.meanCIELab = None
        self.CIELabHist = None
        self.realCIELab = None
    
    def __repr__(self):
        return str(self.label)

class superPixelGraph:
    
    def __init__(self):
        self.rangeOfL = [0,256]
        self.rangeOfa = [0,256]
        self.rangeOfb = [0,256]
        self.noOfBins = [32,32,32]
    
    def twoDdist (self, p0, p1):
        return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)
    
    def drawCentroids(self, I, superPixelList):
        for each in superPixelList:
            i,j = each.centroid
            I[i][j] = 128
        return I
    
    def drawSPMask(self, I,SP):
        I1 = np.zeros(I.shape)
        I1 = np.copy(I)
        mask = SP.getLabelContourMask()
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i][j] == -1 or mask[i][j] == 255:
                    I1[i][j] = [128,128,128]
        return I1
    
    def SPSLICGenerator(self, I, sizeOfRegion):
        SLICO = 101
        iterations = 4
        SPSLIC = cv2.ximgproc.createSuperpixelSLIC(I, algorithm = SLICO, region_size = sizeOfRegion, ruler = 10.0)
        SPSLIC.iterate(num_iterations = iterations)
        return SPSLIC
    
    def graphGenerator (self, superPixelList, objHist, bkgdHist):
        G = nx.Graph()
        s = SPNode()
        s.label = 's'
        t = SPNode()
        t.label = 't'
        lambda1 = .9
        sigma1 = 5
        objHistSum = int(objHist.sum())
        bkgdHistSum = int(bkgdHist.sum())
        for u in superPixelList:
            K = 0
            radius = math.sqrt(len(u.pixels) / math.pi)
            for v in superPixelList:
                if u != v:
                    if self.twoDdist(u.centroid, v.centroid) <= 2.5 * radius:
                        sim = math.exp(-(cv2.compareHist(u.CIELabHist,v.CIELabHist,3)**2 / 2 * sigma1**2)) * (1 / self.twoDdist(u.centroid, v.centroid))
                        K+=  sim
                        G.add_edge(u, v, sim = sim)
            if(u.type == 'na'):
                l1,a1,b1 = [int(x) for x in u.meanCIELab]
                li = int(l1 // ((self.rangeOfL[1] - self.rangeOfL[0]) / self.noOfBins[0]))
                ai = int(a1 // ((self.rangeOfa[1] - self.rangeOfa[0]) / self.noOfBins[1]))
                bi = int(b1 // ((self.rangeOfb[1] - self.rangeOfb[0]) / self.noOfBins[2]))
                objProb = int(objHist[li,ai,bi]) / objHistSum
                bkgdProb = int(bkgdHist[li,ai,bi]) / bkgdHistSum
                sim_s = 100000
                sim_t = 100000
                if bkgdProb > 0:
                    sim_s = lambda1 * -np.log(bkgdProb)
                if objProb > 0:
                    sim_t = lambda1 * -np.log(objProb)
                G.add_edge(s, u, sim = sim_s)
                G.add_edge(t, u, sim = sim_t)
            if(u.type == 'ob'):
                G.add_edge(s, u, sim = 1 + K)
                G.add_edge(t, u, sim = 0)
            if(u.type == 'bg'):
                G.add_edge(s, u, sim = 0)
                G.add_edge(t, u, sim = 1 + K)		
        return G