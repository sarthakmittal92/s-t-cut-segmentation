import sys, getopt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import MaxFlowMinCutBK
import SPGraph

drawing = False
mode = "ob"
objectPixels = []
bkgdPixels = []
I = None
I1 = None
SPGObj = SPGraph.superPixelGraph()

def markSeeds (event,x,y,flags,param):
	global drawing, mode, bkgdPixels, objectPixels, I1
	h,w,_ = I1.shape
	if event ==  cv2.EVENT_LBUTTONDOWN:
		drawing = True
	elif event ==  cv2.EVENT_MOUSEMOVE:
		if drawing ==  True:
			if mode ==  "ob":
				if(x >= 0 and x <= w - 1) and (y > 0 and y <= h - 1):
					objectPixels.append((y,x))
				cv2.line(I1,(x - 3,y),(x + 3,y),(0,0,255))
			else:
				if(x >= 0 and x <= w - 1) and (y > 0 and y <= h - 1):
					bkgdPixels.append((y,x))
				cv2.line(I1,(x - 3,y),(x + 3,y),(255,0,0))
	elif event ==  cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode ==  "ob":
			cv2.line(I1,(x - 3,y),(x + 3,y),(0,0,255))
		else:
			cv2.line(I1,(x - 3,y),(x + 3,y),(255,0,0))

inputfile = ''
try:
    opts, args = getopt.getopt(sys.argv[1:], "i:", ["input-image = "])
except getopt.GetoptError:
    print('Segmentation.py -i <input image>')
    sys.exit(2)
for opt, arg in opts:
    if opt in ("-i"):
        inputfile = arg
print('Using image:', inputfile)

I = cv2.imread(inputfile)
I1 = np.zeros(I.shape)
I1 = np.copy(I)

h,w,c = I.shape
region_size = 20
cv2.namedWindow('Mark foreground/background')
cv2.setMouseCallback('Mark foreground/background', markSeeds)
while(1):
    cv2.imshow('Mark foreground/background',I1)
    k = cv2.waitKey(1) & 0xFF
    if k ==  ord('o'):
        mode = "ob"
    elif k ==  ord('b'):
        mode = "bg"
    elif k ==  27:
        break
cv2.destroyAllWindows()

CLELabI = cv2.cvtColor(I, cv2.COLOR_BGR2Lab)
SP = SPGObj.SPSLICGenerator(I,region_size)
superPixelLabels = SP.getLabels()
listOfSuperPixels = [None for each in range(SP.getNumberOfSuperpixels())]

for i in range(h):
    for j in range(w):
        if not listOfSuperPixels[superPixelLabels[i][j]]:
            tempSP = SPGraph.SPNode()
            tempSP.label = superPixelLabels[i][j]
            tempSP.pixels.append((i,j))
            listOfSuperPixels[superPixelLabels[i][j]] = tempSP
        else:
            listOfSuperPixels[superPixelLabels[i][j]].pixels.append((i,j))

for sp in listOfSuperPixels:
    noOfPixels = len(sp.pixels)
    sumOfi = 0
    sumOfj = 0
    sumOfLab = [0,0,0]
    tempMask = np.zeros((h,w),np.uint8)
    for each in sp.pixels:
        i,j = each
        sumOfi += i
        sumOfj += j
        sumOfLab = [x + y for x, y in zip(sumOfLab, CLELabI[i][j])]
        tempMask[i][j] = 255
    sp.CIELabHist = cv2.calcHist([CLELabI],[0,1,2],tempMask,SPGObj.noOfBins,SPGObj.rangeOfL + SPGObj.rangeOfa + SPGObj.rangeOfb)
    sp.centroid += (sumOfi // noOfPixels, sumOfj // noOfPixels,)
    sp.meanCIELab = [x / noOfPixels for x in sumOfLab]
    sp.realCIELab = [sp.meanCIELab[0] * 100 / 255,sp.meanCIELab[1] - 128,sp.meanCIELab[2] - 128]

for pixels in objectPixels:
    x,y = pixels
    listOfSuperPixels[superPixelLabels[x][y]].type = "ob"
for pixels in bkgdPixels:
    x,y = pixels
    listOfSuperPixels[superPixelLabels[x][y]].type = "bg"
I2 = SPGObj.drawSPMask(I,SP)
I2 = SPGObj.drawCentroids(I2,listOfSuperPixels)
objMask = np.zeros((h,w),dtype = np.uint8)
for pixels in objectPixels:
    i,j = pixels
    objMask[i][j] = 255
bkgdMask = np.zeros((h,w),dtype = np.uint8)
for pixels in bkgdPixels:
    i,j = pixels
    bkgdMask[i][j] = 255

objHist = cv2.calcHist([CLELabI],[0,1,2],objMask,SPGObj.noOfBins,SPGObj.rangeOfL + SPGObj.rangeOfa + SPGObj.rangeOfb)
bkgdHist = cv2.calcHist([CLELabI],[0,1,2],bkgdMask,SPGObj.noOfBins,SPGObj.rangeOfL + SPGObj.rangeOfa + SPGObj.rangeOfb)
G = SPGObj.graphGenerator(listOfSuperPixels,objHist,bkgdHist)

for each in G.nodes():
    if each.label == 's':
        s = each
    if each.label == 't':
        t = each

RG = MaxFlowMinCutBK.BK(G, s, t, capacity = 'sim').getResidual()
sTree, tTree = RG.graph['trees']
partition = (set(sTree), set(G) - set(sTree))
F = np.zeros((h,w),dtype = np.uint8)
for sp in partition[0]:
    for pixels in sp.pixels:
        i,j = pixels
        F[i][j] = 1
Final = cv2.bitwise_and(I,I,mask = F)

CIELabSP = np.zeros(I.shape,dtype = np.uint8)
for sp in listOfSuperPixels:
    for pixels in sp.pixels:
        i,j = pixels
        CIELabSP[i][j] = sp.meanCIELab
CIELabSP = cv2.cvtColor(CIELabSP, cv2.COLOR_Lab2RGB)

plt.subplot(2,2,1)
plt.tick_params(labelcolor = 'black', top = 'off', bottom = 'off', left = 'off', right = 'off')
plt.imshow(I[...,::-1])
plt.axis("off")
plt.xlabel("Input image")

plt.subplot(2,2,2)
plt.imshow(Final[...,::-1])
plt.axis("off")
plt.xlabel("Output Image")

print('Writing output to: ./results/out.png')
cv2.imwrite("./results/out.png",Final)
plt.show()