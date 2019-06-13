#
# from src.Utilities import *
# import cv2
# import numpy as np
#

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def showImages(images, titles=None, mainTitle=None):
    # This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    if mainTitle is not None: fig.suptitle(mainTitle)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def getYBoundaries(img):
    # note that this function is used on images from the IAM database only
    lines = cv2.HoughLinesP(image=img, rho=1, theta=np.pi / 180, threshold=50, minLineLength=int(0.2 * img.shape[1]),
                            maxLineGap=10)#TODO: if none adjust

    topHorizontals = []  # will contain the y coordinates of the 2 points representing the line
    bottomHorizontals = []  # will contain the y coordinates of the 2 points representing the line
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if (abs(y1 - y2) < int(0.1 * img.shape[1] - 1)):
            if (y1 > int(img.shape[0] / 2)):  # horizontal lines at the bottom of the page
                bottomHorizontals.append([y1, y1])
            else:  # horizontal lines at the top of the page
                topHorizontals.append([y1, y1])
    return (np.min(np.asarray(bottomHorizontals)) - 10, np.max(np.asarray(topHorizontals)) + 10)


# local thresholding (may not be used in the project)
def localThreshRows(img):
    imgLocalRows = np.zeros(img.shape, np.uint8)
    for row in range(img.shape[0]):
        _, imgLocalRows[row:row + 1, :] = cv2.threshold(img[row:row + 1, :], 0, 255,
                                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgLocalRows


def localThreshColumns(img):
    imgLocalColumns = np.zeros(img.shape, np.uint8)
    for col in range(img.shape[1]):
        _, imgLocalColumns[:, col:col + 1] = cv2.threshold(img[:, col:col + 1], 0, 255,
                                                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgLocalColumns


def randomColor():
    levels = range(32, 256, 32)
    return tuple(int(random.choice(levels)) for _ in range(3))



def mergeRects(corners1, corners2):
    x1 = min([corners1[0], corners2[0]])
    y1 = min([corners1[1], corners2[1]])
    x2 = max([corners1[2], corners2[2]])
    y2 = max([corners1[3], corners2[3]])
    return [x1, y1, x2, y2]


def getSWROnRight(swrI, SWRCentroids, CRCntrs, SWRCorners):
    # get nearest 3 of the SWRs  on the right of this SWR denoted by swrI
    deltaX = SWRCentroids[:, 0] - SWRCentroids[swrI, 0]
    deltaX[swrI] = 1
    deltaY = SWRCentroids[:, 1] - SWRCentroids[swrI, 1]
    deltaY[swrI] = 2
    tanTheta = deltaY / deltaX
    thetas = abs(np.arctan(tanTheta))

    nonOverlappingMask = np.bitwise_and(((SWRCorners[swrI, 3] - SWRCorners[:, 1])>0) , ((SWRCorners[swrI, 1] - SWRCorners[:, 3])<0))
    onRightMask = np.bitwise_and((deltaX > 0), (thetas <= (30 * np.pi / 180)))
    onRightMask = np.bitwise_and(onRightMask,nonOverlappingMask)
    onRight = np.where(onRightMask)[0]  # np.where(thetas<=(np.pi/4))[0] #np.where((SWRCentroids[:,0]-SWRCentroids[swrI,0])>0)[0]
    onRight = onRight[onRight > 0]
    if onRight.shape[0] == 0:
        return -1
    rightCentroids = SWRCentroids[onRight, :]

    # k = 3
    # tempNearestKIdx = []
    # if onRight.shape[0] <= 3:
    #     k = onRight.shape[0] - 1
    # if k <= 1:
    #     tempNearestKIdx = np.asarray([np.argmin(np.sum((rightCentroids[:] - SWRCentroids[swrI]) ** 2, axis=1))])
    # else:
    #     tempNearestKIdx = np.argpartition(np.sum((rightCentroids[:] - SWRCentroids[swrI]) ** 2, axis=1), k)[:k]
    #
    # nearestKIdx = onRight[tempNearestKIdx]

    nearestKIdx = onRight
    dists = []
    for ki in nearestKIdx:
        dists.append(
            cv2.pointPolygonTest(contour=np.asarray(CRCntrs[ki]), pt=(SWRCentroids[swrI, 0], SWRCentroids[swrI, 1]),
                                 measureDist=True))
    label = nearestKIdx[np.argmax(np.asarray(dists))]
    return label

randomColors = np.zeros((2500, 3), np.uint8)
for i in range(2500):
    randomColors[i, :] = randomColor()
randomColors[0, :] = [255, 255, 255]

def segment(imgPath,kernelSize,swrDistThreshCoff,occCoff,distanceMergingPercentile=None):
    #Extracting image

    # print("(1) Exctacting image", imgPath)
    imgName = filename[filename.find('\\') + 1:]
    rawImg = cv2.imread(imgPath)
    imgGray = cv2.cvtColor(rawImg, cv2.COLOR_BGR2GRAY)
    imgInverted = 255 - imgGray

    otsuThresh, imgOtsuFull=cv2.threshold(imgInverted,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    yBottom, yTop = getYBoundaries(imgOtsuFull)
    imgOtsu = np.copy(imgOtsuFull)
    imgOtsu = imgOtsu[yTop:yBottom,15:2465] #these boundaries are specific to IAM images.
    imgOriginal = rawImg[yTop:yBottom,15:2465,:]
    # showImages([imgOtsuFull,imgOtsu])

    ######################## 2- Getting connected components and computing average height ########################
    # print("(2) getting connected components")

    ccNumOtsu,ccImgOtsu,ccStatsOtsu,ccCentroidsOtsu = cv2.connectedComponentsWithStats(imgOtsu)
    #for illustraion purposes
    # print("\tFetched components")

    #indexes of statistics in ccStats
    leftIndex, topIndex, widthIndex, heightIndex, areaIndex = range(5)

    #coloring each CC with a random color
    # imgCCLayeredOtsu = np.zeros(imgOriginal.shape).astype(np.uint8)
    # imgCCLayeredOtsu[:,:]=randomColors[ccImgOtsu]
    #removing CCs resulting from noise
    noisyCCArea = 20
    noiseCCIdx = np.where(ccStatsOtsu[:,areaIndex]<=noisyCCArea)[0]
    ccNumOtsu -= len(noiseCCIdx)
    ccCentroidsOtsu = ccCentroidsOtsu[ccStatsOtsu[:,areaIndex]>noisyCCArea]
    ccStatsOtsu = ccStatsOtsu[ccStatsOtsu[:,areaIndex]>noisyCCArea]

    #surrounding each CC with a rectangle
    # imgCCLayeredWithRectOtsu = np.copy(imgCCLayeredOtsu)
    # for ccIndex in range(1,ccNumOtsu):
    #     stat = ccStatsOtsu[ccIndex]
    #     cv2.rectangle(imgCCLayeredWithRectOtsu, (stat[leftIndex], stat[topIndex]), (stat[leftIndex] +stat[widthIndex], stat[topIndex]+stat[heightIndex]), (0, 255, 0), 7)


    #calculating average height of connected components
    avgHeight = np.average(ccStatsOtsu[:,heightIndex])
    dividingHeight = np.percentile(ccStatsOtsu[:,heightIndex],80)
    distanceMergingHeight = avgHeight
    if distanceMergingPercentile is not None:
        distanceMergingHeight = np.percentile(ccStatsOtsu[:, heightIndex], distanceMergingPercentile)

    ######################## 3- Filtering binary image with isotropic LoG filter to get filtered image ########################
    # print("(3) Applying gaussian")
    #todo: look at it again with oso and timo
    # ddepth = cv2.CV_16S
    # kernelSize = 61#31

    gaussianImg = cv2.GaussianBlur(imgOtsu,(kernelSize,kernelSize),2.5*avgHeight)
    # laplacianImg = cv2.Laplacian(gaussianImg,ddepth,kernelSize)
    imgFiltered = gaussianImg

    ######################## 4- Binarizing filtered image ########################
    # print("(4) Binarizing filtered image")
    _,imgFilteredBinarized = cv2.threshold(imgFiltered,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    ccNumFilteredBinarized,ccImgFilteredBinarized,ccStatsFilteredBinarized,ccCentroidsFilteredBinarized = cv2.connectedComponentsWithStats(imgFilteredBinarized)

    #get contours of connected regions
    CRCntrs = [[]]
    for crI in range(1,ccNumFilteredBinarized):#loop on connected regions

        roi = (ccImgFilteredBinarized==crI).astype(np.uint8)
        _, roiCntrs, hierarchy = cv2.findContours(roi,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #CHAIN_APPROX_NONE
        cntr = max(roiCntrs, key=lambda lis: len(lis))
        CRCntrs.append(cntr)

    #for illustraion purposes
    # imgCCLayeredFilteredBinarized = np.zeros(imgOriginal.shape).astype(np.uint8)
    # imgCCLayeredFilteredBinarized[:,:]=randomColors[ccImgFilteredBinarized]
    # print("end")
    ######################## 5- Assigning each CC to neares CR (connected region) to form SWRs (sub word regions) ########################
    # print("(5) Forming SWRs by assigning each cc to nearest CR")


    #bounding rect of SWR
    SWRCrdnts = np.zeros((ccNumFilteredBinarized,4),int)
    SWRCrdnts[:,0:2] = ccCentroidsFilteredBinarized[:]
    SWRCrdnts[:,2:4] = ccCentroidsFilteredBinarized[:]
    #to which SWR does each CC belong to #todo: find out whether we're using or not
    otsuLabels = np.zeros((ccNumOtsu),np.uint8)

    #get nearest k CRs to each CC in binarized image using distance between centroids
    k = 8
    for ccOtsuI in range(1,ccNumOtsu):
        nearestKIdx = np.argpartition(np.sum((ccCentroidsFilteredBinarized[1:] - ccCentroidsOtsu[ccOtsuI])**2,axis=1),k)[:k]
        nearestKIdx += 1
        dists = []
        for ki in nearestKIdx:
            dists.append(cv2.pointPolygonTest(contour=np.asarray(CRCntrs[ki]),pt=(ccCentroidsOtsu[ccOtsuI,0],ccCentroidsOtsu[ccOtsuI,1]),measureDist=True))
        label = nearestKIdx[np.argmax(np.asarray(dists))]

        #assign this cc to its SWR
        otsuLabels[ccOtsuI] = label

        #adjust bounding rect
        x1 = min([SWRCrdnts[label,0], ccStatsOtsu[ccOtsuI,leftIndex]])
        y1 = min([SWRCrdnts[label,1], ccStatsOtsu[ccOtsuI,topIndex]])
        x2 = max([SWRCrdnts[label,2], ccStatsOtsu[ccOtsuI,leftIndex] + ccStatsOtsu[ccOtsuI,widthIndex]])
        y2 = max([SWRCrdnts[label,3], ccStatsOtsu[ccOtsuI,topIndex] + ccStatsOtsu[ccOtsuI,heightIndex]])
        SWRCrdnts[label,:] = [x1,y1,x2,y2]



    #get centroids of SWRs
    SWRCentroids = np.zeros((ccNumFilteredBinarized,2))
    SWRCentroids[:,0] = np.average(SWRCrdnts[:,0::2],axis=1)
    SWRCentroids[:,1] = np.average(SWRCrdnts[:,1::2],axis=1)

    #for illustraion purposes
    for swrI in range(1,ccNumFilteredBinarized):#loop on SWRs
        roi = ccImgOtsu[SWRCrdnts[swrI, 1]:SWRCrdnts[swrI, 3], SWRCrdnts[swrI, 0]:SWRCrdnts[swrI, 2]]
        roi[roi[:,:]!=0]=swrI
    # for illustraion purposes
    # imgSWRLayer = np.zeros(imgOriginal.shape).astype(np.uint8)
    # imgSWRLayer[:, :] = randomColors[ccImgOtsu]
    # imgSWRLayerWithRect = np.copy(imgSWRLayer)
    # for swrI in range(1,ccNumFilteredBinarized):#loop on SWRs
    #     cv2.rectangle(imgSWRLayerWithRect, (SWRCrdnts[swrI,0],SWRCrdnts[swrI,1]), (SWRCrdnts[swrI,2], SWRCrdnts[swrI,3]), (0, 255, 0), 7)

    ######################## 6- Merging SWRs to form word regions (WRs) according to distances between SWRs ########################
    # print("(6) Merging SWRs to form word regions (WRs)")

    swrRights = np.asarray(list(range(0,ccNumFilteredBinarized)))
    # def union(child, parent):
    #     swrRights[swrRights==child] = parent

    #merge SWRs if horizontal distance is less than a threshold swrDistThresh
    swrDistThresh = swrDistThreshCoff*distanceMergingHeight#swrDistThreshCoff*avgHeight #todo: adjust swrDistThresh
    #for illustration purposes
    # imgLinesLayout = np.copy(imgSWRLayerWithRect)

    #first we find the SWR on the right of each SWR
    for swrI in range(1,ccNumFilteredBinarized):
        rightI = getSWROnRight(swrI,SWRCentroids,CRCntrs,SWRCrdnts)
        if rightI == -1:
            continue
        if (SWRCrdnts[rightI,0] - SWRCrdnts[swrI,2])<swrDistThresh:
            swrRights[swrRights==swrI] = rightI#union(swrI, rightI)
        #for illustration purposes
        # cv2.line(imgLinesLayout,(int(SWRCentroids[swrI,0]),int(SWRCentroids[swrI,1])),(int(SWRCentroids[rightI,0]),int(SWRCentroids[rightI,1])),(255,0,0),15)

    tempWRs = np.unique(swrRights[1:])
    wrNum = len(tempWRs)
    wrIdx = np.asarray(list(range(len(tempWRs))))

    wrCorners = np.zeros((wrNum,4),int)
    wrCorners[:,0] = 99999
    wrCorners[:,1] = 99999
    wrCorners[:,2] = -1
    wrCorners[:,3] = -1


    for swrI in range(1,ccNumFilteredBinarized):
        assignedWR = np.where(swrRights[swrI]==tempWRs)[0][0]
        wrCorners[assignedWR] = mergeRects(SWRCrdnts[swrI],wrCorners[assignedWR])
    for swrI in range(1,ccNumFilteredBinarized):
        assignedWR = np.where(swrRights[swrI] == tempWRs)[0][0]
        roi = ccImgOtsu[wrCorners[assignedWR, 1]:wrCorners[assignedWR, 3], wrCorners[assignedWR, 0]:wrCorners[assignedWR, 2]]
        roi[roi[:, :] == swrI] = assignedWR + 1
    # imgWRLayer = np.zeros(imgOriginal.shape).astype(np.uint8)
    # imgWRLayer[:, :] = randomColors[ccImgOtsu]


    # imgWRLayerWithRect = np.copy(imgWRLayer)
    # for wrI in range(0,wrNum):#loop on WRs
    #     cv2.rectangle(imgWRLayerWithRect, (wrCorners[wrI,0],wrCorners[wrI,1]), (wrCorners[wrI,2], wrCorners[wrI,3]), (randomColors[wrI+1,:]).tolist(), 7)

    ######################## 7- Splitting overlapping connected components (OCCs) running along multiple lines ########################
    # print("(7) Splitting OCCs")
    wrHeights = np.zeros((wrNum),int)
    wrHeights = wrCorners[:,3]-wrCorners[:,1]
    occIdx = np.where(wrHeights>=(occCoff*dividingHeight))[0]

    occCorners = np.copy(wrCorners[occIdx])
    occHeights = np.copy(wrHeights[occIdx])

    newWRCorners = []
    newWRParents = []
    for occI in range(0,occCorners.shape[0]):#todo change it to loop 3ale tala3nah
        numCCs = 2#int(round(occHeights[occI]/maxHeight))
        # print(occHeights[occI],dividingHeight,occCoff*dividingHeight,numCCs)
        # if numCCs <= 1:
        #     numCCs = 2
        ccHeight = int(round(occHeights[occI]/numCCs))
        x1, x2 = occCorners[occI,0], occCorners[occI,2]
        y1, y2 = 0, 0
        dividedWRs = []
        for yi in range(0,numCCs-1):
            y1 = yi * ccHeight +occCorners[occI,1]
            y2 = y1 + ccHeight - 1
            dividedWRs.append([x1,y1,x2,y2])
        y1 = y2 + 1
        y2 = occCorners[occI,3]
        dividedWRs.append([x1,y1,x2,y2])
        dividedWRs = np.asarray(dividedWRs)

        divWRI = occIdx[occI]
        # div = ccImgOtsu[wrCorners[divWRI,1]:wrCorners[divWRI,3],wrCorners[divWRI,0]:wrCorners[divWRI,2]]
        # wrMask = (div == (divWRI+1))
        wrMask = (ccImgOtsu == (divWRI + 1))
        # temp = np.bitwise_and(wrMaskTemp[:,:,0],wrMaskTemp[:,:,1])
        # wrMask = np.bitwise_and(temp,wrMaskTemp[:,:,2])
        wrImg = np.zeros((imgOtsu.shape[0],imgOtsu.shape[1]),np.uint8)
        # div = wrImg[wrCorners[divWRI, 1]:wrCorners[divWRI, 3], wrCorners[divWRI, 0]:wrCorners[divWRI, 2]]
        # div[wrMask]=255
        wrImg[wrMask]=255
        for divWRNum in range(0,len(dividedWRs)):
            ccMask = np.copy(wrMask)
            roiMask = np.zeros(wrMask.shape,bool)
            roiMask[dividedWRs[divWRNum,1]:dividedWRs[divWRNum,3],dividedWRs[divWRNum,0]:dividedWRs[divWRNum,2]] = True
            ccMask = np.bitwise_and(ccMask,roiMask)

            #find bounding box
            xs = np.argmax(ccMask,axis=1)
            xs[xs==0]=99999
            ys = np.argmax(ccMask,axis=0)
            ys[ys==0]=99999
            y1= min(ys)
            x1= min(xs)

            #get y2
            ccMaskTemp = ccMask[::-1,:]
            ys = np.argmax(ccMaskTemp,axis=0)
            ys = ccMaskTemp.shape[0]-np.argmax(ccMaskTemp,axis=0)-1
            ys[ys==(ccMaskTemp.shape[0]-1)]=0
            y2=max(ys)

            #get x2
            ccMaskTemp = ccMask[:,::-1]
            xs = np.argmax(ccMaskTemp,axis=1)
            xs = ccMaskTemp.shape[1]-np.argmax(ccMaskTemp,axis=1)-1
            xs[xs==(ccMaskTemp.shape[1]-1)]=0
            x2=max(xs)
            if not((y2-y1)<0 or (x2-x1)<0):
                newWRCorners.append([x1,y1,x2,y2])
                newWRParents.append(divWRI)
            else:
                print("encountered weird")
    wrParents = np.arange(wrNum)
    if len(newWRParents)>0:#todo: update 3and osama law mesh 3ando el 7eta de
        wrCorners = np.delete(wrCorners,occIdx,axis=0)
        wrParents = np.delete(wrParents,occIdx,axis=0)
        wrCorners = np.vstack([wrCorners, np.asarray(newWRCorners)])
        wrParents = np.hstack([wrParents, np.asarray(newWRParents)])
        wrNum = wrCorners.shape[0]

    for wrI in range(0,wrNum):#loop on WRs
        roi = ccImgOtsu[wrCorners[wrI, 1]:wrCorners[wrI, 3], wrCorners[wrI, 0]:wrCorners[wrI, 2]]
        roi[roi[:, :] == (wrParents[wrI]+1)] = wrI + 1
    # for illustraion purposes
    imgWRLayerNoOCC = np.zeros(imgOriginal.shape).astype(np.uint8)
    imgWRLayerNoOCC[:, :] = randomColors[ccImgOtsu]


    imgWRLayerNoOCCWithRect = np.copy(imgWRLayerNoOCC)

    wrs = []
    for wrI in range(0,wrNum):#loop on WRs
        cv2.rectangle(imgWRLayerNoOCCWithRect, (wrCorners[wrI,0],wrCorners[wrI,1]), (wrCorners[wrI,2], wrCorners[wrI,3]), randomColors[wrI+1].tolist(), 7)
        roi = np.zeros((wrCorners[wrI, 3] - wrCorners[wrI, 1], wrCorners[wrI, 2] - wrCorners[wrI, 0]), np.uint8)
        roi[ccImgOtsu[wrCorners[wrI,1]:wrCorners[wrI,3],wrCorners[wrI,0]:wrCorners[wrI,2]]==(wrI+1)]=255
        wrs.append(roi)


    # r = 500.0 / imgWRLayerNoOCCWithRect.shape[1]
    # dim = (500, int(imgWRLayerNoOCCWithRect.shape[0] * r))
    # illustratinImg1 = cv2.resize(imgWRLayerNoOCCWithRect, dim, interpolation=cv2.INTER_AREA)

    # r = 500.0 / imgLinesLayout.shape[1]
    # dim = (500, int(imgLinesLayout.shape[0] * r))
    # illustratinImg2 = cv2.resize(imgLinesLayout, dim, interpolation=cv2.INTER_AREA)

    # cv2.imshow('WRs',illustratinImg1)
    # cv2.imshow('WRs2', illustratinImg2)
    # keyPress = cv2.waitKey(0)
    """
    Upkey : 2490368
    DownKey : 2621440
    LeftKey : 2424832
    RightKey: 2555904
    Space : 32
    Delete : 3014656
    """
    # if keyPress==32 :
    #     return wrs
    # gStack = np.zeros((gaussianImg.shape[0],gaussianImg.shape[1],3),np.uint8)
    # gStack[:,:,0]=gStack[:,:,1]=gStack[:,:,2]=gaussianImg
    # oStack = np.zeros((imgOtsu.shape[0],imgOtsu.shape[1],3),np.uint8)
    # oStack[:,:,0]=oStack[:,:,1]=oStack[:,:,2]=imgOtsu
    # resultString = "bin  (2)CCs  (3)gaus  (4)bin  (5)SWRs  (6)lines (6)WRs  (7)OCCs"
    # processImg = np.hstack([oStack,imgCCLayeredWithRectOtsu,gStack,imgCCLayeredFilteredBinarized,imgSWRLayerWithRect,imgLinesLayout,imgWRLayerWithRect,imgWRLayerNoOCCWithRect])
    # showImages([rawImg], ["raw"])
    # showImages([processImg],[resultString])
    # if not cv2.imwrite('./formsE-H errors/'+imgName, rawImg):
    #     print("failed to save "+imgName)
    # else:
    #     print("saved " + imgName)
    # return wrs


#hyper parameters:
kernelSize = 61
swrDistThreshCoff = 0.3 #multiplied by height
occCoff = 2


import glob
# import os
#
# dataPath = os.path.join(img_dir,"*g")
# files = glob.glob(dataPath)
# for f in files:
#     img = cv2.imread(f)

from datetime import datetime


#do something

#Python 3:


startTimeAll = datetime.now()
for filename in glob.glob('../iam/DBs/iamDB/data/my forms/formsE-H/*.png'):
    # imgName = filename[filename.find('\\')+1:]
    # startTime = datetime.now()
    print(filename[filename.find("\\")+1:])
    # wrs = segment(filename, 47, 0.3, 2.5)
    wrs=[]
    # if(filename[filename.find("\\")+1:]=="e07-012.png"):#e02-025.png"):

    wrs = segment(filename,47,0.3,2.5)#wrs = segment(filename,51,0.3,2.5)
    # print(datetime.now() - startTime)
    # x = input("w")
    # for wr in wrs:
    #     cv2.imshow("wr",wr)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow("wr")
    # cv2.destroyAllWindows()
print("all:",(datetime.now() - startTimeAll))