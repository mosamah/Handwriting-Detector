import numpy as np
from  scipy.spatial.distance import cdist
import math

def SIFTDescriptorSignatureExtraction(SD,C,t):
    # SD : SIFT descriptors  which are extracted from an offline handwriting image I  (n,128) n : number of keypoints in the image
    # C :  SD codebook with size N.  where N = 300   C (N,128) 128 correspond to descriptors

    SDS = np.zeros(C.shape[0])
    ED = cdist(SD,C,metric="euclidean")   # [i][j] distance between keypoint i and Cj)
    idx = np.argpartition(ED, t,axis=1)
    for x in range(ED.shape[0]):   #segma of x = 1
        ED[x,idx[x,t:]] = 0
    SDS = SDS + np.sum(ED > 0,axis=0)
    SDS = SDS/np.sum(SDS)
    return  SDS  #test function

def scaleOrientationHistogramExtraction(scales,orientations,phi,octaves,subLevels):
    Z= octaves * subLevels
    Obin = math.ceil(360/phi) #intervals
    SOH = np.zeros(Z*Obin)

    bin = np.ceil(orientations/phi)
    idx =Obin + (scales-1)+1
    SOH[idx] = SOH[idx] + 1

    SOH = SOH/np.sum(SOH)
    return SOH


def featureMatching (SDSI,SDSF):#,SOHI,SOHF,w):
    #SDSI : SDS OF UNKNOWNN IMAGE
    #SDSF : SDS FROM FEATURE TEMPLATE
    #SOHI : SOH OF UNKOWN IMAGE
    #SOHF : SOH FROM FEATURE TEMPLATE
    #w : weight  hyperParameter Cross-Validation

    manhattanDistance = np.sum(np.abs(SDSI - SDSF),axis=1)
    # normalize manhattanDistance to range [0,1]
    manhattanDistance = (manhattanDistance - np.min(manhattanDistance))/(np.max(manhattanDistance)-np.min(manhattanDistance))

    #chiSquareDistance = np.sum((np.power(SOHI-SOHF,2))/(SOHI+SOHF),axis=1)
    # normalize chiSquareDistance to range [0,1]
    #chiSquareDistance = (chiSquareDistance-np.min(chiSquareDistance))/(np.max(chiSquareDistance)-np.min(chiSquareDistance))

    dissimilarity = manhattanDistance # * w + (1-w) * chiSquareDistance

    ### the one with min dissimilarity is the correct writer


test = np.array([2,9,4,6])

template = np.array([[2,9,4,6],
                    [5,7,8,9],
                    [8,2,3,5]])


SD = np.array([[2,9],
              [3,11]]).reshape(2,2)
C = np.array([[2,10],
             [3,4],
              [8,11]]).reshape(3,2)



SIFTDescriptorSignatureExtraction(SD,C,2)     # just to test the functions.

featureMatching(test,template)#,test,template,2) # just to test the functions
