import xlsxwriter
import numpy as np
import cv2
import os
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler

##PATH = "dices"
##EXT = ".*"
##shapeDB = (50,50)
PATH = "img_time_real"
EXT = ".*"
shapeDB = (90,90)

BLUR = 3

def getData(PATH):
    imgPaths = []
    labels = []

    for root, dirs, files in os.walk(PATH):
        path = root.split(os.sep)
        # print((len(path) - 1) * '>>>'+ os.path.basename(root))
        label = os.path.basename(root)
        for i,file in enumerate(files):
            # print((len(path)-1) * '\t' + label + '\t' + file + '\t' + str(path))
            # if (file.endswith(EXT)):
            filePath = root + "/" + file
            imgPaths.append(filePath)
            try:
                labels.append(int(label))
            except:
                labels.append(0)
            #break
    return imgPaths , labels

if __name__ == '__main__':

    imgPath , labels = getData(PATH)

    maxImg = len(imgPath)
    #maxImg = 1
    for i in range(maxImg):
        img = cv2.imread(imgPath[i], 1)
        img = cv2.resize(img,shapeDB,interpolation = cv2.INTER_NEAREST)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray,BLUR)
        ret,edges = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        canny = cv2.Canny(gray,50,150,apertureSize = 3)

        contours,hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cnt = max(contours, key = cv2.contourArea)

        x,y,w,h = cv2.boundingRect(cnt)
        
        A = cv2.contourArea(cnt)
        p = cv2.arcLength(cnt,True)
        RA = w/float(h)
        Comp = A/float(p*p)
        R = A/float(w*h)
        M = cv2.moments(cnt)
        Hu = cv2.HuMoments(M)
        upperCount = cv2.countNonZero(edges[0:shapeDB[1]/2 , :])
        lowerCount = cv2.countNonZero(edges[shapeDB[1]/2:shapeDB[1] , :])
        leftCount = cv2.countNonZero(edges[: , 0:shapeDB[0]/2])
        rightCount = cv2.countNonZero(edges[: , shapeDB[0]/2:shapeDB[0]])
        
        circles = cv2.HoughCircles(gray,cv2.cv.CV_HOUGH_GRADIENT,1,20,
                            param1=30,param2=20,minRadius=0,maxRadius=0)

        try:
            circlesCount = len(circles[0])
            meanCR = np.mean(circles[0,:,2])
        except:
            circlesCount = 0
            meanCR = 0
        
        meanB = np.sum(img[:,:,0])/cv2.countNonZero(edges[:,:])
        meanG = np.sum(img[:,:,1])/cv2.countNonZero(edges[:,:])
        meanR = np.sum(img[:,:,2])/cv2.countNonZero(edges[:,:])
        
        featuresArray = [A , upperCount, lowerCount, leftCount,
                         rightCount, p, Comp, RA, R, Hu[0][0], Hu[1][0], Hu[2][0],
                         Hu[3][0], Hu[4][0], Hu[5][0], Hu[6][0],
                         len(contours), meanB, meanG, meanR, circlesCount, meanCR]

        featuresArray1 = np.array(featuresArray)
        
        sample = featuresArray1.reshape(1,-1)
        
        mlp = joblib.load('Modelos_Predict/mlp.pkl')
        svm = joblib.load('Modelos_Predict/svm.pkl')
        knn = joblib.load('Modelos_Predict/knn.pkl')
        standard_scaler = joblib.load('Modelos_Predict/mlpScaler.pkl')

        X = standard_scaler.transform(sample)

        predMLP = mlp.predict(X)
        predSVM = svm.predict(X)
        predKNN = knn.predict(X)

        print 'MLP:\t', int(predMLP[0])
        print 'SVM:\t', int(predSVM[0])
        print 'KNN:\t', int(predKNN[0])
        print '======================='
        cv2.imshow('edges',edges)
        cv2.waitKey(0)

##    cv2.imshow('img',img)
##    cv2.imshow('gray',gray)
##    cv2.imshow('canny',canny)
##    cv2.imshow('edges',edges)
##    cv2.waitKey(0)
    cv2.destroyAllWindows()
