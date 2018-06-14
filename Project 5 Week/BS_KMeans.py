from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

import math
import numpy as np
import cv2
from matplotlib import pyplot as plt


capture = cv2.VideoCapture('dfd1.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(500,1600,False)

while(capture.isOpened()):

    ret, frame = capture.read()
    ret, frame = capture.read()

    # 영상 특성 변경 전처리 작업
    Fgmask = fgbg.apply(frame)
    Fgmask = fgbg.apply(frame)
    # 픽셀 합치기를 통한 잡음제거
    pixeling = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(Fgmask, cv2.MORPH_OPEN, pixeling)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, pixeling)
    AfterPixeling = cv2.dilate(closing, pixeling, iterations=1)


    # 화면 출력
    cv2.imshow('CORE', frame)
    cv2.imshow('AP', AfterPixeling)
    result = []

    y=0
    while y<540:
        x = 0
        while x<960:
            if AfterPixeling[y][x] == 255:
                result.append([x,y])
            x=x+3
        y=y+3

    #print(result)
    X = np.array(result)
    #print(X)
    #print(int(X.shape[0]))
    distortions = []
    if(X!=[]):

        for k in range(1,X.shape[0]):
            kmeanModel = KMeans(n_clusters=k).fit(X)
            distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
            print('---------------------------------------------------------------')
            print('k : ', k, '\n 클러스터링 중심값 : \n', kmeanModel.cluster_centers_)
            if (len(distortions) >= 2):
                test = int(distortions[k - 2]) / int(distortions[k - 1])
                print('test 값 :', test, 'distortions:', distortions)
                if (test == 1):
                    print(k)
                    print('========================================')
                    break
            if k > 20:
                break
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

capture.release()
cv2.destroyAllWindows()