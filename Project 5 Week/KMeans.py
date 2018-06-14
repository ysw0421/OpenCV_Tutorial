import math
import numpy as np
import cv2
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist


if (__name__ == "__main__"):
    cap = cv2.VideoCapture('dfd1.mp4')
    mog = cv2.createBackgroundSubtractorMOG2(detectShadows=0)
    count = 0

    #list = ['video' + str(n) for n in range(100)]
    while True:
        list = []
        ret, frame = cap.read()
        ret1, frame1 = cap.read()
        fgmask = mog.apply(frame)
        mask = np.zeros_like(frame1)
        mask1 = np.zeros_like(frame1)


        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        dilation = cv2.dilate(closing, kernel, iterations=1)

        canny = cv2.Canny(dilation, 100, 200)
        cnts, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.rectangle(frame, (220, 100), (550, 160), (0, 255, 0), 2)

        cv2.imshow('mask', fgmask)
        cv2.imshow('mask3', dilation)
        cv2.imshow('mask15', canny)
        cv2.imshow('mask4', frame)
        cv2.imshow('mask8', frame[100:160, 220:550])

        for i in range(len(contours)):
            point = []
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame1, (int(x+w/2), int(y+h/2)), (int(x+w/2), int(y+h/2)), (255, 0, 0), 3)
            X = int(x+w/2)
            Y = int(y+h/2)
            distance = math.sqrt(X^2+Y^2)
            mask[y:y + h, x:x + w] = frame1[y:y + h, x:x + w]

            #(0,0)에서 좌표 거리 계산 후 리스트에 첨가
            point.append(distance)
            point.append(X)
            point.append(Y)
            list.append(point)

            #같은 좌표 값 제거
            if count == 0:
                print("List has one List")
            elif list[count][1] == list[count-1][1] and list[count][2] == list[count-1][2] :
                a = list.pop()
                count = count - 1
            count = count + 1
        count = 0

        #(0,0)에서 부터의 거리 오름차순 정리
        if not list:
            print("empty")
        else:
            list.sort()
            print(list)
            '''
            for i in range(len(list)):
                if count == 0:
                    print("list 내용 한개")
                else:
                    #오름차순 정리된 점 거리 계산
                    distance1 = math.sqrt((list[count][1] - list[count-1][1]) ** 2 + (list[count][2] - list[count-1][2]) ** 2)
                    print(count)
                    print(list[count][1],list[count][2])
                    print(list[count-1][1],list[count-1][2])
                    print("거리 ",distance1)
                count = count + 1
            count = 0
            '''
        cv2.imshow('mask2', frame1)


        print('                                                                    장면 전환')
        cv2.imshow('mask7', mask)



        k = cv2.waitKey(300) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()