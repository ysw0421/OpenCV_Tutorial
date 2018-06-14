import numpy as np
import cv2
from matplotlib import pyplot as plt


if (__name__ == "__main__"):
    cap = cv2.VideoCapture('dfd1.mp4')
    mog = cv2.createBackgroundSubtractorMOG2(detectShadows=0)

    #list = ['video' + str(n) for n in range(100)]
    list_1 = []
    point = []
    i1 = 0
    count = 1
    queue = []

    while True:
        ret, frame = cap.read()
        ret1, frame1 = cap.read()
        fgmask = mog.apply(frame)

        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        dilation = cv2.dilate(closing, kernel, iterations=1)

        canny = cv2.Canny(dilation, 100, 200)
        cnts, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.rectangle(frame, (220, 100), (550, 160), (0, 255, 0), 2)

        cv2.imshow('mask', fgmask)
        cv2.imshow('mask3', dilation)
        cv2.imshow('mask4', frame)
        cv2.imshow('mask8', frame[100:160, 220:550])

        for i in range(len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            rect_area = w * h  # area size
            aspect_ratio = float(w) / h  # ratio = width/height




            if (aspect_ratio >= 1) and (aspect_ratio <= 2) and (rect_area >= 3000) and (rect_area <= 8000) \
                    and (x + w / 2 > 220) and (x + w / 2 <= 550) and (y > 100) and (y <= 160):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



                if i1 == 0:
                    list_1.append(frame[y:y + h, x:x + w])
                    i1 = i1 + 1
                    cv2.imwrite("video" + str(count) + ".jpg", frame1[y-10:y + h + 10, x - 10:x + w + 10])
                    cv2.imshow('mask5', frame1[y:y + h, x:x + w])
                    cv2.imwrite("video" + str(count+10) + ".jpg", frame1)
                    '''
                    gray = cv2.cvtColor(frame1[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
                    corners = np.int0(corners)
                    for i in corners:
                        x1, y1 = i.ravel()
                        cv2.circle(frame1, (x1+x, y1+y), 3, 255, -1)
                        cv2.imshow('mask4', frame1)
                        cv2.imwrite("video" + str(count+10) + ".jpg", frame1)
                    '''


                else:
                    hist1 = cv2.calcHist([list_1[0]], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([frame[y:y + h, x:x + w]], [0], None, [256], [0, 256])
                    retval = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
                    if (retval > 0.28):
                        count = count + 1
                        cv2.imwrite("video" + str(count) + ".jpg", frame1[y - 10:y + h + 10, x - 10:x + w +10])
                        cv2.imshow('mask5', frame1[y:y + h, x:x + w])
                        cv2.imwrite("video" + str(count+10) + ".jpg", frame1)
                        list_1.pop(0)
                        list_1.append(frame[y:y + h, x:x + w])
                        '''
                        gray = cv2.cvtColor(frame1[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                        corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
                        corners = np.int0(corners)
                        for i in corners:
                            x1, y1 = i.ravel()
                            cv2.circle(frame1, (x1+x , y1+y), 3, 255, -1)
                            cv2.imshow('mask4', frame1)
                        '''

        k = cv2.waitKey(300) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()