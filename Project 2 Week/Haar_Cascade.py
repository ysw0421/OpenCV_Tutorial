import numpy as np
import cv2
cap = cv2.VideoCapture('cars.avi')
car_cascade = cv2.CascadeClassifier('cascade.xml')
mog = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    ret, frame1 = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    frame1 = cv2.resize(frame1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    imgray = cv2.Canny(imgray, 100, 200, 3)

    ret,thresh = cv2.threshold(imgray,200,255,cv2.THRESH_BINARY_INV)

    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgray, contours, 1, (0,255,0))

    #cv2.imshow('result.png', imgray)
    fgmask = mog.apply(imgray)
    #cv2.imshow('result.png2', fgmask)


    cnts, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        rect_area = w * h  # area size
        aspect_ratio = float(w) / h  # ratio = width/height

        if (aspect_ratio >= 1) and (aspect_ratio <= 2) and (rect_area >= 300) and (rect_area <= 8000):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('mask2', frame)
        else:
            cv2.imshow('mask2', frame)


    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    cv2.imshow('show2', gray)
    cars = car_cascade.detectMultiScale(gray, 1.3, 2)
    for (x,y,w,h) in cars:
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        canny = cv2.Canny(frame1, 100, 200)
        cv2.imshow('show1', frame1)
    cv2.imshow('show1', frame1)


    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()