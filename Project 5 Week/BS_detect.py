import numpy as np
import cv2



capture = cv2.VideoCapture('detect_test.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(500,1600,False)

while(capture.isOpened()):

    ret, frame = capture.read()

    # 영상 특성 변경 전처리 작업
    Fgmask = fgbg.apply(frame)
    Fgmask = fgbg.apply(frame)
    # 픽셀 합치기를 통한 잡음제거
    pixeling = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(Fgmask, cv2.MORPH_OPEN, pixeling)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, pixeling)
    AfterPixeling = cv2.dilate(closing, pixeling, iterations=1)
    cv2.imshow('CORE1', Fgmask)


    # 화면 출력
    canny = cv2.Canny(AfterPixeling, 100, 200)
    cnts, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        AfterPixeling[y - 12:y + h + 12, x - 12:x + w + 12] = 255

    canny = cv2.Canny(AfterPixeling, 100, 200)
    cnts, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        AfterPixeling[y -12:y + h+12, x-12:x + w+12] = 255
        if(w*h<500):
            print(x,y,w,h)
            if(y<=8):
                AfterPixeling[0:y + h + 8, x - 8:x + w + 8] = 255
            else:
                AfterPixeling[y-8:y + h + 8, x - 8:x + w + 8] = 255

    canny = cv2.Canny(AfterPixeling, 100, 200)
    cnts, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #X = np.array(result)

    compare =[]
    print('-------------')
    for i in range(len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        AfterPixeling[y:y + h, x:x + w] = 255
        if ((w) * (h) >= 500):
            compare.append([x, y, w, h])
            if len(compare)!=1:
                if compare[len(compare)-1] == compare[len(compare)-2]:
                    compare.pop()

    compare.sort()
    print(compare)
    for i in range(len(compare)):
        cv2.rectangle(frame, (compare[i][0], compare[i][1]),
                      (compare[i][0] + compare[i][2], compare[i][1] + compare[i][3]) , (0, 0, 255), 2)


    cv2.imshow('CORE', frame)
    cv2.imshow('AP', AfterPixeling)



    #while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()