import numpy as np
import cv2
from random import shuffle
import math

def nothing(x):
    pass

mode, drawing = True, False
ix, iy = -1, -1
B = [i for i in range(256)]
G = [i for i in range(256)]
R = [i for i in range(256)]
cv2.namedWindow('paint')
cv2.createTrackbar('choose filter','paint', 0, 3,nothing)
def onMouse(event, x, y, flags, param):
    global ix, iy, drawing, mode, B, G, R,w,h, mode1
    img = np.zeros(param.shape, np.uint8)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        shuffle(B), shuffle(G), shuffle(R)
        mode1 = cv2.getTrackbarPos('choose filter', 'paint')
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv2.rectangle(img, (ix,iy),(x,y),(B[0], G[0], R[0]),1)
                cv2.imshow('paint',img)
                cat =cv2.addWeighted(param, 1, img, 1,0)
                cv2.imshow('paint', cat)
            else:
                r = (ix-x)**2 + (iy-y)**2
                r = int(math.sqrt(r))
                cv2.circle(img, (ix,iy),r,(B[0], G[0], R[0]),1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        w = abs(-ix+x)
        h = abs(-iy+y)
        if w%2==0 or h%2==0:
            if w%2==0 and w>3 and h>3:
                w = w - 1
            if h%2==0 and w>3 and h>3:
                h = h -1
            if w != 0 and h != 0:
                if ix>x:
                    change = ix
                    ix = x
                    x = change
                if iy>y:
                    change = iy
                    ix = y
                    y = change
                print("(", ix, iy, ")", "(", x, y, ")", w, h)
        if mode:
            if mode1==1:
                print("블러링입니다. 이미지를 smooth하게 만드는 효과가 있습니다.")
                kernel = np.ones((3, 3), np.float32) / 9
                print(kernel)
                part=cv2.filter2D(param[iy:y,ix:x], -1, kernel)
                param[iy:y,ix:x]=part
                cv2.imshow('paint', param)
            elif mode1==2:
                print("가우시안 블러링입니다.")
                part = cv2.GaussianBlur(param[iy:y, ix:x],(3,3),0)
                param[iy:y, ix:x] = part
                cv2.imshow('paint', param)
            elif mode1==3:
                print("미디언 필터입니다.")
                part = cv2.medianBlur(param[iy:y, ix:x], 3)
                param[iy:y, ix:x] = part
                cv2.imshow('paint', param)
        else:
            r = (ix - x) ** 2 + (iy - y) ** 2
            r = int(math.sqrt(r))
            cv2.circle(param, (ix, iy), r, (B[0], G[0], R[0]), -1)
def mouseBrush():
    global mode
    cat = cv2.imread('cat.jpg')
    cv2.setMouseCallback('paint',onMouse,param= cat)
    while True:
        cv2.imshow('paint',cat)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('m'):
            mode = not mode
    cv2.destroyAllWindows()
mouseBrush()