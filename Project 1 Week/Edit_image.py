import cv2 as cv
import numpy as np
img = cv.imread('test6.jpg')  # 960 X 540 복도 사진
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,100,150,apertureSize = 3)
lines = cv.HoughLinesP(edges,1,np.pi/180,100,minLineLength=50,maxLineGap=10)
'''
#-----------------------------------------------
cv.circle(img, (390,335), 5, (0,255,0),-1)
cv.circle(img, (565,335), 5, (0,0,255),-1)
#-----------------------------------------------
cv.circle(img, (240,535), 5, (255,0,0),-1)
cv.circle(img, (730,535), 5, (255,255,255),-1)
cv.circle(img, (240,45), 5, (255,0,255),-1)
cv.circle(img, (730,45), 5, (255,0,255),-1)
#-----------------------------------------------
'''
cv.rectangle(img,(242,50),(732,540),(0,255,0),1)
cv.rectangle(img,(385,155),(570,340),(0,255,0),1)

#=====================================================
chess_black= cv.imread('test7.png') # 240 X 240 체스판 사진.
cv.imshow('aa',chess_black)
chess_black = cv.resize(chess_black,None,fx=490/500, fy=490/500, interpolation = cv.INTER_CUBIC)
chess_white = cv.bitwise_not(chess_black)

rows1, cols1, ch1 = chess_black.shape
rows2, cols2, ch2 = chess_white.shape
pts1 = np.float32([[0,0],[490,0],[0,490],[490,490]])
pts2 = np.float32([[143,290],[328,290],[0,490],[490,490]])

M = cv.getPerspectiveTransform(pts1, pts2)
dst1 = cv.warpPerspective(chess_black, M, (cols1,rows1))
dst2 = cv.warpPerspective(chess_white, M, (cols2,rows2))

# 배경이미지 img
# 넣어야할 이미지 img1
#========================체스판 (흰색) 부분========================
rows, cols, channels = dst1.shape
roi = img[50:540,242:732]
gray = cv.cvtColor(dst1,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(gray,10,255,cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
img_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
img1_fg = cv.bitwise_and(dst1, dst1, mask=mask)
dstA = cv.add(img_bg,img1_fg)
img[50:540,242:732]= dstA
#========================체스판 (검은색) 부분========================
rows, cols, channels = dst2.shape
#roi = img[45:535,240:730]
gray = cv.cvtColor(dst2,cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(gray,10,255,cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
img_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
img1_fg = cv.bitwise_and(dst1, dst2, mask=mask)
dstB = cv.add(img_bg,img1_fg)
img[50:540,242:732]= dstB



'''
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv.line(img,(x1,y1),(x2,y2),(0,255,0),2)
'''
cv.circle(img, (477,340), 2, (0,0,255),-1)
cv.line(img,(477,340),(485,540),(0,0,255),2)

cv.line(img,(546,309),(729,536),(255,0,0),2)
cv.line(img,(243,537),(392,332),(255,0,0),2)

cv.line(img,(565,332),(392,332),(0,0,255),2)

x=((729-546)/(536-309))*(332-309)+546
print(x)
cv.circle(img, (565,332), 2, (0,0,255),-1)
cv.imshow('test3.jpg',img)
#cv.imwrite('test_space.jpg',img)

cv.waitKey(0)
cv.destroyAllWindows()