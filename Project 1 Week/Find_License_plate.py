import numpy as np
import cv2
from matplotlib import pyplot as plt
#px = img[100,100,3] 행렬 100, 열 100 위치 채널 ?? 3
#이미지 속성에 액세스 하기
Number = 'find4.jpg'
img = cv2.imread(Number, cv2.IMREAD_COLOR)
copy_img = img.copy()
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img2, (3, 3), 0)
canny = cv2.Canny(blur, 100, 200)
cv2.imshow('show1',canny)
cnts, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

box1 = []
f_count = 0
select = 0
plate_width = 0

for i in range(len(contours)):
    cnt = contours[i]
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h  # area size
    aspect_ratio = float(w) / h  # ratio = width/height

    if (aspect_ratio >= 0.2) and (aspect_ratio <= 1.0) and (rect_area >= 100) and (rect_area <= 700):
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        box1.append(cv2.boundingRect(cnt))
        print(box1)

        # to find number plate measureing length between rectangles
        for m in range(len(box1)):
            count = 0
            for n in range(m + 1, (len(box1) - 1)):
                delta_x = abs(box1[n + 1][0] - box1[m][0])
                if delta_x > 150:
                    break
                delta_y = abs(box1[n + 1][1] - box1[m][1])
                if delta_x == 0:
                    delta_x = 1
                if delta_y == 0:
                    delta_y = 1
                gradient = float(delta_y) / float(delta_x)
                if gradient < 0.25:
                    count = count + 1
            # measure number plate size
            if count > f_count:
                select = m
                f_count = count;
cv2.imshow('show',img)

number_plate = copy_img[box1[select][1] - 10:box1[select][3] + box1[select][1] + 20,
               box1[select][0] - 90:140 + box1[select][0]]
resize_plate = cv2.resize(number_plate, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC + cv2.INTER_LINEAR)
plate_gray = cv2.cvtColor(resize_plate, cv2.COLOR_BGR2GRAY)
ret, th_plate = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)

cv2.imshow('plate_th.jpg',th_plate)
kernel = np.ones((3, 3), np.uint8)
er_plate = cv2.erode(th_plate, kernel, iterations=1)
er_invplate = er_plate
cv2.imwrite('er_plate.jpg', er_invplate)
#print(pytesseract.image_to_string(Image.open('er_plate.jpg'), lang='eng'))


cv2.waitKey(0)
cv2.destroyAllWindows()
