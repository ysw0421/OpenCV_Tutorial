import numpy as np
import cv2
from matplotlib import pyplot as plt
img1 = cv2.imread('sample.jpg')          # trainImage
imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

surf = cv2.xfeatures2d.SURF_create()
# 그림을 표현하기위해 헤시안 값을 10000으로 놓는다. 실제로는 300 - 500 좋음
surf.setHessianThreshold(10000)
kp1, des1 = surf.detectAndCompute(imgray,None)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # find the keypoints and descriptors with SURF
    frgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = surf.detectAndCompute(frame, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    clusters = np.array([des1])
    flann.add(clusters)
    # Train: Does nothing for BruteForceMatcher though.
    flann.train()
    # Match descriptors.
    matches = flann.knnMatch(des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), matchesMask=matchesMask, flags=0)
    img3 = cv2.drawMatchesKnn(img1, kp1, frame, kp2, matches, None, **draw_params)

    cv2.imshow('image', img3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
