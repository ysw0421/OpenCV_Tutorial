import numpy as np
import cv2

## Step 1: Detect the keypoints using SIFT/SURF Detector
img = cv2.imread('sample3.jpg')
height, width = img.shape[:2]
print(height,width)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2, img3 = None, None
# - Sift Feature Detector -- Hessian
# SURF객체를 만든다.
surf = cv2.xfeatures2d.SURF_create()
# 그림을 표현하기위해 헤시안 값을 10000으로 놓는다. 실제로는 300 - 500 좋음
surf.setHessianThreshold(400)
kp1, des1 = surf.detectAndCompute(img, None)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    ## Step 2: Calculate descriptors (feature vectors)
    # 키포인트와 디스크립터를 직접 찾는다.
    kp2, des2 = surf.detectAndCompute(frame, None)
    ## Step 3: Matching descriptor vectors with a brute force matcher
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2,crossCheck=True)
    clusters = np.array([des1])
    bf.add(clusters)
    # Train: Does nothing for BruteForceMatcher though.
    bf.train()
    # Match descriptors.
    matches = bf.match(des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(frame, kp2, img, kp1, matches[:10], None,flags=2)
    # - Show detected keypoints
    cv2.imshow('SURF1', img3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


