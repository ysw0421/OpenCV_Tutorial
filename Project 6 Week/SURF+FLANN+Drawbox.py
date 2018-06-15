import numpy as np
import cv2

MIN_MATCH_COUNT = 10
img1 = cv2.imread('sample3.jpg')          # queryImage
height, width = img1.shape[:2]
print(height,width)

imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
surf = cv2.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SUFT
kp1, des1 = surf.detectAndCompute(imgray,None)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    kp2, des2 = surf.detectAndCompute(frame,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    if des2 is None:
        continue
    else:
        matches = flann.knnMatch(des1,des2,k=2)


    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        if M is None:
            continue
        else:
            dst = cv2.perspectiveTransform(pts,M)
            frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None


    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(img1,kp1,frame,kp2,good,None,**draw_params)
    cv2.imshow('image', img3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
