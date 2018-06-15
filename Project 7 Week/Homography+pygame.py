import numpy as np
import pygame, math
from pygame.locals import *
import cv2
import numpy as np
import sys
from scripts.UltraColor import  *

# w = 500, h = 700
chessboard = [[0, 0], [0, 99], [0, 199], [0, 299],[0, 399],[0, 499],[0, 599],[0, 699],
              [99, 0], [99, 99], [99, 199], [99, 299],[99, 399],[99, 499],[99, 599],[99, 699],
              [199, 0], [199, 99], [199, 199], [199, 299],[199, 399],[199, 499],[199, 599],[199, 699],
              [299, 0], [299, 99], [299, 199], [299, 299],[299, 399],[299, 499],[299, 599],[299, 699],
              [399, 0], [399, 99], [399, 199], [399, 299],[399, 399],[399, 499],[399, 599],[399, 699],
              [499, 0], [499, 99], [499, 199], [499, 299],[499, 399],[499, 499],[499, 599],[499, 699]]

MIN_MATCH_COUNT = 10
img1 = cv2.imread('sample3.jpg')          # queryImage
imgray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# Initiate SIFT detector
surf = cv2.xfeatures2d.SURF_create()
# find the keypoints and descriptors with SUFT
kp1, des1 = surf.detectAndCompute(imgray,None)

def rotate2d(pos,rad): x, y=pos; s,c = math.sin(rad),math.cos(rad); return x*c-y*s, y*c+x*s

class Cam:
    def __init__(self,pos=(0,0,0),rot=(0,0)):
        self.pos = list(pos)
        self.rot = list(rot)

    def events(self,event):
        if event.type == pygame.MOUSEMOTION:
            x,y = event.rel;   x/=200;  y/=200
            self.rot[0] += y; self.rot[1] += x

    def update(self,dt,key):
        s = dt*10
        if key[pygame.K_q]: self.pos[1] += s
        if key[pygame.K_e]: self.pos[1] -= s

        x,y = s*math.sin(self.rot[1]), s*math.cos(self.rot[1])

        if key[pygame.K_w]: self.pos[0] += x ; self.pos[2] += y
        if key[pygame.K_s]: self.pos[0] -= x ; self.pos[2] -= y

        if key[pygame.K_a]: self.pos[0] -= y; self.pos[2] += x
        if key[pygame.K_d]: self.pos[0] += y; self.pos[2] -= x
pygame.init()
radian=0
w,h = 400,400; cx,cy = w//2, h//2
screen = pygame.display.set_mode((w,h))
clock = pygame.time.Clock()
verts = (-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)
edges = (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)

faces = (0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)
colors = (255,0,0), (255,128,0),(255,255,0),(255,255,255),(0,0,255),(0,255,0)
cam = Cam((-3,-3,-5))

pygame.event.get();pygame.mouse.get_rel()
pygame.mouse.set_visible(0);pygame.event.set_grab(1)
# s와 t의 값이 0과 1사이를 벗어나는 경우, 두 선은 교차하지 않는다고 판정
# s와 t를 구하는 공식에서 분모가 0인 경우 두 선은 평행한다는 의미 교점은 존재하지 않는다.
# 분모와 분자가 모두 0인 경우 두선은 동일한 선.
def Compare_Line(ax,ay,bx,by,cx,cy,dx,dy):
    # ax + t*(bx - ax);     cx + s*(dx - cx)
    # ay + t*(by - ay);     cy + s*(dy - cy)
    t = 0;  s = 0
    ts_PVector = (dy-cy)*(bx-ax) - (dx-cx)*(by-ay)
    if(ts_PVector==0): return False

    _t = (dx-cx)*(ay-cy) - (dy-cy)*(ax-cx)
    _s = (bx-ax)*(ay-cy) - (by-ay)*(ax-cx)
    t = _t/ts_PVector; s = _s/ts_PVector

    if t<0 or t>1 or s<0 or s>1: return False
    if _t == 0 and _s == 0: return False
    return True;

cap = cv2.VideoCapture(0)
pygame.display.set_caption("OpenCV camera stream on Pygame")
screen = pygame.display.set_mode([int(cap.get(3)),int(cap.get(4))])

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
        # w = 500, h = 700
        print(w,h)
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        pts2 = np.float32(chessboard).reshape(-1, 1, 2)

        if M is None:
            continue
        else:
            dst = cv2.perspectiveTransform(pts,M)
            dst2 = cv2.perspectiveTransform(pts2, M)
            a = dst[0][0];b = dst[1][0];c = dst[2][0];d = dst[3][0]
            if a.all==b.all or a.all==c.all or a.all==d.all or b.all==c.all or b.all==d.all or c.all==d.all:
                print("점 중첩")
            elif Compare_Line(a[0],a[1],b[0],b[1],c[0],c[1],d[0],d[1]) == True or \
                Compare_Line(a[0],a[1],d[0],d[1],b[0],b[1],c[0],c[1]) == True:
                # 뒤틀림 경우 제외
                print("뒤틀림")
            elif Compare_Line(a[0],a[1],c[0],c[1],b[0],b[1],d[0],d[1]) == False:
                print("엇갈림")
            else:
                frame = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                for i in range(0,8):
                    frame = cv2.line(frame, (np.int32(dst2[i][0][0]),np.int32(dst2[i][0][1])), (np.int32(dst2[i+40][0][0]),np.int32(dst2[i+40][0][1])), (0, 255, 0), 3)
                for i in range(0, 6):
                    frame = cv2.line(frame, (np.int32(dst2[8*i][0][0]), np.int32(dst2[8*i][0][1])),(np.int32(dst2[8*i+7][0][0]), np.int32(dst2[8*i+7][0][1])), (0, 255, 0), 3)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
    img3 = cv2.drawMatches(img1,kp1,frame,kp2,good,None,**draw_params)
    cv2.imshow('image', img3)

    # opengl ---------------------------------------
    dt = clock.tick() / 1000
    for event in pygame.event.get():
        if event.type == pygame.QUIT:pygame.quit();sys.exit()
        cam.events(event)
    screen.fill([0, 0, 0])

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = cv2.flip(frame, 0)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0),None)
    # 1,2,3,4,5,6,7,8,(X,Y,Z)

    vert_list = [];screen_coords = []
    for x,y,z in verts:
        x -= cam.pos[0];y -= cam.pos[1];z -= cam.pos[2]
        x, z = rotate2d((x, z), cam.rot[1])
        y, z = rotate2d((y, z), cam.rot[0])
        vert_list += [[x,y,z]]

        f = 200 / z
        x, y = x * f, y * f
        screen_coords += [(cx + int(x), cy + int(y))]
    face_list = []; face_color = []; depth = []
    for f in range(len(faces)):
        face = faces[f]
        on_screen = False
        for i in face:
            x, y = screen_coords[i]
            if vert_list[i][2] > 0 and x > 0 and x < w and y > 0 and y < h: on_screen = True; break

        if on_screen:
            coords = [screen_coords[i] for i in face]
            face_list += [coords]
            face_color += [colors[f]]
            depth += [sum(sum(vert_list[j][i] for j in face) ** 2 for i in range(3))]

    order = sorted(range(len(face_list)), key=lambda i: depth[i], reverse=1)
    for i in order:
        try: pygame.draw.polygon(screen,face_color[i],face_list[i])
        except: pass
    key = pygame.key.get_pressed()
    cam.update(dt, key)
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            sys.exit(0)
    pygame.display.update()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
