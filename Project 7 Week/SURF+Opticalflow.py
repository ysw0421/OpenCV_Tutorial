import numpy as np
import cv2
'''
optical flow를 사용하는 것
1. 움직임을 통한 구조 분석
2. 비디오 압축
3. Video Stabilization: 영상이 흔들렸거나 블러가 된 경우 깨끗한 영상으로 처리하는 기술
'''
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict( maxCorners = 200, qualityLevel = 0.3, minDistance = 7, blockSize = 7)
lk_params = dict( winSize  = (15,15), maxLevel = 2,criteria = termination)

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.blackscreen = False
        self.width = int(self.cam.get(3))
        self.height = int(self.cam.get(4))
        self.kp1 = []; self.kp2 = []
        self.des1 = []; self.des2 = []

    def find_KeyPoint(self,frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame 크기의 하얀색 바탕 제작
        mask = np.zeros_like(frame_gray)
        mask[:] = 255
        # tr[-1] 최신 특이점 점들, 하얀색 바탕위에 특이점 찍음.
        for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)
        # SURF_create 안에 헤시안 한계 값 넣어라.
        surf = cv2.xfeatures2d.SURF_create()
        # 현재 헤시안 한계 값 확인(300~500)사이가 좋음.
        surf.setHessianThreshold(400)
        # 키포인트와 디스크립터를 직접 찾는다.
        kp, des = surf.detectAndCompute(frame, mask=mask)
        # key point의 좌표
        for keyPoint in kp:
            x = np.float32(keyPoint.pt[0])
            y = np.float32(keyPoint.pt[1])
            # 좌표들을 tracks에서 관리
            self.tracks.append([(x, y)])
        return kp, des


    def run(self):
        while True:
            ret, frame = self.cam.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()


            # b 누르면 blackscreen bool값 1 or 0 으로 바뀐다.
            if self.blackscreen:
                if len(self.tracks) > 0:
                    # 이전이미지, 현재이미지
                    img0, img1 = self.prev_gray, frame_gray
                    # 들어간 최근 track
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    print(p0)
                    # lk_params = Parameters for lucas kanade optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                    # 점들의 차이가 1이하면 True 아니면 False
                    good = d < 1

                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        # 1이 아니면 무시한다.
                        if not good_flag:
                            continue
                        # tr에 x,y를 넣는다.
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            kp2, des2 = self.find_KeyPoint(frame)
                            self.kp2.append(kp2)
                            self.des2.append(des2)
                            self.blackscreen = not self.blackscreen
                            break
                            # 여기서 종료

                        cv2.line(vis, (tr[-1][0], tr[-1][1]), (tr[0][0], tr[0][1]), (255, 0, 0), 2)
                        cv2.circle(vis, (tr[0][0], tr[0][1]), 2, (0, 255, 0), -1)
                        cv2.circle(vis, (tr[-1][0], tr[-1][1]), 2, (255, 0, 0), -1)


                #처음 keypoint 설정
                if self.frame_idx == 0:
                    kp1, des1 = self.find_KeyPoint(frame)
                    self.kp1.append(kp1)
                    self.des1.append(des1)
                    vis = cv2.drawKeypoints(vis, kp1, vis, (0, 0, 255), 4)
                self.frame_idx += 1
                self.prev_gray = frame_gray


            cv2.imshow('frame',vis)
            k=cv2.waitKey(30) & 0xFF
            if k == ord('q'):
                break


            if k == ord('b'):
                #여기서 한번
                #이전의 값 초기화
                self.blackscreen = not self.blackscreen
                self.tracks = []
                self.frame_idx = 0
                self.kp1 = []; self.kp2 = []
                self.des1 = []; self.des2 = []
        self.cam.release()


video_src = 0
App(video_src).run()
cv2.destroyAllWindows()