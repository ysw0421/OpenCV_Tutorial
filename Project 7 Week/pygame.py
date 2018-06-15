import pygame, sys, math
from scripts.UltraColor import  *

class Cam:
    def __init__(self,pos=(0,0,0),rot=(0,0)):
        self.pos = list(pos)
        self.rot = list(rot)

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
verts = (0,0,0),(2,0,0),(2,2,0),(0,2,0),(0,0,2),(2,0,2),(2,2,2),(0,2,2)
edges = (0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)

faces = (0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(0,3,7,4),(1,2,6,5)
colors = (255,0,0), (255,128,0),(255,255,0),(255,255,255),(0,0,255),(0,255,0)
cam = Cam((0,0,-14))

while True:
    dt = clock.tick()/1000

    for event in pygame.event.get():
        if event.type == pygame.QUIT:pygame.quit();sys.exit()
    screen.fill((Color.Black))

    vert_list = []; screen_coords = []
    for x,y,z in verts:
        x -= cam.pos[0];y -= cam.pos[1];z -= cam.pos[2]
        vert_list += [[x,y,z]]

        f = 200 / z
        x, y = x * f, y * f
        screen_coords += [(cx + int(x), cy + int(y))]

    face_list = []; face_color = [] ; depth = []
    for f in range(len(faces)):
        face = faces[f]
        on_screen = False
        for i in face:
            x,y = screen_coords[i]
            if vert_list[i][2]>0 and x>0 and x<w and y>0 and y<h: on_screen = True; break

        if on_screen:
            coords = [screen_coords[i] for i in face]
            face_list += [coords]
            face_color += [colors[f]]
            depth += [sum(sum(vert_list[j][i] for j in face)**2 for i in range(3))]

    order = sorted(range(len(face_list)), key=lambda i: depth[i], reverse=1)

    for i in order:
        try: pygame.draw.polygon(screen,face_color[i],face_list[i])
        except: pass


    pygame.display.flip()
    key = pygame.key.get_pressed()
    cam.update(dt,key)