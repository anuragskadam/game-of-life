import numpy as np
from scipy.signal import convolve2d
import cv2
import os
import time
import datetime


NO_ITERATIONS = 500
DIMENSION_VAR = 1
FPS = 10
PERCENT_FILL = 20

DIMENSIONS = np.array([[50, 50],[100,100],[500, 500], [720, 1280]])[DIMENSION_VAR]
HEIGHT, WIDTH = DIMENSIONS[0], DIMENSIONS[1]


WORLD = np.zeros(DIMENSIONS, dtype=np.uint8)
TEMP_WORLD = np.zeros(DIMENSIONS, dtype=np.uint8)


SINGLE_BIRD = np.array([255, 255, 255], np.uint8)
SINGLE_SKY = np.array([0, 0, 0], np.uint8)

IMAGE = np.full((HEIGHT, WIDTH, 3), SINGLE_SKY)

KERNEL = np.array([[1,1,1],
                   [1,0,1],
                   [1,1,1]])

def random_world_maker():
    ini_arr = np.random.choice(np.product(DIMENSIONS),np.product(DIMENSIONS)*PERCENT_FILL//100, replace=0)
    WORLD[ini_arr//DIMENSIONS[1], ini_arr % DIMENSIONS[1]] = 1
    print(len(ini_arr)/np.product(DIMENSIONS))

random_world_maker()

def manual_maker():
    WORLD[[0,1,2,2,2,16,17,18],[1,2,0,1,2,17,17,17]] = 1
    
# manual_maker()



def next_val(self_val, adj_sum):

    if self_val == 0:
        return adj_sum == 3
    else:
        return adj_sum in [2,3]


def step():
    global WORLD
    adj_sum_arr = convolve2d(WORLD, KERNEL,mode='same', boundary='wrap').astype(np.uint8)       # can be made faster probably

    for n, i in enumerate(WORLD):
        for m, j in enumerate(i):
            TEMP_WORLD[n,m] = next_val(j,adj_sum_arr[n,m])
    
    WORLD = TEMP_WORLD


def main():
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    vid_no_var = 0
    while 1:
        file_name = f'generated_videos/life{vid_no_var}.mp4'
        if not os.path.exists(file_name):
            out = cv2.VideoWriter(file_name, fourcc, FPS, (WIDTH, HEIGHT))
            break
        vid_no_var += 1
    

    print(f"\nGenerating video life{vid_no_var}.mp4")

    start_time = time.time()
    no_decimals_perc = 1

    out.write( cv2.cvtColor(WORLD*255, cv2.COLOR_GRAY2BGR))

    for frame in range(NO_ITERATIONS):
        step()
        out.write( cv2.cvtColor(WORLD*255, cv2.COLOR_GRAY2BGR))
        print(f'\r{(frame+1)*10**(no_decimals_perc + 2)//NO_ITERATIONS/10**no_decimals_perc}%\t|    Time remaining: {datetime.timedelta(seconds=(NO_ITERATIONS-frame)//((frame+1)/(time.time() - start_time + 1)))}', end='\t\t')

main()