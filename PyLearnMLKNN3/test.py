# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# nemo = cv2.imread('input/nemo.jpg')
# nemo = cv2.resize(nemo, (0,0), fx=.25, fy=.25)

# nemo_rgb = cv2.cvtColor(nemo, cv2.COLOR_BGR2RGB)
# nemo_hsv = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)

# pixels_list_rgb = nemo_rgb.reshape(-1, 3)
# pixels_list_hsv = nemo_hsv.reshape(-1, 3)

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(projection='3d')

# ax.scatter(pixels_list_hsv[:,0], pixels_list_hsv[:,1], pixels_list_hsv[:,2], c=pixels_list_rgb/255)
# ax.set_xlabel('red')
# ax.set_ylabel('green')
# ax.set_zlabel('blue')
# plt.show()

# for i in range(100):
#     print i/100," percent complete         \r",
from __future__ import print_function
import sys
import time

for i in range(100):
    print(i/100, end='\r')
    time.sleep(.1)
    sys.stdout.flush()