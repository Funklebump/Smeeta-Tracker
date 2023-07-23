
import cv2
import numpy as np
import os
from pathlib import Path
script_folder = Path(__file__).parent.absolute()

fpath = os.path.join(script_folder, '_Lotus_Interface_Icons_GeneticLab_TemperamentCheshire.png')

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=255)
    return result

img = cv2.imread(fpath, flags = cv2.IMREAD_UNCHANGED)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

weighted_img = 255-thresh1[:,:, 3]

for ui_scale in [1,1.5,2,2.5,3]:
    rotated =  rotate_image(weighted_img, 3.5/ui_scale)
    scaled = cv2.resize(rotated, (0,0), fx=0.25*ui_scale, fy=0.25*ui_scale) 
    thresh1=scaled
    #ret,thresh1 = cv2.threshold(scaled,180,255,cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(script_folder, f'{int(100*ui_scale)}p_new.png'), thresh1)
    #cv2.imshow("", scaled)
    #cv2.waitKey(0)
