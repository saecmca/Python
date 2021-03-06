# -*- coding: utf-8 -*-
"""Final_worked_Perno_OpenCV_ImageStitching-safe.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XOIdrchKDSpI5b_84GXnk8rbZ5Iu_FBJ

<a href="https://colab.research.google.com/github/sthalles/computer-vision/blob/master/project-4/project-4.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# from google.colab import drive
# drive.mount('/content/gdrive')

# import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
import glob
import os
# cv2.ocl.setUseOpenCL(False)
import subprocess

# import sys 
# sys.path.append('/home/siva/Desktop/ivy/stitching_images')


from utl_necess import *

tester_name="Omar Feliciano Vega-SC PLAZA ARBOLEDAS"
rel_path='/home/siva/Desktop/ivy/stitching_images/ivy_dec30/'
dirs=os.listdir(rel_path+tester_name)
dirs

# per=20

# def remove_oldfiles(dir):
#     import os
#     if os.path.exists(rel_path+tester_name+"/"+str(dir)+"/temp_1.png"):
#         os.remove(rel_path+tester_name+"/"+str(dir)+'/temp_1.png')
#     if os.path.exists(rel_path+tester_name+"/"+str(dir)+'/temp_2.png'):
#           os.remove(rel_path+tester_name+"/"+str(dir)+'/temp_2.png')
#     if os.path.exists(rel_path+tester_name+"/"+str(dir)+'/temp_3.png'):
#           os.remove(rel_path+tester_name+"/"+str(dir)+'/temp_3.png')   
#     if os.path.exists(rel_path+tester_name+"/"+str(dir)+'/final_stitched.png'):
#           os.remove(rel_path+tester_name+"/"+str(dir)+'/final_stitched.png') 
#     print("old results deleted...")

# def apply_remove_oldfiles():
#     for direct in dirs:
#         remove_oldfiles(direct,rel_path,tester_name)   
        
apply_remove_oldfiles(dir,rel_path,tester_name,dirs)



# import cv2 as cv

# im = cv.imread("/home/siva/Desktop/ivy/stitching_images/ivy_dec23/Alejandro Rivera Albarrán -SM TOREO/26bd088538d2e2d6015a9d2f30fe4a44/26bd088538d2e2d6015a9d2f30fe4a44image_0_0.png")

# height, width = im.shape[:2]

# thumbnail = cv.resize(im, (1440, 1920), interpolation=cv.INTER_AREA)

# # cv.imshow('exampleshq', thumbnail)


# thumbnail.shape[:2]
# cv.waitKey(0)
# cv.destroyAllWindows()

#

        
for dir in dirs:  
    print(dir)
    final_stitching(dir,rel_path,tester_name,per) 
        





