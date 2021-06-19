import cv2
import os

ime= cv2.imread('data/images/2020.jpg')
#ime2= cv2.imshow('image',ime)

path = 'data/images/'
for i in os.listdir(path):
    print(i)

#flags.DEFINE_string('image', './data/'++'.jpg', 'path to input image')



cv2.waitKey(0)
cv2.destroyAllWindows()

