import cv2
import numpy as np
from utilities import imshow


img = cv2.imread("Data/CV/3.jpg")
img = cv2.resize(img,None,fx=0.5,fy=0.5)
cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow("img",img)

rows,cols,channels = img.shape
print(f"img has size (rows,cols,channels) = ({rows},{cols},{channels})")



img_zeroes = np.zeros((10,10),np.uint8) # Create a zero matrix of size 10x10
img_ones   = np.ones_like(img_zeroes) # Create a ones matrix of size 10x10
print("img_zeroes",img_zeroes,"Shape = ",img_ones.shape)
print("img_ones",img_ones,"Shape = ",img_ones.shape)
img_white = img_ones*255

print("img_White.dtype",img_white.dtype)# check type of numpy array

imshow("img_zeroes",img_zeroes)
imshow("img_White",img_white)



# Creating a duplicate of an image in python
img_gray = img_white.copy() - 128

imshow("img_halfnhalf",img_gray)
cv2.waitKey(0)

vid = cv2.VideoCapture("Data\CV\Megamind.avi")

while(vid.read()[0]):
    frame = vid.read()[1]
    imshow("frame",frame)
    k = cv2.waitKey(33)
    if k==27: # Break on Esc
        break


