from email.mime import image
from pickletools import uint8
from re import L
import cv2
import numpy as np
from utilities import imshow

r_off = 0
g_off = 0
b_off = 0
a_off = 255

def on_r_change(val):
    global r_off
    r_off = val

def on_g_change(val):
    global g_off
    g_off = val

def on_b_change(val):
    global b_off
    b_off = val

def on_a_change(val):
    global a_off
    a_off = val


def main():
    img = cv2.imread("Data\CV\messi5.jpg")
    imshow("img",img)
    cv2.waitKey(0)

    cv2.namedWindow("Trackbar",cv2.WINDOW_NORMAL)
    cv2.createTrackbar("r_off","Trackbar",r_off,255,on_r_change)
    cv2.createTrackbar("g_off","Trackbar",g_off,255,on_g_change)
    cv2.createTrackbar("b_off","Trackbar",b_off,255,on_b_change)
    cv2.createTrackbar("a_off","Trackbar",a_off,255,on_a_change)


    while(1):
        img_w_transparency = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        txt = f"(r_off,g_off,b_off,a_off) = ( {r_off} , {g_off} , {b_off} , {a_off}"
        cv2.putText(img_w_transparency,txt,(50,50),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
        img_w_transparency[:,:,0] = img_w_transparency[:,:,0] + b_off
        img_w_transparency[:,:,1] = img_w_transparency[:,:,1] + g_off
        img_w_transparency[:,:,2] = img_w_transparency[:,:,2] + r_off
        img_w_transparency[:,:,3] = img_w_transparency[:,:,3] + a_off
        imshow("Trackbar",img_w_transparency)
        k = cv2.waitKey(1)
        if k==27:
            break
    
    # Assignment. Turn football from yellow to blue
    # Hint (Use SelectROI and trackbar for getting to the desired result) [91,92,96]
    messi_img = img
    imshow("messi",messi_img)
    cv2.waitKey(0)


if __name__=="__main__":
    main()