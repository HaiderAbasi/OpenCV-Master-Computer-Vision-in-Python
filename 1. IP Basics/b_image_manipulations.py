import cv2
import numpy as np
from utilities import imshow,get_optimal_font_scale,putText




def main():
    
	# Task : Image Pixel Manipulation
	# Pixels : A pixel is light intensity value in digital terms for an img_ones at a specific location

	# img_ones = imread("data/Messi5.jpg")
	img_ones = np.eye(10,dtype=np.uint8)*255

	# #In case of multi - channels type, only the first channel will be initialized with 1's, the others will be set to 0's.
	imshow("Before Pixel Acces", img_ones)
	# Assesing img_ones info prior for pixel manipulation


	# 1: Assesing pixel location from an img_ones
	data = img_ones[5,5]
	print("data = ",data)

	# 2: Changing pixel value 
	img_ones[5][5] = 0
	imshow("After Pixel Acces", img_ones)

	#4: Numpy array slicing
	img_ones[0:5,0:5] = 255
	imshow("Modified Top Left Quarter of img_ones", img_ones)

	#5: Use-Case: Get only the roi in an image
	sunset_img = cv2.imread("Data/CV/3.jpg")
	imshow("sunset_img",sunset_img)

	rows,cols = sunset_img.shape[0:2]
	man_watching_sunset = sunset_img[int(rows*0.45):int(rows*0.75),int(cols*0.45):int(cols*0.75)]
	imshow("man_watching_sunset",man_watching_sunset)

	# 6 Drawing a horizontal line intersecting the image center
	pt_strt = (0,int(rows/2))
	pt_end = (cols,int(rows/2))
	sunset_img_horizon = cv2.line(sunset_img.copy(),pt_strt,pt_end,(0,255,0),3)
	imshow("sunset_img_horizon",sunset_img_horizon)

	# 7: Draw a mask using rectangle func around the roi
	mask_roi = np.zeros((sunset_img.shape[0],sunset_img.shape[1]),np.uint8)
	rect_strt = (int(cols*0.45),int(rows*0.45)) # 45% of the rows and cols becomes the rect start
	rect_end = (int(cols*0.75),int(rows*0.75)) # 75% of the rows and cols becomes the rect start
	cv2.rectangle(mask_roi,rect_strt,rect_end,255,-1)
	imshow("mask_roi",mask_roi)

	# 8: Only display the roi in sunset_img (Bitwise Operation)
	sunset_img_roi = cv2.bitwise_and(sunset_img,sunset_img,mask= mask_roi )
	imshow("sunset_img_roi",sunset_img_roi)

	# 9: Displaying bbox pts using putText on sunset_img_roi
	sunset_img_roi_disp = sunset_img_roi.copy()
	org = (250,500)
	fontScale = get_optimal_font_scale("ROI", cols)[0]
	cv2.putText(sunset_img_roi_disp,"ROI", org ,cv2.FONT_HERSHEY_PLAIN,fontScale,(0,255,0),4 )

	off_col = 160
	off_row = 80
	lft_col ,top_row = rect_strt
	rgt_col ,btm_row = rect_end
	toplft = (  lft_col - off_col , top_row  - off_row )
	toprgt = (  rgt_col + off_col , top_row  - off_row )
	btmrgt = (  rgt_col + off_col , btm_row  + off_row )
	btmlft = (  lft_col - off_col , btm_row  + off_row )

	orig_list = [toplft,toprgt,btmrgt,btmlft]# Clockwise
	txt_list = [str(orig_list[0]),str(orig_list[1]),str(orig_list[2]),str(orig_list[3])]
	putText(sunset_img_roi_disp,txt_list,orig_list)
	
	imshow("sunset_img_roi_disp",sunset_img_roi_disp)

	cv2.waitKey(0)

	# Assignments : Modify the sunset_img to get everything except the sun (Hint: See circle, bitwise_not in OPENCV)


if __name__ == "__main__":
    main()