import cv2
import numpy as np
from src.a__IP_Basics.utilities import imshow
from utilities import get_optimal_font_scale,putText_bbox,build_montages,print_h


def remove_sun(sunset_img):
	sunset_without_sun = sunset_img.copy()
	
	return sunset_without_sun

def assignment():
	print_h("[Assignment]: Remove particular ROI(Sun) from an img(Sunset_img)")
	# Assignments : Write remove_sun code to get everything except the sun in the sunset_img 
	#         Hint: See circle, bitwise_not in OPENCV
	sunset_img = cv2.imread("Data/CV/3.jpg")
	imshow("sunset_img",sunset_img)

	sunset_without_sun = remove_sun(sunset_img)
	imshow("sunset_without_sun",sunset_without_sun)
	cv2.waitKey(0)

def main():
	# Task : Simple Image Manipulation
	print_h("1: Image creation using numpy")
	images = []
	titles = []
	# a) creating an image using numpy
	img_zeroes = np.zeros((10,10),np.uint8) # Create a zero matrix of size 10x10
	print(f"\nimg_zeroes {img_zeroes.shape}\n",img_zeroes)
	images.append(img_zeroes)
	titles.append("img_zeroes")
	
	img_ones   = np.ones_like(img_zeroes) # Create a ones matrix of size 10x10
	print(f"\nimg_ones {img_zeroes.shape}\n",img_ones)

	img_white = img_ones*255
	images.append(img_white)
	titles.append("img_white")
	print("\nimg_White.dtype",img_white.dtype)# check type of numpy array

	# b) duplicating an image without modyfing the original
	img_gray = img_white.copy() - 128	
	images.append(img_gray)
	titles.append("img_halfnhalf")


	# Task : Pixel-Wises Manipulation Manipulation
	# Pixels : A pixel is light intensity value in digital terms for an img_eyes at a specific location
	print_h("2: Pixel-Wise Image Manipulation")

	# img_eyes = imread("data/Messi5.jpg")
	img_eyes = np.eye(10,dtype=np.uint8)*255

	# #In case of multi - channels type, only the first channel will be initialized with 1's, the others will be set to 0's.
	#imshow("Before Pixel Acces", img_eyes)
	# Assesing img_eyes info prior for pixel manipulation
	images.append(img_eyes)
	titles.append("img_eyes")

	# 1: Assesing pixel location from an img_eyes
	data = img_eyes[5,5]
	print("data = ",data)

	# 2: Changing pixel value 
	img_eyes_pix_acces = img_eyes.copy()
	img_eyes_pix_acces[5][5] = 0
	images.append(img_eyes_pix_acces)
	titles.append("After Pixel Acces")

	#4: Numpy array slicing
	img_eyes_slicing = img_eyes.copy()
	img_eyes_slicing[0:5,0:5] = 255
	images.append(img_eyes_slicing)
	titles.append("Modified Top Left Quarter")

	#5: Use-Case: Get only the roi in an image
	print_h("3: Extracting/Modifying ROI in image")
	sunset_img = cv2.imread("Data/CV/3.jpg")
	images.append(sunset_img)
	titles.append("sunset_img")

	rows,cols = sunset_img.shape[0:2]
	man_watching_sunset = sunset_img[int(rows*0.45):int(rows*0.75),int(cols*0.45):int(cols*0.75)]
	images.append(man_watching_sunset)
	titles.append("man_watching_sunset")

	# 6 Drawing a horizontal line intersecting the image center
	pt_strt = (0,int(rows/2))
	pt_end = (cols,int(rows/2))
	sunset_img_horizon = cv2.line(sunset_img.copy(),pt_strt,pt_end,(0,255,0),3)
	images.append(sunset_img_horizon)
	titles.append("sunset_img_horizon")

	montages = build_montages(images,None,None,titles,True,True)
	for img in montages:
		cv2.imshow("image",img) # Show large image
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Clearning both images and titles lists
	images.clear()
	titles.clear()
	# 7: Draw a mask using rectangle func around the roi
	print_h("4: Masking ROI in Image")

	mask_roi = np.zeros((sunset_img.shape[0],sunset_img.shape[1]),np.uint8)
	rect_strt = (int(cols*0.45),int(rows*0.45)) # 45% of the rows and cols becomes the rect start
	rect_end = (int(cols*0.75),int(rows*0.75)) # 75% of the rows and cols becomes the rect start
	cv2.rectangle(mask_roi,rect_strt,rect_end,255,-1)
	images.append(mask_roi)
	titles.append("mask_roi")


	# 8: Only display the roi in sunset_img (Bitwise Operation)
	sunset_img_roi = cv2.bitwise_and(sunset_img,sunset_img,mask= mask_roi )
	images.append(sunset_img_roi)
	titles.append("sunset_img_roi")

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
	putText_bbox(sunset_img_roi_disp,txt_list,orig_list)
	
	images.append(sunset_img_roi_disp)
	titles.append("sunset_img_roi_disp")
	montages = build_montages(images,None,None,titles,True,True)
	for img in montages:
		cv2.imshow("image",img) # Show large image
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

    # b) assignment for the student to test his knowledge
	assignment()