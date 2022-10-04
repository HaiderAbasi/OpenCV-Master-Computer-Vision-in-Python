import cv2
import numpy as np
from src.a__IP_Basics.utilities import imshow,describe,build_montages,print_h



def highlight_roi(image):
	# Hint: Enhancing meteor edges could be the key but there is alot of noise so some smoothing is required
	roi_highlighted = image.copy()
	# Write Code here...

	return roi_highlighted

def assignment():
	print_h("[Assignment]: Highlight falling meteor in the scene")
	# Assignment: Define the algorithm whos goal is to highlight meteor (roi) in the whole scene

	# Input
	vid = cv2.VideoCapture("Data\CV\meteor.mpg")
	while(vid.isOpened()):
		ret,frame = vid.read()
		if ret:

			# Task Function
			roi_highlighted = highlight_roi(frame)
			
			# Output (Display)
			imshow("meteor strike",roi_highlighted)
			k=cv2.waitKey(1)
			if k==27:
				break
		else:
			print("Video Ended")
			break



k_w = 3
def on_k_w_Change(val):
	global k_w
	k_w = 2*val + 1

	if (k_w<3):
		k_w = 3

k_h = 3
def on_k_h_Change(val):
	global k_h
	k_h = 2*val + 1

	if (k_h<3):
		k_h = 3

ksize = 3
def onksizeChange(val):
	global ksize
	ksize = 2*val + 1

	if (ksize<3):
		ksize = 3


image_no = 0
def on_image_no_Change(val):
	global image_no
	image_no = val

thresh = 100
def on_thresh_Change(val):
	global thresh
	thresh = val

def noise_removal_test(noisy_img):
	global image_no

	images = []
	titles = []   


	images.append(noisy_img)
	titles.append("noisy_img")

	#   a) LPF # 1: Box filter (Linear filter Used for smoothing, edge preservation minimal)
	kernel = np.ones((5,5),np.float32)/25 # box filter 
	img_filtered = cv2.filter2D(noisy_img,-1,kernel) # depth = -1 (As input depth)
	images.append(img_filtered)
	titles.append("filtered (box)")


	#   b) LPF #2: Guassian filter (Linear filter Used for smoothing or to reduce noise , edge preservation okay)
	#                               Gives more weightage to closer pixels then farther in deciding result
	img_guass = cv2.GaussianBlur(noisy_img,(k_w,k_h),0,0)# Standard deviation in x and y direction of guassian kernel to be computed using kernel widht and height
	images.append(img_guass)
	titles.append(f"filtered (guass {(k_w,k_h)} )")
	
	#   b) LPF #2: Median filter (Non-linear filter used for denoising (Salt-pepper noise), edge preservation better)
	#                             Slow, because it needs to perform sorting to find the median in the underlying array
	img_median = cv2.medianBlur(noisy_img,ksize)
	images.append(img_median)
	titles.append(f"filtered (median {ksize})")

	#   c) Combo #A Median -< Guassian
	img_medguass = cv2.GaussianBlur(img_median,(k_w,k_h),0,0)# Standard deviation in x and y direction of guassian kernel to be computed using kernel widht and height
	images.append(img_medguass)
	titles.append(f"filtered (medguass)") 

	total_images = len(images)
	if image_no >= (total_images):
		image_no = (total_images-1) - image_no

	threshed = cv2.threshold(images[image_no],thresh,255,cv2.THRESH_BINARY)[1]
	images.append(threshed)
	titles.append(f"threshed {titles[image_no]}") 

	montage_shape = (300,200)
	montage = build_montages(images,montage_shape,None,titles,True,True)
	for img in montage:
		imshow("Blurring (Noise Removal)",img)


def edge_detection_test(image):
	images = []
	titles = []
	
	images.append(image)
	titles.append("image (Original)")

	edge_X = cv2.Sobel(image,-1,1,0)
	images.append(edge_X)
	titles.append("edges (X)")
	
	edge_X_64f = cv2.Sobel(image,cv2.CV_64F,1,0)
	edge_X_char = cv2.convertScaleAbs(edge_X_64f, alpha=(255/edge_X_64f.max()))
	#describe(edge_X_64f,"edge_X (64f)")
	#describe(edge_X_char,"edge_X (Char)")
	#cv2.waitKey(0)
	images.append(edge_X_char)
	titles.append("edge_X (Scaleabs)")

	edge_Y = cv2.Sobel(image,-1,0,1)
	images.append(edge_Y)
	titles.append("edges (Y)")


	# Func: convertScaleAbs() Path: opencv/opencv/modules/core/src/convert_scale.dispatch.cpp
	# 1.The first operation is to re-scale the source image by the factor alpha. (Normalizing to the new desired range)
	# 2.The second is to offset by (add) the factor beta.
	# 3.The third is to compute the absolute  value of that sum.
	# 4.The fourth is to cast that result (with saturation) to an unsigned char (8-bit).
	# find minimum and maximum intensities
	#minVal = np.amin(sobelx)
	#maxVal = np.amax(sobelx)
	#draw = cv.convertScaleAbs(sobelx, alpha=255.0/(maxVal - minVal), beta=-minVal * 255.0/(maxVal - minVal))
	edge_Y_64f = cv2.Sobel(image,cv2.CV_64F,0,1)
	minVal = np.amin(edge_Y_64f)
	maxVal = np.amax(edge_Y_64f)
	edge_Y_char = cv2.convertScaleAbs(edge_Y_64f, alpha=255.0/(maxVal - minVal), beta=-minVal * 255.0/(maxVal - minVal))

	images.append(edge_Y_char)
	titles.append("edge_Y (Scaleabs)")


	edge_XY = cv2.Sobel(image,cv2.CV_64F,1,1)
	edge_XY = cv2.convertScaleAbs(edge_XY, alpha=(255/edge_XY.max()))
	images.append(edge_XY)
	titles.append("edges (XY)")

	edges_laplace = cv2.Laplacian(image,cv2.CV_64F,3)
	edges_laplace = cv2.convertScaleAbs(edges_laplace, alpha=(255/edges_laplace.max()))
	images.append(edges_laplace)
	titles.append("edges (Laplacian)")

	edges_canny = cv2.Canny(image,50,150,None,3)
	images.append(edges_canny)
	titles.append("edges (Canny)")

	montage_shape = (300,200)
	montage = build_montages(images,montage_shape,None,titles,True,True)
	for img in montage:
		imshow("Edge detection",img)


def main():
	print_h("[main]: Applying different type of Image filters to input and analyzing their effects.")
	images = []
	titles = []

	img = cv2.imread("Data\CV\HappyFish.jpg")
	images.append(img)
	titles.append("img (Orig)")

	# 1) Smoothing using filter2d (box filter)
	kernel = np.ones((5,5),np.float32)/25 # box filter 
	print("kernel = ",kernel) # anchor is where the result of the filter after computation be applied (default:center)
	img_filtered = cv2.filter2D(img,-1,kernel) # depth = -1 (As input depth)
	
	images.append(img_filtered)
	titles.append("filtered (box)")

	#montage_shape = (300,200)
	montage = build_montages(images,None,None,titles,True,True)
	for img in montage:
		imshow("Image Filtering",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	print_h("[Noise Removal]: Utilizing low-pass filters for noise removal in the given image.")
	# 1) Noise Removal using low pass filters (Choise of filter depends on the noise characteristics)
	noisy_img = cv2.imread("Data/CV/noisy2.png",cv2.IMREAD_ANYDEPTH) # AnyDepth to ensure it is read as a grayscale

	# Creating all trackbars for live parameter tuning
	cv2.namedWindow("Noise Removal",cv2.WINDOW_NORMAL)
	cv2.createTrackbar("k_w","Noise Removal",k_w,30,on_k_w_Change)
	cv2.createTrackbar("k_h","Noise Removal",k_h,30,on_k_h_Change)
	cv2.createTrackbar("ksize","Noise Removal",ksize,30,onksizeChange)
	cv2.createTrackbar("thresh","Noise Removal",thresh,255,on_thresh_Change)
	cv2.createTrackbar("image_no","Noise Removal",image_no,10,on_image_no_Change)
	while(1):
		noise_removal_test(noisy_img)
		k=cv2.waitKey(1)
		if k==27:
			break

	print_h("[Edge detection]: Leveraging high-pass filters to extract areas of change in image.")
	# 1) Edge detection using high pass filters (Choise of filter depends on the type of edge we want)
	shapes_img = cv2.imread("Data\CV\shapes.PNG",cv2.IMREAD_ANYDEPTH) # AnyDepth to ensure it is read as a grayscale

	# Creating all trackbars for live parameter tuning
	# cv2.namedWindow("Noise Removal",cv2.WINDOW_NORMAL)
	# cv2.createTrackbar("k_w","Noise Removal",k_w,30,on_k_w_Change)
	# cv2.createTrackbar("k_h","Noise Removal",k_h,30,on_k_h_Change)
	# cv2.createTrackbar("ksize","Noise Removal",ksize,30,onksizeChange)
	# cv2.createTrackbar("thresh","Noise Removal",thresh,255,on_thresh_Change)
	# cv2.createTrackbar("image_no","Noise Removal",image_no,10,on_image_no_Change)
	while(1):
		edge_detection_test(shapes_img)
		k=cv2.waitKey(1)
		if k==27:
			break
	cv2.destroyAllWindows()
		


		
if __name__ == "__main__":
	main()

	assignment()