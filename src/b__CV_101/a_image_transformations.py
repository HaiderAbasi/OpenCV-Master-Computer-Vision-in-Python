import cv2
import numpy as np
from imutils import build_montages

from src.a__IP_Basics.utilities import imshow,draw_points,print_h
from loguru import logger


def get_bookcover(book_in_scne):
	# Hint : Use setmousecallBack() to retrieve points neccesary for computing the transformation matrix
	bookcover = book_in_scne.copy()
	# Type code here

	return bookcover


def assignment():
	# Assignment : Use transformations and previous knowledge to recover only the book front (no background)
	print_h("[Assignment]: Use transformations and previous knowledge to recover only the book front (no background)")

	#Input
	book_img = cv2.imread("Data/CV/book_perspective.jpg")
	imshow("book on table",book_img)
	cv2.waitKey(0)
	cv2.destroyWindow("book on table")

	# Task Function
	bookcover = get_bookcover(book_img)
	if np.array_equal(book_img,bookcover):
		logger.warning("get_bookcover() needs to be coded to get the required(book cover) result.")
		exit(0)


	# Output (Display)
	imshow("bookcover",bookcover)
	cv2.waitKey(0)


def main():
	print_h("[main]: Perform different image transformation on an simple rectangle and analyze the resultant image.")

	images = []
	titles = []

	# Perform Image transformations, Given an interesting assignment to utilize transformations
	img = np.zeros((200,300),np.uint8)
	rows,cols = img.shape[0:2]
	img[80:120,140:160] = 255

	# Adding image to list of images for displaying as a montage
	images.append(img)
	titles.append("Orig")

	# 1) Resizing : Resizing is perhaps the simplest yet most important image transformation.
	#               Its is perhaps the most used preprocessing steps for alot of CV tasks.
	#               One advanced topic is Classification where we use CNN to classify the image.
	#               Now CNN have a fixed input layer meaning the image need to be of a specific size
	#               so we resize the image according to the input dim of CNN
	width = 150
	height = 100
	new_size = (width,height)
	img_resized = cv2.resize(img,new_size)

	# Adding image to list of images for displaying as a montage
	images.append(img_resized)
	titles.append(f"Resized {new_size}")


	# 2) Translation : 
	tx = 100
	ty = 50
	M = np.float32([[1,0,tx],[0,1,ty]])
	img_translated = cv2.warpAffine(img,M,(cols,rows))

	# Adding image to list of images for displaying as a montage
	images.append(img_translated)
	titles.append(f"translated (tx,ty)=({tx},{ty})")


	# 3) Rotation :
	# cols-1 and rows-1 are the coordinate limits.
	angle = 90
	M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),angle,1)
	img_rotated = cv2.warpAffine(img,M,(cols,rows))
	# Adding image to list of images for displaying as a montage
	images.append(img_rotated)
	titles.append(f"rotated {angle} deg")

	# Extracting contours representing our rectangle corners
	cnts = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
	cnt = cnts[0].reshape(4,2)

	# 4) Affine :
	pts1 = np.float32([cnt[0],cnt[1],cnt[2]])
	pts2 = np.float32([[100,67],[100,134],[200,134]])# Default size = (cols,rows) = (300,200)
	M = cv2.getAffineTransform(pts1,pts2)
	img_affine = cv2.warpAffine(img,M,(cols,rows))
	img_affine = draw_points(img_affine,cnt[0:3])
	img_affine = draw_points(img_affine,pts2)
	# Adding image to list of images for displaying as a montage
	images.append(img_affine)
	titles.append("affine")

	# 4) Perspective :
	pts1 = np.float32([cnt[0],cnt[1],cnt[2],cnt[3]]) # Anti-Clockwise
	pts2 = np.float32([[100,67],[120,134],[170,134],[200,67]])# Default size = (cols,rows) = (300,200)
	M = cv2.getPerspectiveTransform(pts1,pts2)
	img_perspective = cv2.warpPerspective(img,M,(cols,rows))
	img_perspective = draw_points(img_perspective,cnt)
	img_perspective = draw_points(img_perspective,pts2)
	# Adding image to list of images for displaying as a montage
	images.append(img_perspective)
	titles.append("perspective")
	
	
	im_shape = (300,200)
	montages = build_montages(images, im_shape, None,titles,False,True)
	
	for img in montages:
		cv2.imshow("image",img) # Show large image
	cv2.waitKey(0)



if __name__ == "__main__":
	main()

	assignment()