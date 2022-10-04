import cv2
import numpy as np
from numpy import random
from utilities import imshow,get_optimal_font_scale,print_h
import glob
import os
from loguru import logger


WIDTH = 480
HEIGHT = 360

img = None
img_iter = 0 # Current image we are viewing
N = 100

RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
ORANGE = (0,140,255)
BROWN = (42,42,165)

def bubblepop(img_smarties):
	colors = [RED,GREEN,BLUE,ORANGE,BROWN]
	colors_name = ["RED","GREEN","BLUE","ORANGE","BROWN"]
	while(1):
		curr_iter = random.randint(0,len(colors))
		clr = colors[curr_iter]
		clr_name = colors_name[curr_iter]

		#hue = (cv2.cvtColor(img_smarties,cv2.COLOR_BGR2HLS))[:,:,0]
		gray = cv2.cvtColor(img_smarties,cv2.COLOR_BGR2GRAY)
		img_smarties_disp = img_smarties.copy()
		imshow("gray",gray)


		# cv2.putText(frame_draw,"Sign Detected ==> "+str(signTrack.Tracked_class),(20,85),cv2.FONT_HERSHEY_COMPLEX,0.75,(255,255,0),2)
		NumOfVotesForCircle = 40 #parameter 1 MinVotes needed to be classified as circle
		CannyHighthresh = 130 # High threshold value for applying canny
		mindDistanBtwnCircles = 10 # kept as sign will likely not be overlapping
		max_rad = 200 # smaller circles dont have enough votes so only maxRadius need to be controlled 
						# As signs are right besides road so they will eventually be in view so ignore circles larger than said limit

		# 4. Detection (Localization)
		circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,mindDistanBtwnCircles,param1=CannyHighthresh,param2=NumOfVotesForCircle,minRadius=16,maxRadius=max_rad)
		# 4a. Detection (Localization) Checking if circular regions were localized
		if circles is not None:
			circles = np.uint16(np.around(circles))
			# 4b. Detection (Localization) Looping over each localized circle
			for i in circles[0,:]:
				center =(i[0],i[1])
				radius = i[2]
				cv2.circle(img_smarties_disp,center,radius,(120,128,255),4)

		cv2.putText(img_smarties_disp,str(clr_name),(15,25),cv2.FONT_HERSHEY_PLAIN,1,clr,2)
		imshow("img_smarties_disp",img_smarties_disp)
		
		k = cv2.waitKey(0)
		if k==27:
			break

def assignment():
	print_h("[Assignment]: Select the smarties in order their color is named out")
	# Assignment : Complete the game where user has to select the smarties in order their color is named out.
	#                                            In case of 3 wrong attempt. You will have to start over
	#              Closing-Condition: When all smarties have been selected in the right order, Succcess!
	
	#Input
	img_smarties = cv2.imread("Data\CV\smarties.png")    
	imshow("img_smarties",img_smarties)
	cv2.waitKey(0)
	cv2.destroyWindow("img_smarties")

	# Task Function
	result = bubblepop(img_smarties)
	
	# Output (Display)
	imshow("Smarties Game (Result)",result)
	cv2.waitKey(0)


# [NEW]: Update the destination to user selected location
def click_event(event, x, y, flags, params):
	global img_iter,N,img
	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:
		if x < int(WIDTH/2):
			if img_iter>0:
				img_iter-=1
		elif x > int(WIDTH/2):
			if img_iter<N-1:
				img_iter+=1
	elif event == cv2.EVENT_MOUSEMOVE:
		if img is not None:
			if x < int(WIDTH/2):
				cv2.arrowedLine(img, (40      ,int(HEIGHT/2)), (10      ,int(HEIGHT/2)),(0,140,255), 13,tipLength=0.8)
			elif x > int(WIDTH/2):
				cv2.arrowedLine(img, (WIDTH-40,int(HEIGHT/2)), (WIDTH-10,int(HEIGHT/2)),(0,140,255), 13,tipLength=0.8)
			cv2.imshow("Image Viewer",img)
		else:
			logger.warning("img is None, Check assignment!")

def main():
	print_h("[main]: Understanding and utilizing mouseevents to create our own Image Viewer :)")
	global img
	# Lets develop a system to scroll through images in a directly. Through a mouse click?
	image_list = [filename for filename in glob.glob('Data/CV/*.png')]
	image_list = image_list + [filename for filename in glob.glob('Data/CV/*.png')]
	N = len(image_list)
	
	
	cv2.namedWindow("Image Viewer",cv2.WINDOW_AUTOSIZE)
	windowSize = (WIDTH,HEIGHT)
	cv2.setMouseCallback("Image Viewer",click_event)
	while(1):
		img_path = image_list[img_iter]
		img_name = os.path.basename(img_path)
		new_width = cv2.getTextSize(img_name, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0][0]
		img_orig = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
		img = cv2.resize(img_orig,windowSize) # Resizing every image to default windowSize (480,360)
		cv2.rectangle(img,(20,20),(25 + new_width,50),(0,0,0),-1)
		cv2.putText(img,img_name,(25,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2) # Displaying the image name
		cv2.imshow("Image Viewer",img)
		k = cv2.waitKey(1)
		if k==27:# break loop on Esc
			break
	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()

	assignment()