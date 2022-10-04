import cv2
from utilities import imshow,print_h,build_montages
from loguru import logger


def get_plant(field_img):
	selected_plant = field_img.copy()
	# Write Code here... 
	# Hint: Investigate HSL channels for the right answer

	return selected_plant

def assignment():
	print_h("[Assignment]: Retreive and display only the (bottom-left) plant in the field")
	# Assignment: Display only the bottom-left plant in the field of image given below.
	# Conditions: Result should be in single-channel (Without-Plant-Shadow)
	
	# Input
	field_img = cv2.imread("Data\CV\drone_view.png")
	imshow("field_img",field_img)
	cv2.waitKey(0)

	# Task Function
	selected_plant = get_plant(field_img)

	# Output (Display)
	imshow("selected_plant",selected_plant)
	cv2.waitKey(0)


def main():
	# Read an image
	img = cv2.imread("Data\CV\messi5.jpg")
	# Display
	imshow("image",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# a) Select ROI on the image
	rect = cv2.selectROI("Select ROI",img)
	x,y,w,h = rect
	if (w==0 or h==0):
		logger.opt(colors=True).error("[ Invalid ROI provided ]\n<green>Solution:</green> <white>Manually specify the ROI in 'Select ROI' window</white>")
		exit(0)

	images = []
	titles = []
	# b) Crop the Selected ROI 
	roi = img[y:y+h,x:x+w] # img[row:row+height,col+col+width]
	images.append(roi)
	titles.append("ROI")
	
	# c) Change color Space of the image
	roi_gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
	images.append(roi_gray)
	titles.append("roi_gray")
	

	images.append(img)
	titles.append("img")

	# d) Change color Space of the image to hsv and display each channel
	hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)

	
	images.append(hls[:,:,0])
	titles.append("hue")
	
	images.append(hls[:,:,1])
	titles.append("lit")
	
	images.append(hls[:,:,2])
	titles.append("sat")
	
	montages = build_montages(images,None,None,titles,True,True)
	for img in montages:
		cv2.imshow("image",img) # Show large image
	cv2.waitKey(0)
	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()

	# b) assignment for the student to test his knowledge
	assignment()