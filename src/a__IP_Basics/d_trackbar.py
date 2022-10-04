import cv2
from utilities import imshow,print_h


def change_roi_clr(messi_img):
	messi_img_bluefootball = messi_img.copy()
	# Write Code here... 
	# Hint (Use SelectROI and trackbar for getting to the desired result) [91,92,96]


	return messi_img_bluefootball

def assignment():
	print_h("[Assignment]: Modify img(messi_img) where yellow football turns blue")
	# Assignment. Turn football from yellow to blue

	# Input
	messi_img = cv2.imread("Data\CV\messi5.jpg")
	imshow("messi",messi_img)
	cv2.waitKey(0)

	# Task Function
	messi_img_bluefootball = change_roi_clr(messi_img)
	
	# Output (Display)
	imshow("messi_img (bluefootball)",messi_img_bluefootball)
	cv2.waitKey(0)


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
	print_h("[main]: Utilizing trackbar to understand the effect of change on individuals colors and transparency")
	img = cv2.imread("Data\CV\messi5.jpg")
	imshow("messi",img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	cv2.namedWindow("Trackbar",cv2.WINDOW_AUTOSIZE)
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
	




if __name__=="__main__":
	main()

	assignment()