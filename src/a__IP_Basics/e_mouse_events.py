import cv2
# Path/Files access imports
import glob
import os
# Logging import
from loguru import logger
# Utility imports
from src.utilities import imshow,get_optimal_font_scale,print_h


# Our Picture Viewer Default Window Size
WIDTH = 480
HEIGHT = 360
# Global Variables
img = None
img_iter = 0 # Current image we are viewing
total_images = 100


# Callback function for mouse event
def onclick(event,x,y,flags,param):
    global img_iter,img
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < int(WIDTH/2):
            img_iter -=1
        elif x > int(WIDTH/2):
            if img_iter<total_images-1:
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
    # [TASK]: Develop a system to scroll through images in a directory. Using mouse click?
    print_h("[main]: Utilizing OpenCV Mouseevents to create our own Image Viewer :)")
    
    global total_images,img
    
    # Fetch image paths from directory
    image_list = [filepaths for filepaths in glob.glob("Data/*.png")]
    total_images = len(image_list)
    
    # Set default windowsize for picture viewer
    windowSize = (WIDTH,HEIGHT)
    
    # [Mouse Callback]: a> Create a NamedWindow on which mousecallback will be checked
    cv2.namedWindow("Image Viewer",cv2.WINDOW_AUTOSIZE)
    
    # [Mouse Callback]: b> Creating mouse callback on given window and passing the callback function
    cv2.setMouseCallback("Image Viewer",onclick)
    
    while(1):
        # Fetching current image path
        img_path = image_list[img_iter]
        # Reading current image
        img_orig = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        # Resizing to fit in callback window
        img = cv2.resize(img_orig,windowSize)
        # Overlaying image name on callback window
        img_name = os.path.basename(img_path)
        new_width = cv2.getTextSize(img_name, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0][0]
        cv2.rectangle(img,(20,20),(25 + new_width,50),(0,0,0),-1)
        cv2.putText(img,img_name,(25,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        
        
        # Display the image viewer
        cv2.imshow("Image Viewer",img)
        k = cv2.waitKey(1)
        if k==27:# break loop on Esc
            break
    cv2.destroyAllWindows()
        
        









if __name__ == "__main__":
    main()