import cv2
# Path/Files access imports
import glob
import os
# Logging import
from loguru import logger
# Utility imports
from src.utilities import print_h



# Callback function for mouse event
def onclick_comp(event,x,y,flags,param):
    global img_iter,curr_img,windowSize

    curr_w,curr_h = windowSize
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < int(curr_w/2):
            img_iter -=1
        elif x > int(curr_w/2):
            if img_iter<total_images-1:
                img_iter+=1
                
    elif event == cv2.EVENT_MOUSEMOVE:
        if curr_img is not None:
            img_disp = curr_img.copy()
            
            if x < int(curr_w/2):
                cv2.arrowedLine(img_disp, (40      ,int(curr_h/2)), (10      ,int(curr_h/2)),(0,140,255), 13,tipLength=0.8)
            elif x > int(curr_w/2):
                cv2.arrowedLine(img_disp, (curr_w-40,int(curr_h/2)), (curr_w-10,int(curr_h/2)),(0,140,255), 13,tipLength=0.8)
            cv2.imshow("Image Viewer",img_disp)
        else:
            logger.warning("curr_img is None, Check assignment!")


def assignment(debug = True):
    # Assignment : Complete the Image Viewer by adding two new features
    #              1) Resize (Upscale/Downscale) an image using mouse
    #              2) Display current pixel intenstiy alongside the cursor
    #
    # Returns    : (curr_img) Image that you are currently navigating in whatever state it is in.
    #
    # Restrictions: No Qt or HighGui usage
    #
    # Hints      : Utilize mousescroll wheel event to control resizing of image
    #            : Handle changing pixel dimensions (gray or bgr) when displaying intensity
    #            : Write <code to disp pixel intensity> in assignments while loop to get a
    #            : constant display
    if debug:
        print_h('''[Assignment]: Complete the Image Viewer, By adding following two features.\n
                1: Resize on Mouse Scroll Wheel Up/Dow (Remember resizing should be only in the current image
                Navigating to the next image should default back to original window size)
                2: Display current pixel intensity alongside the cursor
                ''')
    #Input
    global total_images,curr_img,windowSize
    # List of image file extensions to include
    image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']

    # Generate a list of file paths for images with the specified extensions
    image_list = []
    for ext in image_extensions:
        image_list.extend(glob.glob(f"Data/*.{ext}"))
    # Fetch image paths from directory
    total_images = len(image_list)
    
    # [Mouse Callback]: a> Create a NamedWindow on which mousecallback will be checked
    cv2.namedWindow("Image Viewer",cv2.WINDOW_AUTOSIZE)
    
    # [Mouse Callback]: b> Creating mouse callback on given window and passing the callback function
    cv2.setMouseCallback("Image Viewer",onclick_comp)
    
    while(1):
        # Fetching current image path
        img_path = image_list[img_iter]
        # Reading current image
        img_orig = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        # Resizing to fit in callback window
        windowSize = (WIDTH,HEIGHT)
        curr_img = cv2.resize(img_orig,windowSize)
        # Overlaying image name on callback window
        img_name = os.path.basename(img_path)
        new_width = cv2.getTextSize(img_name, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0][0]        
        cv2.rectangle(curr_img,(20,20),(25 + new_width,50),(0,0,0),-1)        
        cv2.putText(curr_img,img_name,(25,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        
        
        # Display the image viewer
        cv2.imshow("Image Viewer",curr_img)
        k = cv2.waitKey(1)
        if k==27:# break loop on Esc
            break
    cv2.destroyAllWindows()
        
    # Current the current image in its state
    return curr_img


# Our Picture Viewer Default Window Size
WIDTH = 480
HEIGHT = 360
# Global Variables
curr_img = None
img_iter = 0 # Current image we are viewing
total_images = 100

# Callback function for mouse event
def onclick(event,x,y,flags,param):
    global img_iter,curr_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < int(WIDTH/2):
            img_iter -=1
        elif x > int(WIDTH/2):
            if img_iter<total_images-1:
                img_iter+=1
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if curr_img is not None:
            if x < int(WIDTH/2):
                cv2.arrowedLine(curr_img, (40      ,int(HEIGHT/2)), (10      ,int(HEIGHT/2)),(0,140,255), 13,tipLength=0.8)
            elif x > int(WIDTH/2):
                cv2.arrowedLine(curr_img, (WIDTH-40,int(HEIGHT/2)), (WIDTH-10,int(HEIGHT/2)),(0,140,255), 13,tipLength=0.8)
            cv2.imshow("Image Viewer",curr_img)
        else:
            logger.warning("curr_img is None, Check assignment!")


def main():
    # [TASK]: Develop a system to scroll through images in a directory. Using mouse click?
    print_h("[main]: Utilizing OpenCV Mouseevents to create our own Image Viewer :)")
    
    global total_images,curr_img
    
    # Fetch image paths from directory
    # List of image file extensions to include
    image_extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']

        # Generate a list of file paths for images with the specified extensions
    image_list = []
    for ext in image_extensions:
            image_list.extend(glob.glob(f"Data/*.{ext}"))
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
        curr_img = cv2.resize(img_orig,windowSize)
        # Overlaying image name on callback window
        img_name = os.path.basename(img_path)
        new_width = cv2.getTextSize(img_name, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0][0]
        cv2.rectangle(curr_img,(20,20),(25 + new_width,50),(0,0,0),-1)
        cv2.putText(curr_img,img_name,(25,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        
        
        # Display the image viewer
        cv2.imshow("Image Viewer",curr_img)
        k = cv2.waitKey(1)
        if k==27:# break loop on Esc
            break
    cv2.destroyAllWindows()
        
        






if __name__ == "__main__":
    i_am_ready = False
    
    if i_am_ready:
        assignment()
    else:
        main()