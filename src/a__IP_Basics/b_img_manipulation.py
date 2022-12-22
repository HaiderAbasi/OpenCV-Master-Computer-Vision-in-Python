import cv2
import numpy as np

from src.utilities import putText_bbox,build_montages,print_h,imshow

from loguru import logger

def apocalypse(day,moon,moon_mask,sun_loc,rad):
    # [Solar Eclipse]
    # Where Moon comes in between the sun and the earth. :)
    
    eclipse = day.copy()
    
    # Write Code here...
    
        
    return eclipse


def assignment(debug = True):
    # Assignments    :    Complete apocalypse() to turn sunset to solar eclipse
    #
    # Expected Output:    (img) Moon instead of sun and the whole area is much darker.
    #
    # Hint           :    See circle, bitwise_not in OPENCV
    #                :    Changing image brightness might be difficult with simple image manipulation
    #                     because of overflow issues. Try multiplication to avoid them.
    
    if debug:
        print_h("[Assignment]: Remove particular ROI(Sun) from an img(Sunset_img)\n")
    
    # [ Inputs you are provided ]
    # > Sunset image + Sun Location (x,y) + Sun radius
    day  = cv2.imread("Data/sunset.jpg")
    SUN_LOC = ( int(day.shape[1]/2) + 38 , int(day.shape[0]/2) )
    RAD = 300
    # > Moon Image + Moon Mask
    moon_w_tranparency = cv2.imread(r"Data\moon.png",cv2.IMREAD_UNCHANGED)
    moon = moon_w_tranparency[:,:,:3]
    moon_mask = moon_w_tranparency[:,:,3] # Extracting the alpha(mask) chnnel from a bgra image
    
    if debug:
        # Displaying Inputs
        imshow("day",day)
        imshow("moon_img",moon)
        imshow("moon_mask",moon_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Function that takes our sunset image and turns it into a gloomy solar eclipse
    eclipse = apocalypse(day,moon,moon_mask,SUN_LOC,RAD)
    
    if np.array_equal(day,eclipse):
        logger.error("apocalypse() needs to be coded to get the required(solar-eclipse) result.\n")
        exit(0)
    
    if debug:
        # Displaying the final output
        imshow("eclipse",eclipse)
        cv2.waitKey(0)
        
    return eclipse




def main():
    # Task : Simple Image Manipulation
    
    # a) Image creation using numpy
    print_h("1: Image creation using numpy")
    
    # a-1 : Zeros Image
    img_zeroes = np.zeros((10,10),np.uint8)
    print(f"\nimg_zeroes {img_zeroes.shape}\n",img_zeroes)
    images = []
    titles = []
    images.append(img_zeroes)
    titles.append("img_zeroes")
    
    # a-2 : Ones Image
    img_ones = np.ones_like(img_zeroes)
    print(f"\nimg_ones {img_zeroes.shape}\n",img_ones)
    img_white = img_ones*255
    print("\nimg_White.dtype",img_white.dtype)# check type of numpy array
    images.append(img_white)
    titles.append("img_white")
    
    # a-3 : Gray Image + Image duplication without modyfing the original
    img_gray = img_white - 128
    images.append(img_gray)
    titles.append("img_halfnhalf")
    
    # Task : Pixel-Wises Manipulation Manipulation
    print_h("2: Pixel-Wise Image Manipulation")
    
    # 0: Identity image creation using numpy
    img_eyes = np.eye(10,dtype=np.uint8)*255
    images.append(img_eyes)
    titles.append("img_eyes")
    
    # 1: Checking pixel value
    data = img_eyes[5,5]
    print("data = ",data)
    
    # 2: Changing pixel value
    img_eyes_pix_acces = img_eyes.copy()
    img_eyes_pix_acces[5,5] = 0
    images.append(img_eyes_pix_acces)
    titles.append("After Pixel Acces")
    
    # 4: Numpy array slicing
    img_eyes_slicing = img_eyes.copy()
    img_eyes_slicing[0:5,0:5] = 255
    images.append(img_eyes_slicing)
    titles.append("Modified Top Left Quarter")
    
    # 5: Use-Case: Get only the roi in an image (Image Cropping)
    print_h("3: Extracting/Modifying ROI in image")
    sunset_img = cv2.imread("Data/sunset.jpg")
    images.append(sunset_img)
    titles.append("sunset_img")
    
    rows, cols = sunset_img.shape[0:2]
    man_watching_sunset = sunset_img[int(rows*0.45):int(rows*0.75),int(cols*0.45):int(cols*0.75)]
    images.append(man_watching_sunset)
    titles.append("man_watching_sunset")
    
    # 6 Drawing a horizontal line intersecting the image center
    strt = (0,int(rows/2))
    end = (cols,int(rows/2))
    sunset_img_horizon= cv2.line(sunset_img.copy(),strt,end,(0,255,0),8)
    images.append(sunset_img_horizon)
    titles.append("sunset_img_horizon")
    
    montage = build_montages(images,None,None,titles,True,False)
    for img in montage:
        cv2.imshow("montage",img)
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # (build_montages) Clearing both images and titles lists for next montage
    images.clear()
    titles.clear()

    #=======================================================================================================
    # 7: Masking Operation     
    print_h("4: Masking ROI in Image")
    
    # 7-a: Create Mask using cv2.rectangle
    mask_roi = np.zeros((sunset_img.shape[0],sunset_img.shape[1]),np.uint8)
    rect_strt = (int(cols*0.45),int(rows*0.45))
    rect_end = (int(cols*0.75),int(rows*0.75))
    cv2.rectangle(mask_roi,rect_strt,rect_end,255,-1)
    images.append(mask_roi)
    titles.append("mask_roi")
    
    
    # 7-b: Perform masking using cv2.bitwise_and (mask parameter)
    sunset_img_roi = cv2.bitwise_and(sunset_img,sunset_img,mask=mask_roi)
    images.append(sunset_img_roi)
    titles.append("sunset_img_roi")
    
    
    
    # 7-c: Displaying bbox pts using putText on sunset_img_roi
    sunset_img_roi_disp = sunset_img_roi.copy()
    
    # Extract Mask extremas(4) from Rect Start&End(2) points
    off_col = 160; off_row = 80
    lft_col,top_row = rect_strt
    rgt_col,btm_row = rect_end
    
    toplft = (lft_col-off_col,top_row-off_row)
    toprgt = (  rgt_col + off_col , top_row  - off_row )
    btmrgt = (  rgt_col + off_col , btm_row  + off_row )
    btmlft = (  lft_col - off_col , btm_row  + off_row )
    
    # Displaying Extrema coordintaes as text on image
    orig_list = [toplft,toprgt,btmrgt,btmlft]# Clockwise
    txt_list = [str(orig_list[0]),str(orig_list[1]),str(orig_list[2]),str(orig_list[3])]
    putText_bbox(sunset_img_roi_disp,txt_list,orig_list)
    
    images.append(sunset_img_roi_disp)
    titles.append("sunset_img_roi_disp")
    
    
    # Building and displaying montage
    montages = build_montages(images,None,None,titles,True,True)
    for img in montages:
        cv2.imshow("image",img) # Show large image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    
    

if __name__ =="__main__":
    i_am_ready = False
    
    if i_am_ready:
        assignment()
    else:
        main()