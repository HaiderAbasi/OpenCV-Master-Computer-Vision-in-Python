from distutils.log import error
import cv2
from numpy import random
import numpy as np
import math
from loguru import logger
import sys
import time
import os
logger.remove()
logger.add(sink = sys.stderr, level="INFO")


# IP Basics
def print_h(str):
    print(f"\n##############################################\n{str}")

def imshow(img_name,img,image_shape = None,Window_flag = cv2.WINDOW_AUTOSIZE,boundary = 100,resize_to_default = False):
    """
    (displays complete image on the screen.)

    Parameters:
    img_name (String):    Name of the image
    img (numpy 3d array): Image

    Returns:
    None

    """
    #if img is None:
    #    print('\n[Error!] image is None, Check Origin!\n')
    #    exit()
    logger.debug(f"(Input) img = {img.shape}")
    if ((len(img.shape)==3) and (img.shape[2]==4)):
       # Pick up the alpha channel and delete from original
       alpha = (img[...,-1]/255.0)
       img = np.delete(img, -1, -1)

       # Promote greyscale image to RGB to make coding simpler
       if len(img.shape) == 2:
          img = np.stack((img,img,img))

       h, w, _ = img.shape

       # Make a checkerboard background image same size, dark squares are grey(102), light squares are grey(152)
       f = lambda i, j: 102 + 50*((i+j)%2)
       bg = np.fromfunction(np.vectorize(f), (16,16)).astype(np.uint8)

       # Resize to square same length as longer side (so squares stay square), then trim
       if h>w:
          longer = h
       else:
          longer = w
       bg = cv2.resize(bg, (longer,longer), interpolation=cv2.INTER_NEAREST)
       # Trim to correct size
       bg = bg[:h,:w]

       # Blend, using result = alpha*overlay + (1-alpha)*background
       img = (alpha[...,None] * img + (1.0-alpha[...,None])*bg[...,None]).astype(np.uint8)

    img_disp = img.copy()
    cv2.namedWindow(img_name,Window_flag)

    montage_rows = 1
    montage_cols = 1
    if image_shape == None:
        height,width = img.shape[0:2]
        screen_width = 1280 # cols Usage: https://stackoverflow.com/questions/3129322/how-do-i-get-monitor-resolution-in-python?page=1&tab=scoredesc#tab-top
        screen_height = 720 # rows
        ref_width  =  (screen_width - boundary) // montage_cols 
        ref_height =  (screen_height- boundary) // montage_rows 

        aspect_ratio = width/height
        aspect_ratio_ref = ref_width/ref_height

        computed_height = int(ref_width/aspect_ratio)
        computed_width= int(ref_height*aspect_ratio)

        if (width>height): # Strech or compress width to screen
            if (computed_height*montage_rows)<screen_height: # within allowed limit
                image_shape = (ref_width,computed_height)
            else:
                image_shape = (int(ref_height*aspect_ratio),ref_height)                       
        else:
            if (computed_width*montage_cols)<screen_width: # within allowed limit
                image_shape = (computed_width,ref_height) 
            else:
                image_shape = (ref_width,computed_height)
    if len(image_shape) != 2:
        raise Exception('image shape must be list or tuple of length 2 (rows, cols)')
    if (resize_to_default or (img.shape[0]>image_shape[1] or img.shape[1]>image_shape[0]) ):
        #img = cv2.resize(img, image_shape)
        img_disp = cv2.resize(img_disp, image_shape)

    logger.debug(f"(Output) img_disp = {img_disp.shape}")

    cv2.imshow(img_name,img_disp)    

def putText(img, text,org=(0, 0),font=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(0, 255, 0),thickness=1,color_bg=(0, 0, 0)):
    FONT_SCALE = 3e-3  # Adjust for larger font size in all images
    THICKNESS_SCALE = 2e-3  # Adjust for larger thickness in all images

    if fontScale==1:
        height, width= img.shape[:2]
        fontScale = min(width, height) * FONT_SCALE
        thickness = math.ceil(min(width, height) * THICKNESS_SCALE)

    x, y = org
    rect_x = x ; rect_y = y
    ext = 10
    if x > ext:
        rect_x = rect_x - ext
    if y > ext:
        rect_y = rect_y - ext

    text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
    text_w, text_h = text_size
    rect_w = text_w
    rect_h = text_h
    if img.shape[1] - (rect_x + text_w) > ext:
        rect_w = rect_w + ext
    if img.shape[0] - (rect_y + text_h) > ext:
        rect_h = rect_h + ext

    #cv2.rectangle(img, org, (x + text_w, y + text_h), color_bg, -1)
    cv2.rectangle(img, org, (rect_x + rect_w, rect_y + rect_h), color_bg, -1)
    cv2.putText(img, text, (x,int( y + text_h + fontScale - 1)), font, fontScale, color, thickness)

def build_montages(image_list, image_shape=None, montage_shape=None,titles=[],resize_to_default = True,draw_borders = False):
    """
    ---------------------------------------------------------------------------------------------
    ##### author: Kyle Hounslow - modified by: Haider Abbasi
    ---------------------------------------------------------------------------------------------
    Converts a list of single images into a list of 'montage' images of specified rows and columns.
    A new montage image is started once rows and columns of montage image is filled.
    Empty space of incomplete montage images are filled with black pixels
    ---------------------------------------------------------------------------------------------
    :param image_list: python list of input images
    :param image_shape: tuple, size each image will be resized to for display (width, height)
    :param montage_shape: tuple, shape of image montage (width, height)
    :return: list of montage images in numpy array format
    ---------------------------------------------------------------------------------------------

    example usage:

    # load single image
    img = cv2.imread('lena.jpg')
    # duplicate image 25 times
    num_imgs = 25
    img_list = []
    for i in xrange(num_imgs):
        img_list.append(img)
    # convert image list into a montage of 256x256 images tiled in a 5x5 montage
    montages = make_montages_of_images(img_list, (256, 256), (5, 5))
    # iterate through montages and display
    for montage in montages:
        cv2.imshow('montage image', montage)
        cv2.waitKey(0)

    ----------------------------------------------------------------------------------------------
    """
    if montage_shape == None:
        montage_rows = (len(image_list)//3) + 1 if (len(image_list)%3)!=0 else (len(image_list)//3)
        if len(image_list)<3:
            montage_cols = len(image_list)
        else:
            montage_cols = 3

        montage_shape = (montage_cols,montage_rows)

    if image_shape == None:
        default_img = image_list[0]
        logger.debug(f"Image shape (Start) = {default_img.shape}")
        height,width = default_img.shape[0:2]
        screen_width = 1280 # cols
        screen_height = 720 # rows
        boundary = 20
        ref_width  =  (screen_width - boundary) // montage_cols 
        ref_height =  (screen_height- boundary) // montage_rows 

        aspect_ratio = width/height
        aspect_ratio_ref = ref_width/ref_height

        computed_height = int(ref_width/aspect_ratio)
        computed_width= int(ref_height*aspect_ratio)

        if (width>height): # Strech or compress width to screen
            if (computed_height*montage_rows)<screen_height: # within allowed limit
                image_shape = (ref_width,computed_height)
            else:
                image_shape = (int(ref_height*aspect_ratio),ref_height)                       
        else:
            if (computed_width*montage_cols)<screen_width: # within allowed limit
                image_shape = (computed_width,ref_height) 
            else:
                image_shape = (ref_width,computed_height)

        logger.debug(f"Required Image shape = {(image_shape[1],image_shape[0])}")

      

        # ref_width = 640
        # ref_height = 480
        # if ((height>=ref_height) or (width>=ref_width)): # We have image larger then our screen 
        #     #aspect_ratio = width/height
        #     size_to_ref = width//ref_width if width>height else height//ref_height
        #     image_shape = (int(width/size_to_ref),int(height/size_to_ref))
        # else: 
        #     image_shape = (image_list[0].shape[1],image_list[0].shape[0]) # Use shape of first image in images


    if len(image_shape) != 2:
        raise Exception('image shape must be list or tuple of length 2 (rows, cols)')
    if len(montage_shape) != 2:
        raise Exception('montage shape must be list or tuple of length 2 (rows, cols)')
    
    
    image_montages = []

    # start with black canvas to draw images onto
    montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                          dtype=np.uint8)
    cursor_pos = [0, 0]
    start_new_img = False
    for idx,img in enumerate(image_list):
        #cv2.imshow(f"Current Img {idx}",img)
        if type(img).__module__ != np.__name__:
            raise Exception('input of type {} is not a valid numpy array'.format(type(img)))

        if ( (len(img.shape)<3) or (img.shape[2]<3) ):
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        start_new_img = False
        bg = np.zeros((image_shape[1],image_shape[0],3),np.uint8)
        if (resize_to_default or (img.shape[0]>image_shape[1] or img.shape[1]>image_shape[0]) ):
            img = cv2.resize(img, image_shape)
            logger.debug(f"Image Resized (No background overlaying)")
        else:
            logger.debug(f"Image Not Resized (Overlayed on bg the size of estimated image Shapes)")
            rows,cols = img.shape[0:2]
            row_half = int(rows/2)
            col_half = int(cols/2)
            c_r = int(int(image_shape[1])/2)
            c_c = int(int(image_shape[0])/2)
            #bg[c_r-row_half:c_r+row_half,c_c-col_half:c_c+col_half] = img
            bg[c_r-row_half:c_r-row_half+img.shape[0],c_c-col_half:c_c-col_half+img.shape[1]] = img
            img = bg
        
        logger.debug(f"Image shape (END) = {img.shape}")


        if draw_borders:
            cv2.rectangle(img,(0,0),(img.shape[1],img.shape[0]),(255,255,255),1)
        if len(titles)!=0:
            #print("img.shape =",img.shape)
            if ( (len(img.shape)!=3) or (img.shape[2]==1) ):
                txt_color =  0 if img[0][0] > 128 else 255
            else:
                txt_color = (255,255,255)
            putText(img,titles[idx],(20,20),cv2.FONT_HERSHEY_PLAIN,1,txt_color,1)

        # draw image to black canvas
        montage_image[cursor_pos[1]:cursor_pos[1] + image_shape[1], cursor_pos[0]:cursor_pos[0] + image_shape[0]] = img
        cursor_pos[0] += image_shape[0]  # increment cursor x position
        if cursor_pos[0] >= montage_shape[0] * image_shape[0]:
            cursor_pos[1] += image_shape[1]  # increment cursor y position
            cursor_pos[0] = 0
            if cursor_pos[1] >= montage_shape[1] * image_shape[1]:
                cursor_pos = [0, 0]
                image_montages.append(montage_image)
                # reset black canvas
                montage_image = np.zeros(shape=(image_shape[1] * (montage_shape[1]), image_shape[0] * montage_shape[0], 3),
                                      dtype=np.uint8)
                start_new_img = True
    if start_new_img is False:
        image_montages.append(montage_image)  # add unfinished montage
    return image_montages

def get_optimal_font_scale(text, width):
    """ Returns the optical font Scale for text to fit in the rectangle 

    Args:
        text (String): _description_
        width (int): _description_

    Returns:
        int: _description_
    """
    
    for scale in reversed(range(0, 120, 1)):
        textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=scale/10, thickness=1)
        new_width = textSize[0][0]
        #print(new_width)
        if (new_width <= int(width/2)):
            return (scale/10),new_width
    return 1,1

def putText_bbox(img,text_list,orig_list,type = "bbox"):
    
    fontScale,new_width = get_optimal_font_scale(text_list[0],img.shape[1])

    clr_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,255)]
        
    for idx,txt in enumerate(text_list):
        org = orig_list[idx]
        
        if type=="bbox":
            clr = clr_list[idx]
            if idx==0:
                # is even===> then adjust for pt center
                #print(f"adjusted .... YAyyy! at {idx}")
                org = (org[0]-int(new_width/2),org[1])
            elif idx==3:
                # is even===> then adjust for pt center
                #print(f"adjusted .... YAyyy! at {idx}")
                org = (org[0]-int(new_width/2),org[1])
        else:
            clr = random.random(size=3) * 256

        #cv2.putText(img,txt,org,cv2.FONT_HERSHEY_PLAIN,fontScale,clr,4 )
        putText(img,txt,org,cv2.FONT_HERSHEY_PLAIN,1,clr,1.5)
# IP Basics

# CV 101
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
ORANGE = (0,140,255)
BROWN = (42,42,165)

dark_colors = [RED,GREEN,BLUE,ORANGE,BROWN]

def draw_points(image,pts,radius=2):
    if len(image.shape)<3:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for idx,pt in enumerate(pts):
        if type(pt)!= tuple:
            pt = tuple(pt)
        rand_color = dark_colors[idx]
        cv2.circle(image,pt,radius,rand_color,2)
    return image


def describe(something,title="something",display_content = False):
    """convenience function to get knowledge about something

    Args:
        something (Any): Any type input that you want to know detail characteristics os
    """    
    print(f"\nDescribing {title}......")
    t_s = type(something)
    print(f"type = {t_s}")
    
    if (t_s=="String"):
        print(f"No of chars = {len(something)}")
    #elif (t_s=="<class 'numpy.ndarray'>"):
    elif isinstance(something, np.ndarray):
        print(f"numpy_array.dtype = {something.dtype}")
        print(f"numpy_array.shape = {something.shape}")
    elif (t_s=="<class 'tuple'>" or t_s=="<class 'list'>"):
        print(f"No of elements = {len(something)}")

    if display_content:
        print(f"Here you go = {something}")
    #Min = min(something)
    Min = np.amin(something)
    Max = np.amax(something)
    #Max = max(something)
    print(f"Range = ({Min}<---->{Max})\n")
        


class debugger:

    def __init__(self,window = "Control",trackbars_list=None,max_list=None,odd_list = None):

        self.window = window
        
        self.trackbars_list = trackbars_list
        self.no_of_trackbars = len(trackbars_list)
        self.debug_vars = [0] * self.no_of_trackbars
        self.odd_list = odd_list 
        if self.odd_list!= None:
            idces_to_modify_max = [i for i,v in enumerate(self.odd_list) if v]
            for idx in idces_to_modify_max:
                max_list[idx] = max_list[idx]//2
            print(f"max_list = {max_list}")
        
        cv2.namedWindow(window,cv2.WINDOW_NORMAL)
        for idx in range(self.no_of_trackbars):
            cv2.createTrackbar(trackbars_list[idx],self.window,self.debug_vars[idx],max_list[idx],self.nothing)


    @staticmethod
    def nothing():
        pass

    def update_variables(self):
        for idx in range(self.no_of_trackbars): 
            if ( (self.odd_list!= None) and (self.odd_list[idx]) ): # Incase of odd trackbar requested
                curr_pos = cv2.getTrackbarPos(self.trackbars_list[idx],self.window)
                self.debug_vars[idx] = (curr_pos)*2 + 1
                self.debug_vars[idx] = self.debug_vars[idx] if (self.debug_vars[idx]>=3) else 3
            else:
                self.debug_vars[idx] = cv2.getTrackbarPos(self.trackbars_list[idx],self.window)




class Gui():

    def __init__(self):
        self.pt = None # point instance variable

        # ADVANCED #
        self.img_draw = None
        self.ix,self.iy = -1,-1
        self.fx,self.fy = -1,-1
        self.roi_confirmed = False
        self.selected_rois = []


    # mouse callback function
    def __selectroi_callback(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ix,self.iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.roi_confirmed:
                cv2.rectangle(self.img_draw,(self.ix,self.iy),(self.fx,self.fy),(0,255,0),2)
                if self.ix <= self.fx :
                    strt_col = self.ix
                    width = self.fx - self.ix
                else:
                    strt_col = self.fx
                    width = self.ix - self.fx
                if self.iy <= self.fy:
                    strt_row = self.iy
                    height = self.fy - self.iy
                else:
                    strt_row = self.fy
                    height = self.iy - self.fy
                self.selected_rois.append((strt_col,strt_row,width,height))
                self.roi_confirmed = False
                
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.rectangle(self.img_draw,(self.ix,self.iy),(x,y),(0,140,255),2)
            self.fx = x
            self.fy = y

    def selectROIs(self,img,title = 'SelectROIs'):
        self.img_draw = img.copy()# Dont want to mess up the original XD
        cv2.namedWindow(title)
        cv2.setMouseCallback(title,self.__selectroi_callback)

        while(1):
            cv2.imshow(title,self.img_draw)
            k = cv2.waitKey(1) & 0xFF
            if k == 13:# Enter
                self.roi_confirmed = True
            elif k == 27:
                break
        cv2.destroyAllWindows()
        #print("selected_rois = ",self.selected_rois)
    
    @staticmethod
    def __nothing(val):
        pass

    def selectdata(self,filenames,Window_Name ="Data Selection",trackbar_name = "choice",useMouse = False,onTop =False,data_type = "test data"):
        
        user_choice = 0

        curr_x = 0
        curr_y = 0
        data_selected = False
        def onMouse(event,x,y,flags,param): # Inner function
            nonlocal curr_x,curr_y,data_selected
            if event == cv2.EVENT_LBUTTONDOWN:
                curr_x,curr_y = x,y
            elif event == cv2.EVENT_LBUTTONUP:
                data_selected = True

        def getUserChoice():
            nonlocal unit_row,shift,curr_y
            #curr_y = unit_row+(shift*user_choice)
            user_choice = ( curr_y - unit_row ) / shift
            return user_choice

        cv2.namedWindow(Window_Name)
        if onTop:
            cv2.setWindowProperty(Window_Name, cv2.WND_PROP_TOPMOST, 1) # Reference: https://stackoverflow.com/a/66364178/11432131
        hp_col = 700
        filename_w = 25
        hp_row = 40 + filename_w*(len(filenames))

        home_page = np.zeros((hp_row,hp_col,3),np.uint8)#BGR
        txt_to_display = f"Please choose one of the following as the {data_type}...."
        cv2.putText(home_page,txt_to_display,(20,20),cv2.FONT_HERSHEY_PLAIN,1.2,(0,140,255),1)
        
        col = 20
        unit_row = 50
        shift = 25
        for idx,filename in enumerate(filenames):
            txt_to_display = f"{idx}: {filename}" 
            cv2.putText(home_page,txt_to_display,(col,unit_row+(shift*idx)),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),1)
        cv2.createTrackbar(trackbar_name,Window_Name,user_choice,len(filenames)-1,self.__nothing)

        cv2.setMouseCallback(Window_Name,onMouse)


        prev_txt_to_display = ""
        prev_user_choice = 0
        while(1):
            if useMouse:
                choice = getUserChoice()
                if choice>0:
                    user_choice = round(choice)
                else:
                    user_choice = 0 # If it is less then zero or not choose. Consider it default : 0
                if data_selected:
                    print(">>>>> User_choice = ",user_choice)
                    cv2.waitKey(0)
            else:
                user_choice = cv2.getTrackbarPos(trackbar_name,Window_Name)

            txt_to_display = f"{user_choice}: {filenames[user_choice]}" 
            cv2.putText(home_page,prev_txt_to_display,(col,unit_row+(shift*prev_user_choice)),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),1)
            cv2.putText(home_page,txt_to_display,(col,unit_row+(shift*user_choice)),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),1)
            prev_txt_to_display = txt_to_display
            prev_user_choice = user_choice

            cv2.imshow(Window_Name,home_page)
            k = cv2.waitKey(1) & 0xFF
            if ( (k == 13) or data_selected ):# Enter or data choosen by mouses
                if not data_selected:
                    cv2.destroyWindow(Window_Name)# No longer need it       
                break # Exit loop
            elif k==27: #Esc pressed -> Exiting...
                user_choice = -1
                break

        return user_choice

    # ADVANCED #



    def ret_point(self,event,x,y, flags, param):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            self.pt = (x,y)

    def select_cnt(self,img,cnts,Loop=False):
        cv2.namedWindow("Select Contour")
        cv2.imshow("Select Contour",img)
        cv2.setMouseCallback("Select Contour",self.ret_point)
        matched_cnts = []
        matched_cnts_idx = []
        while(1):
            # Wait for User to Select Contour
            
            if self.pt!=None:
                for idx,cnt in enumerate(cnts):
                    ret = cv2.pointPolygonTest(cnt,self.pt,False)
                    if ret==1: # point is inside a contour ==> return its idx and cnt
                        matched_cnts.append(cnt)
                        matched_cnts_idx.append(idx)

                if matched_cnts==[]:
                    logger.warning("(Incorrect Selection) --> Please only select an object!!")
                    self.pt = None # Reset point to None 
                elif len(matched_cnts) == 1:
                    cnt = matched_cnts[0]
                    idx = matched_cnts_idx[0]
                    if not Loop:
                        cv2.destroyWindow("Select Contour")
                    self.pt = None # Reset point to None                         
                    return idx,cnt
                else:
                    # find the biggest countour (c) by the area
                    cnt = min(matched_cnts, key = cv2.contourArea)
                    idx = matched_cnts.index(cnt)
                    if not Loop:
                        cv2.destroyWindow("Select Contour")
                    self.pt = None # Reset point to None                         
                    return idx,cnt

            k = cv2.waitKey(1)
            if k==27:
                logger.warning("(Esc key pressed) --> No contours selected + Exiting!!!")
                return -1,-1 # idx = -1 (indicating no contour and exiting!)


def noisy(noise_typ,image):
    # > Adds gaussian , salt-pepper , poisson and speckle noise in an image
    # Parameters
    # ----------
    # image : ndarray
    #     Input image data. Will be converted to float.
    # mode : str
    #     One of the following strings, selecting the type of noise to add:

    #     'gauss'     Gaussian-distributed additive noise.
    #     'poisson'   Poisson-distributed noise generated from the data.
    #     's&p'       Replaces random pixels with 0 or 1.
    #     'speckle'   Multiplicative noise using out = image + n*image,where
    #                 n is uniform noise with specified mean & variance.
    
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy    

# CV 101