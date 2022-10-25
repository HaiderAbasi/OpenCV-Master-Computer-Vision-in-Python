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
    author: Kyle Hounslow
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

# IP Basics
