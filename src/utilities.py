import cv2
from numpy import random
import numpy as np
import math
from loguru import logger
import sys
import time
import os
from collections import deque
import subprocess
import gdown
import zipfile


def download_missing_recog_data(recog_dir):
    """Download missing model files from Google Drive."""
    models_dir = os.path.join(recog_dir, 'recog_data', 'models')
    model_files = ['lbfmodel.yaml']  # replace with actual file names
    
    files_id = ['1gipWgZMM14ZlHjvhFSli2fqwwGyYZiyN']  # replace with actual file IDs or URLs
    
    # Create model directory if it doesnot exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    for i, file in enumerate(model_files):
        file_path = os.path.join(models_dir, file)
        if not os.path.exists(file_path):
            print(f'{file} not found. Downloading...')
            file_id = files_id[i]  # replace with the actual file ID or URL
            if file_id.startswith('http'):
                # Use curl to download the file
                subprocess.run(['curl', '-L', file_id, '-o', file_path], check=True)
            else:
                # Use gdown to download the file from Google Drive
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, file_path, quiet=False)
            print(f'{file} downloaded successfully!')


def download_missing_training_data(recog_dir,verbose=0):
    """Download missing model files from Google Drive."""
    models_dir = os.path.join(recog_dir, 'recog_data')
    model_files = ['training.zip']  # replace with actual file names
    files_id = ['1IsEjCloWgBKNp0GxpbXj9q9L2_6eXuxW']  # replace with actual file IDs or URLs
    
    # Create model directory if it doesnot exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    for i, file in enumerate(model_files):
        file_path = os.path.join(models_dir, file)
        if not os.path.exists(file_path):
            print(f'{file} not found. Downloading...')
            file_id = files_id[i]  # replace with the actual file ID or URL
            # Use gdown to download the file from Google Drive
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, file_path, quiet=False)
            if verbose:
                print(f'{file} downloaded successfully!')
            # Extract the downloaded zip file
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(models_dir)
            if verbose:
                print(f'{file} extracted successfully!')
            # Delete the zip file after extraction
            os.remove(file_path)
            if verbose:
                print(f'{file} deleted successfully!')


use_optional = False


if use_optional:
    try:
        from sklearn.cluster import KMeans
        from skimage.feature import local_binary_pattern
        print("Importing kmeans for unsupervised data sorting...")
    except ImportError:
        print("sklearn not installed!")



logger.remove()
logger.add(sink = sys.stderr, level="INFO")

#from skimage import morphology

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

def putText(img, text,org=(0, 0),font=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(0, 255, 0),thickness=1,color_bg=(0, 0, 0),bbox_size = None):
    FONT_SCALE = 2e-3  # Adjust for larger font size in all images
    THICKNESS_SCALE = 2e-3  # Adjust for larger thickness in all images

    if fontScale==1:
        if bbox_size == None:
            height, width= img.shape[:2]
        else:
            FONT_SCALE = 1e-2
            THICKNESS_SCALE = 2e-3
            width,height = bbox_size
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
    if ( ((img.shape[1] - (rect_x + text_w)) > ext) and (x > ext) ):
        rect_w = rect_w + ext
    if ( ((img.shape[0] - (rect_y + text_h)) > ext) and (y > ext) ):
        rect_h = rect_h + ext

    #cv2.rectangle(img, org, (x + text_w, y + text_h), color_bg, -1)
    cv2.rectangle(img, org, (rect_x + rect_w, rect_y + rect_h), color_bg, -1)
    cv2.putText(img, text, (x,int( y + text_h + fontScale - 1)), font, fontScale, color, thickness)

def build_montages(image_list, image_shape=None, montage_shape=None,titles=[],resize_to_default = True,draw_borders = False,title_at_end = False):
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
            
            title_loc = (20,20)
            if title_at_end:
                i_r,i_c = img.shape[0:2]
                title_loc = ( int((2*i_c)/5) + 20 , i_r - 20 )
            putText(img,titles[idx],title_loc,cv2.FONT_HERSHEY_PLAIN,1,txt_color,1)

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

def get_circular_regions(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img_circles = img.copy()

    NumOfVotesForCircle = 40 #parameter 1 MinVotes needed to be classified as circle
    CannyHighthresh = 130 # High threshold value for applying canny
    mindDistanBtwnCircles = 10 # kept as sign will likely not be overlapping
    max_rad = 200 # smaller circles dont have enough votes so only maxRadius need to be controlled 
                    # As signs are right besides road so they will eventually be in view so ignore circles larger than said limit

    mask_circles = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    # 4. Detection (Localization)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,mindDistanBtwnCircles,param1=CannyHighthresh,param2=NumOfVotesForCircle,minRadius=16,maxRadius=max_rad)
    # 4a. Detection (Localization) Checking if circular regions were localized
    circless = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # 4b. Detection (Localization) Looping over each localized circle
        for i in circles[0,:]:
            center =(i[0],i[1])
            radius = i[2]
            circless.append([center,radius])
            #cv2.circle(img_circles,center,radius,(120,128,255),4)
            cv2.circle(mask_circles,center,radius,255,-1)
    
    return mask_circles,circless

def get_rois_mask(img_smarties):
    # hls = cv2.cvtColor(img_smarties,cv2.COLOR_BGR2HLS)
    # hue = hls[:,:,0]
    
    # edges = cv2.Canny(hue,50,150,None,3)
    # cv2.findContours(edges,cv2.RETR_EXTERNAL)
    # Generating ground replica 
    
    
    base_clr = img_smarties[0][0]
    Ground_replica = np.ones_like(img_smarties)*base_clr
    
    # Step 2: Foreground Detection
    change = cv2.absdiff(img_smarties, Ground_replica)
    change_gray = cv2.cvtColor(change, cv2.COLOR_BGR2GRAY)
    change_mask = cv2.threshold(change_gray, 15, 255, cv2.THRESH_BINARY)[1]
    
    return change_mask

def get_centroids(bboxes):
    centers = []
    for bbox in bboxes:
        centers.append(find_centroid(bbox,"ltwh"))
    return centers


def get_centroid(cnt):
    M = cv2.moments(cnt)
    if M['m00']==0: # If its a line (No Area) then use minEnclosingcircle and use its center as the centroid
        (cx,cy) = cv2.minEnclosingCircle(cnt)[0]        
        return (int(cx),int(cy))
    else:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx,cy)

# [NEW]: Find closest point in a list of point to a specific position
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=(nodes.ndim-1))
    return np.argmin(dist_2)

def euc_dist(a,b):
    return math.sqrt( ( (a[1]-b[1])**2 ) + ( (a[0]-b[0])**2 ) )


def keep_blobs_by_mask(img,mask):
    # Algo: 
    # Preprocess: Delete blobs too small in img
    # 1) Find contours in image
    # 2) Loop over each, calculate bbox.
    #    In each rect, check in the mask.
    #                       if any():
    #                                intersects: appemd cnt to list of valid cnts
    # 3) Draw valid cnts on new image
    cnts = cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    valid_cnts = []
    valid_centroids = []
    for cnt in cnts:
        r = cv2.boundingRect(cnt)
        x,y,w,h = r
        if (mask[y:y+h,x:x+w]).any():
            valid_cnts.append(cnt)
            valid_centroids.append(get_centroid(cnt))
    
    blobsbymask = np.zeros_like(mask)
    if valid_cnts!=[]:
        cv2.drawContours(blobsbymask,valid_cnts,-1,255,-1)
        
    return blobsbymask,valid_cnts,valid_centroids
   

def ApproxDistBWCntrs(cnt,cnt_cmp):
    # compute the center of the contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    # compute the center of the contour
    M_cmp = cv2.moments(cnt_cmp)
    cX_cmp = int(M_cmp["m10"] / M_cmp["m00"])
    cY_cmp = int(M_cmp["m01"] / M_cmp["m00"])
    minDist=euc_dist((cX,cY),(cX_cmp,cY_cmp))
    Centroid_a=(cX,cY)
    Centroid_b=(cX_cmp,cY_cmp)
    return minDist,Centroid_a,Centroid_b

def RetLargestContour(gray,cnts = None):
    if cnts is None:
        thresh=np.zeros(gray.shape,dtype=gray.dtype)
        _,bin_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)
        #Find the two Contours for which you want to find the min distance between them.
        cnts = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    else:
        # We would have already had the thresh to extract contours from them
        thresh = gray
    Max_Cntr_area = 0
    Max_Cntr_idx= -1
    for index, cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        if area > Max_Cntr_area:
            Max_Cntr_area = area
            Max_Cntr_idx = index

    Largest_cnt = []
    if (Max_Cntr_idx!=-1):
        thresh = cv2.drawContours(thresh, cnts, Max_Cntr_idx, (255,255,255), -1) # [ contour = less then minarea contour, contourIDx, Colour , Thickness ]
        Largest_cnt = cnts[Max_Cntr_idx]
    
    return thresh, Largest_cnt

     
def extract_blobs_on_pattern(BW,MaxDistance):
    """Estimate the mid-lane trajectory based on the detected midlane (patches) mask

    Args:
        BW (numpy_1d_array): Midlane (patches) mask extracted from the GetLaneROI()
        MaxDistance (int): max distance for a patch to be considered part of the midlane 
                                      else it is noise

    Returns:
        numpy_1d_array: estimated midlane trajectory (mask)
    """
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    BW = cv2.morphologyEx(BW,cv2.MORPH_DILATE,kernel)
    #cv2.namedWindow("BW_zero",cv2.WINDOW_NORMAL)
    BW_zero= cv2.cvtColor(BW,cv2.COLOR_GRAY2BGR)

    # 1. Find the two Contours for which you want to find the min distance between them.
    cnts = cv2.findContours(BW, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]#3ms
    
    # 2. Keep Only those contours that are not lines 
    MinArea=(BW.shape[0]/30)*(BW.shape[0]/30)
    cnts_Legit=[]
    for index, _ in enumerate(cnts):
        area = cv2.contourArea(cnts[index])
        if area > MinArea:
            cnts_Legit.append(cnts[index])
    cnts = cnts_Legit

    # Cycle through each point in the Two contours & find the distance between them.
    # Take the minimum Distance by comparing all other distances & Mark that Points.
    CntIdx_BstMatch = []# [BstMatchwithCnt0,BstMatchwithCnt1,....]
    Cnts_clst_dist = []
    # 3. Connect each contous with its closest 
    for index, cnt in enumerate(cnts):
        prevmin_dist = 100000 ; Bstindex_cmp = 0 ; BstCentroid_a=0  ; BstCentroid_b=0      
        for index_cmp in range(len(cnts)-index):
            index_cmp = index_cmp + index
            cnt_cmp = cnts[index_cmp]
            if (index!=index_cmp):
                min_dist,Centroid_a,Centroid_b  = ApproxDistBWCntrs(cnt,cnt_cmp)

                #Closests_Pixels=(cnt[min_dstPix_Idx[0]],cnt_cmp[min_dstPix_Idx[1]])
                if(min_dist < prevmin_dist):
                    if (len(CntIdx_BstMatch)==0):
                        prevmin_dist = min_dist
                        Bstindex_cmp = index_cmp
                        #BstClosests_Pixels = Closests_Pixels
                        BstCentroid_a = Centroid_a
                        BstCentroid_b = Centroid_b   

                    else:
                        Present= False
                        for i in range(len(CntIdx_BstMatch)):
                            if ( (index_cmp == i) and (index == CntIdx_BstMatch[i]) ):
                                Present= True
                        if not Present:
                            prevmin_dist = min_dist
                            Bstindex_cmp = index_cmp
                            #BstClosests_Pixels = Closests_Pixels
                            BstCentroid_a = Centroid_a
                            BstCentroid_b = Centroid_b
   
        if ((prevmin_dist!=100000 ) and (prevmin_dist>MaxDistance)):
            break
        if (type(BstCentroid_a)!=int):
            CntIdx_BstMatch.append(Bstindex_cmp)
            Cnts_clst_dist.append(prevmin_dist)
            cv2.line(BW_zero,BstCentroid_a,BstCentroid_b,(0,255,0),thickness=2)
    
    BW_gray = cv2.cvtColor(BW_zero,cv2.COLOR_BGR2GRAY)

    # 4. Get estimated midlane by returning the largest contour 
    BW_Largest,Largest_cnt = RetLargestContour(BW_gray)#3msec

    # 5. Return Estimated Midlane if found otherwise send original
    if Largest_cnt is not None:
        return BW_Largest,BW_zero,cnts,CntIdx_BstMatch,Cnts_clst_dist
    else:
        return BW,BW_zero,cnts,CntIdx_BstMatch,Cnts_clst_dist


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
        
    thickness = 2 if image.shape[0]<1280 else int(2 * (image.shape[0]/640))
    for idx,pt in enumerate(pts):
        if type(pt)!= tuple:
            pt = (int(pt[0]),int(pt[1]))
        rand_color = dark_colors[idx]
        cv2.circle(image,pt,radius,rand_color,thickness)
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
        
        self.clicked_pt = []


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
        print("selected_rois = ",self.selected_rois)
        return self.selected_rois
    
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

    def __save_clicked_point(self,event, x, y, flags, param):
        """
        Saves the coordinates of the point where the user clicked.
        
        Parameters:
        event (int): Event type, such as mouse button release
        x (int): X-coordinate of the clicked point
        y (int): Y-coordinate of the clicked point
        flags (int): Additional parameters for the event, unused
        param (list): List to store the clicked point, updated in-place
        
        Returns:
        None
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            #print(f"User clicked on ({x},{y})")
            self.clicked_pt.append((x, y))

    def select_pt(self,Window_Name ="Point Selection"):
        cv2.namedWindow(Window_Name,cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(Window_Name, self.__save_clicked_point)



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

# Advanced

def random_bright_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def disp_fps(frame,start_time):

    FONT_SCALE = 3e-3
    THICKNESS_SCALE = 1e-3

    height, width= frame.shape[:2]
    
    font_scale = min(width, height) * FONT_SCALE
    thickness = math.ceil(min(width, height) * THICKNESS_SCALE)

    fps_txt = f"FPS: = {1.0 / (time.time() - start_time):.2f}"
    putText(frame, fps_txt, org=(10, 20),fontScale=font_scale)
    
def disp_Fps(frame,processing_times,start_frame_time,clr = (0,255,0)):
    end_frame_time = time.perf_counter()
    processing_times.append(end_frame_time - start_frame_time)
    # Calculate the average processing time
    average_processing_time = sum(processing_times) / processing_times.maxlen
    # Calculate the FPS
    fps_txt = "FPS: {:.2f}".format(1/average_processing_time)
    putText(frame, f"{fps_txt}",(20,20),color=clr)


def get_fileName(file_path):
    if not os.path.isfile(file_path):
        print(f"Not a file path...{file_path}\nCheck again!")
        return None

    file_basename = os.path.basename(file_path)
    fileName = os.path.splitext(file_basename)[0]
    return fileName
    
def get_data(topic, type = "img", folder_dir = None):

    if folder_dir==None:
        if topic =="tracking":
            folder_dir = "Data/Advanced/Tracking/test_videos"
            fomrats = (".mp4", ".avi", ".mov", ".mpeg", ".flv", ".wmv",".webm")
        elif topic =="multitracking":
            folder_dir = "Data/Advanced/Tracking/multi_test_videos"
            fomrats = (".mp4", ".avi", ".mov", ".mpeg", ".flv", ".wmv",".webm")
        elif topic =="deepsort":
            folder_dir = "Data/Advanced/Tracking/deepsort"
            fomrats = (".mp4", ".avi", ".mov", ".mpeg", ".flv", ".wmv",".webm")


    data_dirs = []
    filenames = []
    for file in os.listdir(folder_dir):
        #if file.endswith(".jpg"):
        if file.endswith(fomrats):
            filename, _ = os.path.splitext(file)
            file_path = os.path.join(folder_dir, file)
            data_dirs.append(file_path)
            filenames.append(filename)
    return data_dirs,filenames

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {0, 2, 1, 3}
        The (0, 1) position is at the top left corner,
        the (2, 3) position is at the bottom right corner
    bb2 : dict
        Keys: {0, 2, 1, 3}
        The (x, y) position is at the top left corner,
        the (2, 3) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def to_ltrd(ltwh):
    #  __ _ _ _ _ _  X axis
    # |     x1  y1
    # |   (left,top) ---------
    # |              |       |
    # |              |       |
    # |              --------- (right,down)
    #  Y axis                     x2   y2
    #           
    left, top, w, h = ltwh
    right = left + w
    down = top + h
    # Returns (x1, y1, x2  , y2)
    return ((left,top,right,down))


# DeepSort Additions 
def generate_vibrant_color():
    
    def hsv_to_rgb(h, s, v):
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        if h < 1/6:
            r, g, b = c + m, x + m, 0 + m
        elif h < 2/6:
            r, g, b = x + m, c + m, 0 + m
        elif h < 3/6:
            r, g, b = 0 + m, c + m, x + m
        elif h < 4/6:
            r, g, b = 0 + m, x + m, c + m
        elif h < 5/6:
            r, g, b = x + m, 0 + m, c + m
        else:
            r, g, b = c + m, 0 + m, x + m
        return int(r * 255), int(g * 255), int(b * 255)

    h = random.uniform(0, 1)
    s = random.uniform(0.8, 1)
    v = random.uniform(0.8, 1)
    vibrant_clr = hsv_to_rgb(h,s,v)
    return vibrant_clr

def find_centroid(bbox, bbox_type="ltrd"):
    """
    This function computes the centroid of a bounding box given its coordinates in ltrd or ltwh format.

    Parameters:
    bbox (tuple): A tuple of 4 values (x1, y1, x2, y2) or 3 values (x, y, w, h) that define the bounding box coordinates.

    bbox_type (str, optional): The type of bounding box, default is "ltrd".

    Returns:
    tuple: A tuple of 2 values (x_center, y_center) representing the x and y coordinates of the centroid.

    Raises:
    ValueError: If an unsupported bounding box type is provided.
    """

    if bbox_type == "ltrd":
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        return (int(x_center), int(y_center))
    elif bbox_type == "ltwh":
        x, y, w, h = bbox
        x_center = x + w / 2
        y_center = y + h / 2
        return (int(x_center), int(y_center))
    else:
        raise ValueError("[Unsupported bbox_type]: Please provide an 'ltrd' or 'ltwh' bbox!")


def closest_bbox_to_pt(point, bboxes):
    """
    - Finds the closest bounding box center to a given point.

    Parameters:
    point (tuple): A 2D point represented as (x, y)
    bboxes (list): List of bounding boxes represented as [(x1, y1, x2, y2), ...]

    Returns:
    tuple: The closest bounding box represented as (x1, y1, x2, y2) and its index.

    """
    # Calculate center of bounding boxes
    bbox_centers = [(bbox[:2] + (bbox[2:] - bbox[:2]) / 2) for bbox in bboxes]
    # Calculate distances between point and bounding box centers
    distances = [np.linalg.norm(point - bbox_center) for bbox_center in bbox_centers]
    # Find index of closest bounding box center
    closest_idx = np.argmin(distances)
    # Return the closest bounding box and its index
    return bboxes[closest_idx], closest_idx

def add_to_dict_deque(d, key, value,dq_len = 2):
    """
    - Append to dictionary of deques if a key already exists 
      
        OR 
    
        Add a new element deque(length) at location key

    Parameters:
    d (dict): The dictionary where the deque is stored.
    key (hashable): The key under which the deque is stored.
    value (Any): The value to be added to the deque.

    Returns:
    None
    """
    # check if the key is already present in the dictionary
    if key in d:
        # if yes, append the value to the deque
        d[key].append(value)
    else:
        # if not, create a new deque with length 2 and store it in the dictionary
        d[key] = deque([value], maxlen=dq_len)

class dataextractor:
    
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.img_sz = 100
        
        self.debug = False
        
        self.save_dir = None
    

    def extract_data(self,vid_path, data_to_extract = "faces", save_dir = "",skip_frames = 5):
        
        filename = os.path.basename(vid_path)
        # Check if file format is supported by OpenCV
        ext = filename.split('.')[-1]
        if ext not in ['avi', 'mp4', 'mov', 'mkv','webp']:
            print(f"{filename} is not a supported video format.")
            return
        
        if data_to_extract != "faces":
            print("Error!\nFor the moment function only works for extracting faces...Returning\n")
            return
        
        if save_dir == "":
            save_dir = data_to_extract

        # create a directory to store extracted faces
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        print(f"save_dir = {save_dir}")
        
        self.save_dir = save_dir
        
        # load face detection model
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # open video file
        cap = cv2.VideoCapture(vid_path)

        curr_iter = 9
        face_iter = 1
        # loop over frames in the video
        while True:
            curr_iter +=1
            if curr_iter%skip_frames == 0:
                # read a frame from the video
                ret, frame = cap.read()
                if not ret:
                    break

                # convert the frame to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect faces in the grayscale image
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

                # loop over the faces and save them to disk
                for i, (x, y, w, h) in enumerate(faces):
                    img_path = os.path.join(save_dir, f"face_{face_iter}.jpg")
                    print(f"Saving to = {img_path}")
                    # extract the face region from the frame
                    face = gray[y:y+h, x:x+w]
                    face = cv2.resize(face,(self.img_sz,self.img_sz))
                    # save the face image to disk
                    cv2.imwrite(img_path, face)
                    face_iter+=1
                    if self.debug:
                        cv2.imshow("face",face)
                        cv2.waitKey(1)

        # release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def preprocess(img):
        return img
    
    def isort(self,data_dir):
        labels = []
        features = []
        face_images = []
        # Step 1: Load the images and preprocess them
        for filename in os.listdir(data_dir):
            img = cv2.imread(os.path.join(data_dir, filename),cv2.IMREAD_UNCHANGED)
            # Apply face detection and alignment or facial landmark detection
            # to preprocess the images and extract the faces
            face_images.append(self.preprocess(img))
            
            # Step 2: Extract features using LBPH recognizer
            # Extract LBP features using skimage's local_binary_pattern function
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(img, n_points, radius)
            # Flatten the LBP features into a 1D array and append to the list of features
            features.append(lbp.flatten())

        # Step 3: Apply KMeans clustering to group similar faces together
        n_clusters = 8  # number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        clusters = kmeans.fit_predict(features)

        # Step 4: Save the results
        for i in range(n_clusters):
            cluster_dir = os.path.join(data_dir, 'cluster_{}'.format(i))
            os.makedirs(cluster_dir, exist_ok=True)
            cluster_indices = np.where(clusters == i)[0]
            counter = 1  # initialize counter
            for idx in cluster_indices:
                filename = 'person{}_img{}.jpg'.format(i, counter)
                img = face_images[idx]
                cv2.imwrite(os.path.join(cluster_dir, filename), img)
                counter+=1
        for filename in os.listdir(self.save_dir):
            file_path = os.path.join(self.save_dir, filename)
            if os.path.isfile(file_path):
                # Check if file format is supported by OpenCV
                ext = filename.split('.')[-1]
                if ext not in ['jpg', 'jpeg', 'png']:
                    print(f"{filename} is not a supported video format.")
                    return
                print(f"Deleting {filename}.....")
                os.remove(file_path)

    def extract(self,vid_path, data_to_extract = "faces", save_dir = "",skip_frames = 5,use_isort = False):
        
        # Return if save_directory exists and is filled with directoires of already sorted data.
        if os.path.isdir(save_dir) and len([f.path for f in os.scandir(save_dir) if f.is_dir()])>0:
            print("\n(Data Already Sorted!) Returning....\n")
            return
        
        self.extract_data(vid_path,data_to_extract,save_dir,skip_frames)
        
        if use_isort:
            if not use_optional:
                print("\n [Warning: -Isort disabled-] Set (use_optional -> True) in py to enable isort usage")
                return
            self.isort(save_dir)
# DeepSort Additions 

def draw_border(img, pt1, pt2, color, thickness, r, d):
    # https://stackoverflow.com/questions/50548556/python-opencvhow-to-draw-not-complete-rectangle-with-four-corners-on-image
    # https://stackoverflow.com/questions/46036477/drawing-fancy-rectangle-around-face
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
# Advanced