import cv2
from numpy import random
import numpy as np

RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
ORANGE = (0,140,255)
BROWN = (42,42,165)

dark_colors = [RED,GREEN,BLUE,ORANGE,BROWN]


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
        

def draw_points(image,pts,radius=2):
    if len(image.shape)<3:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    for idx,pt in enumerate(pts):
        if type(pt)!= tuple:
            pt = tuple(pt)
        rand_color = dark_colors[idx]
        cv2.circle(image,pt,radius,rand_color,2)
    return image


def imshow(img_name,img,WindowType = cv2.WINDOW_NORMAL):
    """
    (displays complete image on the screen.)

    Parameters:
    img_name (String):    Name of the image
    img (numpy 3d array): Image

    Returns:
    None

    """
    #"""Decorator for OpenCV "imshow()" to handle images with transparency"""
    # Check we got np.uint8, 2-channel (grey + alpha) or 4-channel RGBA image
    #if (img.dtype == np.uint8) and (len(img.shape)==3) and (img.shape[2] in set([2,4])):
    if (len(img.shape)==4):

       # Pick up the alpha channel and delete from original
       alpha = img[...,-1]/255.0
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
    cv2.namedWindow(img_name,WindowType)
    
    if ((img_disp.shape[0]>=480) or (img_disp.shape[1]>=640)):
        img_disp = cv2.resize(img_disp,None,fx=0.5,fy=0.5)



    cv2.imshow(img_name,img_disp)    


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
        print(new_width)
        if (new_width <= int(width/2)):
            return (scale/10),new_width
    return 1,1

def putText(img,text_list,orig_list,type = "bbox"):
    
    fontScale,new_width = get_optimal_font_scale(text_list[0],img.shape[1])

    clr_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,255)]
        
    for idx,txt in enumerate(text_list):
        org = orig_list[idx]
        
        if type=="bbox":
            clr = clr_list[idx]
            if idx==0:
                # is even===> then adjust for pt center
                print(f"adjusted .... YAyyy! at {idx}")
                org = (org[0]-int(new_width/2),org[1])
            elif idx==3:
                # is even===> then adjust for pt center
                print(f"adjusted .... YAyyy! at {idx}")
                org = (org[0]-int(new_width/2),org[1])
        else:
            clr = random.random(size=3) * 256

        cv2.putText(img,txt,org,cv2.FONT_HERSHEY_PLAIN,fontScale,clr,4 )
