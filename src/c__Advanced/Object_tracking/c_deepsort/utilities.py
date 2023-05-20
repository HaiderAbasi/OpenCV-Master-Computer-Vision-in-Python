import cv2
import numpy as np
import random
from collections import deque

# Overload puttext function to add text any of the four corner in image
def putText(img, text, font=cv2.FONT_HERSHEY_PLAIN, font_scale=1, color=(0, 0, 255), thickness=1, margin=10, pos='top-right', bg_color=None):
    """
    Displays text on an image with an adjustable margin.

    Parameters:
    img (numpy.ndarray): The input image.
    text (str): The text to display.
    font (int): The font type to use (default: cv2.FONT_HERSHEY_PLAIN).
    font_scale (float): The font scale factor (default: 1).
    color (tuple): The text color in BGR format (default: (0, 0, 255)).
    thickness (int): The thickness of the text (default: 1).
    margin (int): The margin between the text and the edge of the image (default: 10).
    pos (str): The position of the text. Can be 'top-left', 'top-right', 'bottom-left', or 'bottom-right' (default: 'top-right').
    """

    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness=thickness)

    # Define the position of the text
    if pos == 'top-left':
        text_x = margin
        text_y = margin + text_size[1]
    elif pos == 'top-right':
        text_x = img.shape[1] - text_size[0] - margin
        text_y = margin + text_size[1]
    elif pos == 'bottom-left':
        text_x = margin
        text_y = img.shape[0] - margin
    elif pos == 'bottom-right':
        text_x = img.shape[1] - text_size[0] - margin
        text_y = img.shape[0] - margin
    else:
        raise ValueError("Invalid position. Position should be 'top-left', 'top-right', 'bottom-left', or 'bottom-right'.")
    
    # Draw the text with or without background color
    if bg_color:
        # Calculate the background rectangle size and position
        bg_rect_width = text_size[0] + margin * 2
        bg_rect_height = text_size[1] + margin * 2
        bg_rect_x = text_x - margin
        bg_rect_y = text_y - text_size[1] - margin

        # Draw the background rectangle
        cv2.rectangle(img, (bg_rect_x, bg_rect_y), (bg_rect_x + bg_rect_width, bg_rect_y + bg_rect_height), bg_color, thickness=-1)

    # Draw the text on the image
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)

# Generate a random vibrant color
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

# Find centroid while considering bbox type
def find_centroid(bbox,bbox_type = "ltrd"):
    """
    This function computes the centroid of a bounding box given its coordinates in ltrd format.

    Parameters:
    bbox (tuple): A tuple of 4 values (x1, y1, x2, y2) that define the bounding box coordinates.
    bbox_type (str, optional): The type of bounding box, default is "ltrd".

    Returns:
    tuple: A tuple of 2 values (x_center, y_center) representing the x and y coordinates of the centroid.

    Raises:
    ValueError: If an unsupported bounding box type is provided.
    """
    if (bbox_type == "ltrd"):
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        return (int(x_center), int(y_center))
    else:
        print("[Unsupported bbox_type]: Please provide a 'ltrd' bbox!")
        return (0,0)

# Find bbox closest to the specified point
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


# Add a key to a dictionary of deques
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
        cv2.namedWindow(title,cv2.WINDOW_NORMAL)
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

    # ADVANCED #



    def ret_point(self,event,x,y, flags, param):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            self.pt = (x,y)

    def select_cnt(self,img,cnts,Loop=False):
        cv2.namedWindow("Select Contour",cv2.WINDOW_NORMAL)
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
                    print("(Incorrect Selection) --> Please only select an object!!")
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
                print("(Esc key pressed) --> No contours selected + Exiting!!!")
                return -1,-1 # idx = -1 (indicating no contour and exiting!)

