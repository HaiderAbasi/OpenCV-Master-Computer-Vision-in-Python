import numpy as np
import cv2

roi_confirmed = False
ix,iy = -1,-1
fx,fy = -1,-1
selectedROIs = []


def selectROIs(img,title = 'SelectROIs'):
    global roi_confirmed
    cv2.namedWindow(title)
    cv2.setMouseCallback(title,draw_circle)
    while(1):
        cv2.imshow(title,img)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:# Enter
            roi_confirmed = True
        elif k == 27:
            break
    cv2.destroyAllWindows()
    print("selectedROIs = ",selectedROIs)

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,fx,fy,roi_confirmed,selectedROIs
    if event == cv2.EVENT_LBUTTONDOWN:
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if roi_confirmed:
            cv2.rectangle(img,(ix,iy),(fx,fy),(0,255,0),2)
            if ix <= fx :
                strt_col = ix
                width = fx - ix
            else:
                strt_col = fx
                width = ix - fx
            if iy <= fy:
                strt_row = iy
                height = fy - iy
            else:
                strt_row = fy
                height = iy - fy
            selectedROIs.append([strt_col,strt_row,width,height])
            roi_confirmed = False
            
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(img,(ix,iy),(x,y),(0,140,255),2)
        fx = x
        fy = y




img = np.zeros((512,512,3), np.uint8)
selectROIs(img)
