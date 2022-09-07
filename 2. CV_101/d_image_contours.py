import cv2
import numpy as np

from utilities import describe,imshow
from imutils import build_montages


class gui_control():

    def __init__(self):
        self.pt = None # point instance variable


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
                    print("Incorrect Selection: Please only select an object!")
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

            cv2.waitKey(1)



def get_centroid(cnt):
    M = cv2.moments(cnt)
    if M['m00']==0: # If its a line (No Area) then use minEnclosingcircle and use its center as the centroid
        (cx,cy) = cv2.minEnclosingCircle(cnt)[0]        
        return (int(cx),int(cy))
    else:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx,cy)


def extract_nd_draw_contours(img):
    images = []
    titles = []
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    images.append(gray)
    titles.append("gray")
    edges = cv2.Canny(gray,20,150,None,3)
    images.append(edges)
    titles.append("Segmented (canny)")

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    #edges = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel)

    # Important : If you want to fill all the objects use the Cv2.ret_external and then -1 in cntridx and thickness
    #             If you want to fill all the objects but do that looping use cv2.ret_List or cv2.ret_Tree and then -1 in cntridx and thickness
    
    # Case A: Retreive all the contours in the image and draw them (No hierarchical relationship)
    cnts,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    
    found_contours = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(found_contours,cnts,-1,(0,255,0),1) # -1 in thickness fills the found contours
    images.append(found_contours)
    titles.append("found_contours (List)")

    # Case B: Retreive and display only the outermost contours and fill it in the second step
    cnts,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    found_contours = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(found_contours,cnts,-1,(0,255,0),2) 
    images.append(found_contours)
    titles.append("found_contours (External)")
    
    found_contours_filled = found_contours.copy()
    cv2.drawContours(found_contours_filled,cnts,-1,(0,255,0),-1) # -1 in thickness fills the found contours
    images.append(found_contours_filled)
    titles.append("found_contours (External-Filled)")

    # Case C: Identify contours and their holes as seperate entities [next, previous, first child, parent]
    cnts,heirch = cv2.findContours(edges,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    
    found_contours_ccomp = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)

    boundaries = []
    holes = []
    for idx,cnt in enumerate(cnts):
        curr_h = heirch[0][idx] # Retreiving heirrachy of the current contour
        if curr_h[3]==-1: # If the current contour has no parent: Its a boundary
            boundaries.append(cnt)
        else: # Otherwise its a hole because its only divided into 2 way relationship [Boundary | Hole]
            holes.append(cnt)
    cv2.drawContours(found_contours_ccomp,boundaries,-1,(0,255,0),2)
    cv2.drawContours(found_contours_ccomp,holes,-1,(0,0,255),-1) 
    images.append(found_contours_ccomp)
    titles.append("found_contours (Boundary&Holes)")

    # Displaying image and threshold result
    montage = build_montages(images,None,None,titles,True,True)
    for montage_img in montage:
        #imshow("Found Clusters",cluster,cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Montage",montage_img)
    cv2.waitKey(0)

    # Displaying nested hole can only be done by looping, otherwise outerhole will cover the inner :(
    found_contours_ccomp = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    for hole in holes:
        img_holes = cv2.drawContours(found_contours_ccomp.copy(),[hole],0,(0,0,255),-1)
        cv2.imshow("FoundHoles",img_holes)
        cv2.waitKey(1) 
    cv2.destroyWindow("Montage")     
    cv2.destroyAllWindows()

def analyze_contours(img,Loop = False):
    while(1):
        images = []
        titles = []
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,20,150,None,3)
        # Case A: Retreive all the contours in the image and draw them (No hierarchical relationship)
        cnts,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        cnts_img = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cnts_img,cnts,-1,(0,255,0),1) # -1 in thickness fills the found contours
        images.append(cnts_img)
        titles.append("found_contours (External)")

        # Feature Analysis A: Find the centroid of all object and draw
        img_centroid = cnts_img.copy()
        for cnt in cnts:
            cntr = get_centroid(cnt)
            cv2.circle(img_centroid,cntr,3,(0,0,255),-1)


        # Feature Analysis B: Find the centroid of all object and draw
        gui = gui_control()
        idx,cnt = gui.select_cnt(edges,cnts,Loop)

        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True) # Helpful in approximating the shape incase of misaligned edges (Only polygons)
        cv2.drawContours(img_centroid,[approx],-1,(255,0,0),4) # -1 in thickness fills the found contours

        hull = cv2.convexHull(cnt) # RubberBand Covering of the object boundary (Crevaces will be left empty)
        cv2.drawContours(img_centroid,[hull],-1,(128,0,255),2) # -1 in thickness fills the found contours

        #
        rows,cols = edges.shape[0:2]
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(img_centroid,(cols-1,righty),(0,lefty),(255,255,255),2)

        images.append(img_centroid)
        titles.append("found_contours (External-Centroid)")

        # Displaying image and threshold result
        montage = build_montages(images,None,None,titles,True,True)
        for montage_img in montage:
            #imshow("Found Clusters",cluster,cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Montage",montage_img)
        if Loop:
            k = cv2.waitKey(1)
            if k==27:
                break
        else:
            cv2.waitKey(0)
            break


def assignment_contours(img):
    images = []
    titles = []
    images.append(img)
    titles.append("Img")

    # Displaying image and threshold result
    montage = build_montages(images,None,None,titles,True,True)
    for montage_img in montage:
        #imshow("Found Clusters",cluster,cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Assignment",montage_img)
    cv2.waitKey(0)

def main():
    # Reference: https://stackoverflow.com/questions/17103735/difference-between-edge-detection-and-image-contours
    img = cv2.imread("Data\CV\pic1.png")
    
    # Task1: Investigate find and drawContours
    extract_nd_draw_contours(img)

    # Task2: Use Contour Features for advance analysis
    analyze_contours(img)


    # Assignment : In the Given Image, Use Contours to Identify shape of each object. Also, Color 
    #              each object based on their hierichecal level. [i.e. Innormost would be the strongest color]
    img = cv2.imread("Data\CV\pic5.png")
    
    assignment_contours(img)




if __name__ == "__main__":
    main()