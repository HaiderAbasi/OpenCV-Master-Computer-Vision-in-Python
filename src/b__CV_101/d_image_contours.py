import cv2
from src.utilities import build_montages,print_h,Gui,putText,RetLargestContour

from loguru import logger
import numpy as np
# import the necessary packages

class ShapeDetector:
	def __init__(self):
		print("\n### Kudos to PyimageSearch :)")
	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"
		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			# compute the bounding box of the contour and use the
			# bounding box to compute the aspect ratio
			(x, y, w, h) = cv2.boundingRect(approx)
			ar = w / float(h)
			# a square will have an aspect ratio that is approximately
			# equal to one, otherwise, the shape is a rectangle
			shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = "pentagon"
		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"
		# return the name of the shape
		return shape
  
  
def identify_shape(img,debug = True):

    img_shape_identified = img.copy()
    valid_shapes = [] # Identified Valid shapes
    valid_shapes_cntrs = [] # Corresponding shapes centers
    # Write code here
    
    
    
    
    return img_shape_identified,valid_shapes,valid_shapes_cntrs


def assignment(debug = True):
    # Assignment : Use Contours to Identify shape of each object. + 
    #              Color each object based on their hierichecal level.
    #                [i.e. Innormost would be the strongest color]
    #
    # Returns    : (List[String]) Identified shapes
    #              (List[Tuple])  Centers of those Shapes 
    #
    # Hint       : Contours Hierarchy might be particularly useful here.
    #              Especially parent-child relationship
    #              Resources: https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    #
    #
    if debug:
        print_h("[Assignment]: Use Contours to Identify shape of each object + center of object based on heirarchical level\n")

    # Input
    img = cv2.imread("Data\pic5.png")
    
    if debug:
        cv2.imshow("> Shapes < ",img)
        cv2.waitKey(0)
        cv2.destroyWindow("> Shapes < ")
        
    images = []
    titles = []
    images.append(img)
    titles.append("Img")

    # Task function
    img_shape_identified,shapes,shapes_cntrs = identify_shape(img,debug)

    if np.array_equal(img_shape_identified,img):
        logger.error("identify_shape() needs to be coded to get the required(Shapes identified) result.")
        exit(0)
    
    if debug:
        print(f"\n\nFound {len(shapes)} shapes. These are...\n")
        print(shapes)
        print(shapes_cntrs,"\n")
        
    # Output(Display)
    images.append(img_shape_identified)
    titles.append("img_shape_identified")
    
    if debug:
        # Displaying image and threshold result
        montage = build_montages(images,None,None,titles,True,True,True)
        for montage_img in montage:
            #imshow("Found Clusters",cluster,cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Assignment",montage_img)
        cv2.waitKey(0)
        
    
    return shapes,shapes_cntrs








def extract_nd_draw_contours(img):
    images = []
    titles = []
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    images.append(gray)
    titles.append("gray")
    
    
    # [Step a] Generate a binary Image
    edges = cv2.Canny(gray,20,150,None,3)
    images.append(edges)
    titles.append("edges")
    
    # [Step b] Use find Contours to extract contours
    
    # Case A: Retreive all the contours in the image and draw them (No hierarchical relationship)
    cnts,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    img_all_cnts = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_all_cnts,cnts,-1,(0,255,0),1)
    images.append(img_all_cnts)
    titles.append("img_all_cnts")
    
    # Case B: Retreive and display only the outermost contours and fill it in the second step
    cnts,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    img_ext_cnts = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_ext_cnts,cnts,-1,(0,255,0),2)
    images.append(img_ext_cnts)
    titles.append("img_ext_cnts")


    img_ext_cnts_filled = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_ext_cnts_filled,cnts,-1,(0,255,0),-1)
    images.append(img_ext_cnts_filled)
    titles.append("img_ext_cnts_filled")


    # Case C: Identify contours and their holes as seperate entities [next, previous, first child, parent]
    cnts,hierch = cv2.findContours(edges,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    img_boundariesandholes = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    
    boundaries = []
    holes = []
    for idx,cnt in enumerate(cnts):
        curr_h = hierch[0][idx]
        if curr_h[3] == -1:
            boundaries.append(cnt)
        else:
            holes.append(cnt)
            
    
    cv2.drawContours(img_boundariesandholes,boundaries,-1,(0,255,0),2)
    
    cv2.drawContours(img_boundariesandholes,holes,-1,(0,0,255),-1)
    images.append(img_boundariesandholes)
    titles.append("img_boundariesandholes")


    

    # Displaying image and threshold result
    montage = build_montages(images,None,None,titles,True,True)
    for montage_img in montage:
        #imshow("Found Clusters",cluster,cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Montage",montage_img)
    cv2.waitKey(0)
        
    cv2.destroyAllWindows()


def get_centroid(cnt):
    M = cv2.moments(cnt)
    if M['m00']==0: # If its a line (No Area) then use minEnclosingcircle and use its center as the centroid
        (cx,cy) = cv2.minEnclosingCircle(cnt)[0]        
        return (int(cx),int(cy))
    else:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return (cx,cy)


def analyze_contours(img,Loop = False):
    
    while(1):
        images = []
        titles = []
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,20,150,None,3)
        
        cnts,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
        cnts_img = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cnts_img,cnts,-1,(0,255,0),1) # -1 in thickness fills the found contours
        images.append(cnts_img)
        titles.append("found_contours (External)")
        
        img_analysis = cnts_img.copy()
        
        for cnt in cnts:
            cntr = get_centroid(cnt)
            cv2.circle(img_analysis,cntr,3,(0,0,255),-1)
            
        gui = Gui()
        idx,cnt = gui.select_cnt(edges,cnts,Loop)
        if idx==-1:
            # idx = -1 indicates select_cnt exiting without user selecting a contour... Function will be returned empty
            break
        
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        cv2.drawContours(img_analysis,[approx],-1,(255,0,0),4)
        
        hull = cv2.convexHull(cnt)
        cv2.drawContours(img_analysis,[hull],-1,(128,0,255),2)
        
        rows,cols = edges.shape[0:2]
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(img_analysis,(cols-1,righty),(0,lefty),(255,255,255),2)
        
        images.append(img_analysis)
        titles.append("Contour Analysis")

        # Displaying image and threshold result
        montage = build_montages(images,None,None,titles,True,True)
        for montage_img in montage:
            #imshow("Found Clusters",cluster,cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Montage",montage_img)
        if Loop:
            cv2.waitKey(1)
        else:
            cv2.waitKey(0)
            break
            


def main():
    print_h("[main]: Extracting Contours and Leveraging them for shape analysis.")

    # Reference: https://stackoverflow.com/questions/17103735/difference-between-edge-detection-and-image-contours
    img = cv2.imread("Data\pic1.png")
    
    # Task1: Investigate find and drawContours
    print_h("[a]: Extracting and displaying contours in image.")

    extract_nd_draw_contours(img)


    # Task2: Use Contour Features for advance analysis
    print_h("[b]: Investiagting contour features for shape analysis of objects. \n")
    analyze_contours(img)
    





if __name__ == "__main__":
    i_am_ready = False
    
    if i_am_ready:
        assignment()
    else:
        main()