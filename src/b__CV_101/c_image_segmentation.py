import cv2
import numpy as np
from src.utilities import debugger,build_montages,print_h


class segmentation:
    def __init__(self):
        self.debugger = None
        
    @staticmethod
    def thresholding(img,type = "binary"):
        if type == "binary":
            img_seg_thresh = cv2.threshold(img,150,255,cv2.THRESH_BINARY)[1]
        elif type == "otsu":
            T,img_seg_thresh = cv2.threshold(img,0,255,cv2.THRESH_OTSU|cv2.THRESH_BINARY)
            print("[INFO] otsu's thresholding value: {}".format(T))
        elif type == "adaptive-mean":
            img_seg_thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        elif type == 'adaptive-guass':
            img_seg_thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        else:
            print(f"Unknown thresholding type {type}")
            
        return img_seg_thresh

    def segment_color(self,img,hue_l,hue_h):
        hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        hue = hls[:,:,0]
        mask = cv2.inRange(hue,hue_l,hue_h)
        # Perform Closing operation to connect closely disconnected objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        return mask
    
    def segment_edges(self,img,thresh_l,thresh_h,aperture):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        mask = cv2.Canny(gray,thresh_l,thresh_h,None,aperture)
        return mask
    
    @staticmethod
    def segment_kmeans(img,clusters = 2, attempts=10):
        
        # Each channel/feature is placed in a single column
        twoDimage = img.reshape((-1,3))
        
        twoDimage = np.float32(twoDimage)
        
        # Defining the criteria for kmean algorithm
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_COUNT,10,1.0)
        
        _,label, center= cv2.kmeans(twoDimage,clusters,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        # Kmeans- Display Clusters with seperate coloring
        # Output Image : Convert back to Uint8-3dShaped 
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((img.shape))
        
        return result_image
    
    
    def segment(self,img,method="thresholding",type="binary",tune = False):
        if method == "thresholding":
            segmented = self.thresholding(img,type)
        elif method == "color":
            if tune:
                if self.debugger == None: # initialize debugger
                    print("debugger is None ")
                    self.debugger = debugger("Control",["hue_l","hue_h"],[255,255])
                self.debugger.update_variables() # Get updated variables
                hue_l,hue_h= self.debugger.debug_vars[0:2]
            else:
                hue_l= 100;hue_h = 255
            segmented = self.segment_color(img,hue_l,hue_h)
        elif method == "edges":
            if tune:
                if self.debugger == None: #i nitialize debugger
                    self.debugger = debugger("Control (edges)",["thresh_l","thresh_h","aperture"],[255,255,7],[False,False,True])

                self.debugger.update_variables() # Get updated variables
                thresh_l,thresh_h,aperture= self.debugger.debug_vars[0:4]
            else:
                thresh_l=50;thresh_h=150;aperture=3
            segmented = self.segment_edges(img,thresh_l,thresh_h,aperture)
        elif method =="kmeans":
            if tune:
                if self.debugger == None: #i nitialize debugger
                    self.debugger = debugger("Control",["K","attempts"],[10,50])

                self.debugger.update_variables() # Get updated variables
                k,attempts= self.debugger.debug_vars[0:2]
                if k==0:
                    k = 2
                if attempts<10:
                    attempts = 10
            else:
                k = 2; attempts = 10
            segmented = self.segment_kmeans(img,k,attempts)
        else:
            print(f"Unknown Method: {method}")
            
        return segmented



def main():

    print_h("[main]: Applying different type of Image filters to input and analyzing their effects.")

    img = cv2.imread("Data/boy_who_lived/vignette.jpg",cv2.IMREAD_ANYDEPTH)
    print("img Shape (Input)",img.shape)
    if ( (len(img.shape)==3) and (img.shape[2]==3) ):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    images = []
    titles = []
    images.append(gray)
    titles.append("img")

    img_segmentor = segmentation()
    
    # Task 1: Segmentation using Thresholding
    print_h("[Segmentation Using Thresholding]: Investigating the use cases of Simple to adaptive thresholding.")

    # Case: Where thresholding might be the obvious choice
    #   a) Binary
    img_seg_thresh = img_segmentor.segment(gray)
    images.append(img_seg_thresh)
    titles.append("Thresholding (Simple)")
    
    
    
    #   a-2) Otsu    
    img_seg_thresh = img_segmentor.segment(gray,type ='otsu')
    images.append(img_seg_thresh)
    titles.append("Thresholding (Otsu)")
    
    #   b) Adaptive mean
    img_seg_thresh_adaptive = img_segmentor.segment(gray,type="adaptive-mean")
    images.append(img_seg_thresh_adaptive)
    titles.append("Thresholding (adaptive-mean)")
    
    
    #   b) Adaptive guassian
    img_seg_thresh_adaptive = img_segmentor.segment(gray,type="adaptive-guass")
    images.append(img_seg_thresh_adaptive)
    titles.append("Thresholding (adaptive-guass)")
    
    
    
    
    # Displaying image and threshold result
    montage = build_montages(images,None,None,titles,True,True)
    for img in montage:
        cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyWindow("img")









    # Task 2: Segmentation using Color
    print_h("[Segmentation Using Color]: Performing segmentation based on the selected color.")
    messi_img = cv2.imread("Data\messi5.jpg")

    while(1):
        images.clear()
        titles.clear()
        # Adding image to list of images in montage
        images.append(messi_img)
        titles.append("img (Messi)")
        
        # 2) Segmentation using color
        img_seg_color = img_segmentor.segment(messi_img,"color",tune=False)
        images.append(img_seg_color)
        titles.append("Segmented (color)")
        
        
        # 3) Segmentation using edges
        img_seg_edges = img_segmentor.segment(messi_img,"edges",tune=True)
        images.append(img_seg_edges)
        titles.append("Segmented (edges)")


        # Displaying image and threshold result
        montage = build_montages(images,None,None,titles,True,True)
        for img in montage:
            cv2.imshow("img",img)
        k = cv2.waitKey(1)
        if k==27:
            break
    cv2.destroyAllWindows()
    img_segmentor.debugger = None
    
    
    
    
    # Task 3: Segmentation using Clustering (kmeans)
    print_h("[Segmentation Using Clustering]: Applying kmeans to divide image into similar regions(clusters).")
    baboon_img = cv2.imread("Data/baboon.jpg")

    while(1):
        images.clear()
        titles.clear()
        images.append(baboon_img)
        titles.append("img (baboon)")
        
        # Performing kmean segmentation
        
        img_seg_kmeans = img_segmentor.segment(baboon_img,"kmeans",tune=True)
        images.append(img_seg_kmeans)
        titles.append("Segmented (kmeans)")
        
        
        
        
        
        
        
        
        # Displaying image and threshold result
        montage = build_montages(images,None,None,titles,True,True)
        for img in montage:
            cv2.imshow("img",img)
        
        k = cv2.waitKey(1)
        if k==27:
            break




if __name__ == "__main__":
    main()