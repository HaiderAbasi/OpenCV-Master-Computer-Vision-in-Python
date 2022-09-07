from turtle import title
import cv2
import numpy as np
from utilities import imshow,describe,debugger
from imutils import build_montages



class segmentation:
    def __init__(self):
        self.debugger = None

    @staticmethod
    def thresholding(img,type = 'simple',thresh = 150):

        if type=='simple':
            img_seg_thresh = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)[1]
        elif type=='otsu':
            # apply Otsu's automatic thresholding which automatically determines
            # the best threshold value
            (T, img_seg_thresh) = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            print("[INFO] otsu's thresholding value: {}".format(T))
        elif type == 'adaptive-mean':
            img_seg_thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        elif type == 'adaptive-guass':    
            img_seg_thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        else:
            print(f"Unknown thresholding type {type}")

        return img_seg_thresh

    
    def segment_on_clr(self,img,hue_l = 0,hue_h = 0,lit_l = 0,sat_l = 0):

        hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        # Segmenting regions based on colour range
        lower_range = np.array([hue_l,lit_l,sat_l])
        upper_range = np.array([hue_h,255,255])
        mask_in_range = cv2.inRange(hls,lower_range,upper_range)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask = cv2.morphologyEx(mask_in_range,cv2.MORPH_CLOSE,kernel)
        return mask

    @staticmethod
    def segment_kmeans(img,K=2,attempts=10,display_clusters=False):
        rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        twoDimage = rgb.reshape((-1,3))
        twoDimage = np.float32(twoDimage)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((img.shape))

        if display_clusters:
            images_temp = []
            titles_temp = []

            for cluster_no in range(K):
                label_32f = label.copy() # Create a copy of label32f
                label_32f[label ==cluster_no] = 255 # Get only those cluster with label = user specified
                label_uint8 = np.uint8( label_32f.reshape((img.shape[1],img.shape[0]))) # Convert to uint8 for display
                images_temp.append(label_uint8)
                titles_temp.append(f"[Cluster={cluster_no}]")
            # Displaying image and threshold result
            cluster_montage = build_montages(images_temp,None,None,titles_temp,True,True)
            for cluster in cluster_montage:
                #imshow("Found Clusters",cluster,cv2.WINDOW_AUTOSIZE)
                cv2.imshow("Found Clusters",cluster)
            cv2.waitKey(0)
            cv2.destroyWindow("Found Clusters") 
            images_temp.clear()          
            titles_temp.clear()          
                

        return result_image

    def segment(self,img,method = "thresholding",type = "simple",tune = False):

        if method =="thresholding":
            segmented = self.thresholding(img,type)
        elif method =="color":
            if tune:
                if self.debugger == None: # initialize debugger
                    self.debugger = debugger("Control",["hue_l","hue_h","lit_l","sat_l"],[255,255,255,255])
                
                self.debugger.update_variables() # Get updated variables
                hue_l,hue_h,lit_l,sat_l= self.debugger.debug_vars[0:5]
            else:
                 hue_l= 100;hue_h = 255;lit_l = 0;sat_l = 0
            segmented = self.segment_on_clr(img,hue_l,hue_h,lit_l,sat_l)

        elif method =="edges":
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if tune:
                if self.debugger == None: #i nitialize debugger
                    self.debugger = debugger("Control",["thresh_l","thresh_h","aperture"],[255,255,7],[False,False,True])

                self.debugger.update_variables() # Get updated variables
                thresh_l,thresh_h,aperture= self.debugger.debug_vars[0:4]
            else:
                thresh_l=50;thresh_h=150;edges=None;aperture=3

            segmented = cv2.Canny(gray,thresh_l,thresh_h,edges,aperture)

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

            
        
        return segmented



def main():
    img_segmentor = segmentation()

    #img = cv2.imread("Data/CV/boy_who_lived/normal.jpg")
    img_hp_vig = cv2.imread("Data/CV/boy_who_lived/vignette.jpg")
    img = img_hp_vig
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    images = []
    titles = []
    images.append(gray)
    titles.append("img")

    # 1) Segmentation using Thresholding
    # Case: Where thresholding might be the obvious choice
    #   a) Simple
    img_seg_thresh = img_segmentor.segment(gray)
    images.append(img_seg_thresh)
    titles.append("Thresholding (Simple)")
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

    # 2) Segmentation using Color
    messi_img = cv2.imread("Data\CV\messi5.jpg")

    while(1):
        images.clear()
        titles.clear()
        images.append(messi_img)
        titles.append("img (Messi)")
        
        img_seg_color = img_segmentor.segment(messi_img,"color",tune=False)
        images.append(img_seg_color)
        titles.append("Segmented (color)")
        
        # 3) Segmentation using edges
        img_seg_edges = img_segmentor.segment(messi_img,"edges",tune=False)
        images.append(img_seg_edges)
        titles.append("Segmented (edges)")

        # Displaying image and threshold result
        montage = build_montages(images,None,None,titles,True,True)
        for img in montage:
            cv2.imshow("img",img)
        k = cv2.waitKey(1)
        if k==27:
            break
    cv2.destroyWindow("img")

    # 3) Segmentation using Clustering (kmeans)
    baboon_img = cv2.imread("Data/CV/baboon.jpg")

    while(1):
        images.clear()
        titles.clear()
        images.append(baboon_img)
        titles.append("img (baboon)")
        
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