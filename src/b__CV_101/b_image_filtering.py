import cv2
import numpy as np

from src.utilities import imshow,build_montages,print_h



def highlight_roi(image,debug = True):
    # Hint: Enhancing meteor edges could be the key but there is alot of noise so some smoothing is required

    img_roi_highlighted = image.copy()
    
    return img_roi_highlighted


def assignment(debug = True):
    if debug:
        print_h("[Assignment]: Highlight falling meteor in the scene\n")
    # Assignment: Define the algorithm whos goal is to highlight meteor (roi) in the whole scene
    # 
    #
    # Hint: Along with using filter to silence the noise, Look into Unsharp Masking for highlighting
    #       Reference: https://scikit-image.org/docs/stable/auto_examples/filters/plot_unsharp_mask.html
    #
    # 
    # Output: Video with > Roi-highlighted < saved to disk
    #
    
    vid = cv2.VideoCapture("Data\meteor_mini.mp4")
    
    # Extracting input video properties to be used for output videowriter initialization
    inp_fps = vid.get(cv2.CAP_PROP_FPS)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    size = (int(width),int(height))

    vid_roi_highlighted = cv2.VideoWriter("src/b__CV_101/vid_roi_highlighted.avi",cv2.VideoWriter_fourcc(*'MJPG'),inp_fps,size)
    
    while(vid.isOpened()):
        ret,frame = vid.read()
        if ret:
            if debug:
                imshow("Meteor_strike (Orig)",frame)
            # ### Task Function ###
            roi_highlighted = highlight_roi(frame,debug)
            # Writing video to disk
            vid_roi_highlighted.write(roi_highlighted)
            # Output (Display)
            if debug:
                imshow("[2] Meteor strike",roi_highlighted)
            k=cv2.waitKey(1)
            if k==27:
                break
        else:
            print("Video Ended")
            break

    vid_roi_highlighted.release()



k_w = 3
k_h = 3


ksize = 3

image_no = 0


def on_k_w_Change(val):
    global k_w
    k_w = 2*val + 1

    if (k_w<3):
        k_w = 3

def on_k_h_Change(val):
    global k_h
    k_h = 2*val + 1

    if (k_h<3):
        k_h = 3

def onksizeChange(val):
    global ksize
    ksize = 2*val + 1

    if (ksize<3):
        ksize = 3

def on_image_no_Change(val):
    global image_no
    image_no = val

def denoising(noisy_img):
    images = []
    titles = []   

    images.append(noisy_img)
    titles.append("noisy_img")

    #   a) LPF # 1: Box filter (Linear filter Used for smoothing, edge preservation minimal)
    kernel = np.ones((5,5),np.float32)/25 # box filter 
    img_filtered = cv2.filter2D(noisy_img,-1,kernel) # depth = -1 (As input depth)
    images.append(img_filtered)
    titles.append("filtered (box)")
    
    #   b) LPF #2: Guassian filter (Linear filter Used for smoothing or to reduce noise , edge preservation okay)
    #                               Gives more weightage to closer pixels then farther in deciding result
    img_guass = cv2.GaussianBlur(noisy_img,(k_w,k_h),0,0)
    images.append(img_guass)
    titles.append(f"filtered (guass {(k_w,k_h)} )")
    
    #   b) LPF #2: Median filter (Non-linear filter used for denoising (Salt-pepper noise), edge preservation better)
    #                             Slow, because it needs to perform sorting to find the median in the underlying array
    img_median = cv2.medianBlur(noisy_img,ksize)
    images.append(img_median)
    titles.append(f"filtered (median {ksize})")
    
    #   c) Combo #A Median -< Guassian
    img_medguass = cv2.GaussianBlur(img_median,(k_w,k_h),0,0)
    images.append(img_medguass)
    titles.append(f"filtered (medguass)") 
    
    # Otsu Thresholding for segmenting ROI from the given image
    threshold_img = cv2.threshold(images[image_no],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    images.append(threshold_img)
    titles.append(f"threshed {titles[image_no]}") 
    
    # Displaying the montage
    montage_shape = (300,200)
    montage = build_montages(images,montage_shape,None,titles,True,True)
    for img in montage:
        imshow("Blurring (Noise Removal)",img)


def detect_edges(image):
    images = []
    titles = []
    
    images.append(image)
    titles.append("image (Original)")
    
    edge_X = cv2.Sobel(image,-1,1,0)
    images.append(edge_X)
    titles.append("edges (X)")
    
    edge_X_64f = cv2.Sobel(image,cv2.CV_64F,1,0)
    edge_X_char = cv2.convertScaleAbs(edge_X_64f,alpha=(255/edge_X_64f.max()))
    images.append(edge_X_char)
    titles.append("edge_X (Scaleabs)")
    
    edge_Y = cv2.Sobel(image,-1,0,1)
    images.append(edge_Y)
    titles.append("edges (Y)")
    
    edge_Y_64f = cv2.Sobel(image,cv2.CV_64F,0,1)
    edge_Y_char = cv2.convertScaleAbs(edge_Y_64f, alpha=(255/edge_Y_64f.max()))

    images.append(edge_Y_char)
    titles.append("edge_Y (Scaleabs)")
    
    edge_XY = cv2.Sobel(image,cv2.CV_64F,1,1)
    edge_XY = cv2.convertScaleAbs(edge_XY, alpha=(255/edge_XY.max()))
    images.append(edge_XY)
    titles.append("edges (XY)")
    
    edges_laplace = cv2.Laplacian(image,cv2.CV_64F,3)
    edges_laplace = cv2.convertScaleAbs(edges_laplace, alpha=(255/edges_laplace.max()))
    images.append(edges_laplace)
    titles.append("edges (Laplacian)")
    
    edges_canny = cv2.Canny(image,50,150,None,3)
    images.append(edges_canny)
    titles.append("edges (Canny)")
    
    montage_shape = (300,200)
    montage = build_montages(images,montage_shape,None,titles,True,True)
    for img in montage:
        imshow("Edge detection",img)


def main():
    print_h("[main]: Applying different type of Image filters to input and analyzing their effects.")
    images = []
    titles = []

    # Task 1: Smoothing using filter2d (box filter)
    img = cv2.imread("Data\HappyFish.jpg")
    images.append(img)
    titles.append("img (Orig)")

    # > Creating a box filter of size 5x5
    kernel = np.ones((5,5),np.float32)/25
    print("kernel = ",kernel)
    img_filtered = cv2.filter2D(img,-1,kernel)
    images.append(img_filtered)
    titles.append("filtered (box)")
    
    

    montage = build_montages(images,None,None,titles,True,True)
    for img in montage:
        imshow("Image Filtering",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    # Task 2: Noise Removal using low pass filters (Choise of filter depends on the noise characteristics)
    print_h("[Noise Removal]: Utilizing low-pass filters for noise removal in the given image.")
    noisy_img = cv2.imread("Data/noisy2.png",cv2.IMREAD_ANYDEPTH) # AnyDepth to ensure it is read as a grayscale
    
    
    cv2.namedWindow("Noise Removal",cv2.WINDOW_NORMAL)
    cv2.createTrackbar("k_w","Noise Removal",k_w,30,on_k_w_Change)
    cv2.createTrackbar("k_h","Noise Removal",k_h,30,on_k_h_Change)
    cv2.createTrackbar("ksize","Noise Removal",ksize,30,onksizeChange)
    cv2.createTrackbar("image_no","Noise Removal",image_no,10,on_image_no_Change)

    
    while(1):
        denoising(noisy_img)
        k = cv2.waitKey(1)
        if k==27:
            break

    

    # Task 3: Edge detection using high pass filters (Choise of filter depends on the type of edge we want)
    print_h("[Edge detection]: Leveraging high-pass filters to extract areas of change in image.")
    shapes_img = cv2.imread("Data\shapes.PNG",cv2.IMREAD_ANYDEPTH) # AnyDepth to ensure it is read as a grayscale

    
    detect_edges(shapes_img)
    cv2.waitKey(0)
    
    
    

        
if __name__ == "__main__":
    
    i_am_ready = False
    
    if i_am_ready:
        assignment()
    else:
        main()