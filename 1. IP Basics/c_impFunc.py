import cv2
from utilities import imshow

def main():
    # Read an image
    img = cv2.imread("Data\CV\messi5.jpg")
    # Display
    imshow("image",img)

    # a) Select ROI on the image
    x,y,w,h = cv2.selectROI("Select ROI",img)

    # b) Crop the Selected ROI 
    roi = img[y:y+h,x:x+w] # img[row:row+height,col+col+width]
    imshow("roi",roi)

    # c) Change color Space of the image
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imshow("gray",gray)

    # d) Change color Space of the image to hsv and display each channel
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    imshow("hue",hls[:,:,0]) # Hue Channel : Displays only color information
    imshow("lit",hls[:,:,1]) # Ligthness   : Displays only the amount of light
    imshow("sat",hls[:,:,2]) # Saturation  : Displays only the amount of color
    cv2.waitKey(0)

    # Assignment: Display only the bottom-left plant in the field of image given below.
    #             Result should be in single-channel (Without-Shadow)
    field_img = cv2.imread("Data\CV\drone_view.png")
    imshow("field_img",field_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()