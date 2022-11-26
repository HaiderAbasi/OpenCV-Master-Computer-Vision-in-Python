import cv2
import numpy as np

from src.utilities import draw_points,print_h,build_montages


def main():
    
    print_h("[main]: Perform different image transformation on an simple rectangle and analyze the resultant image.")

    images = []
    titles = []

    # Perform Image transformations, Given an interesting assignment to utilize transformations
    img = np.zeros((200,300),np.uint8)
    rows,cols = img.shape[0:2]
    img[80:120,140:160] = 255

    # Adding image to list of images for displaying as a montage
    images.append(img)
    titles.append("Orig")

    # 1) Resizing :
    width = 150
    height = 100
    new_size = (width,height)
    img_resized = cv2.resize(img,new_size)
    images.append(img_resized)
    titles.append(f"Resized {new_size}")
    
    
    
    # 2) Translation : 
    tx = 100
    ty = 50
    M = np.float32([[1,0,tx],[0,1,ty]])
    img_translated = cv2.warpAffine(img,M,(cols,rows))
    images.append(img_translated)
    titles.append(f"translated (tx,ty)=({tx},{ty})")
    
    
    # 3) Rotation :
    angle = 90
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rotated = cv2.warpAffine(img,M,(cols,rows))
    images.append(img_rotated)
    titles.append(f"rotated {angle} deg")
    

    # 4) Affine :
    cnts = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
    cnt = cnts[0].reshape(4,2)
    
    pts1 = np.float32([cnt[0],cnt[1],cnt[2]])
    pts2 = np.float32([[100,67],[100,134],[200,134]])
    M = cv2.getAffineTransform(pts1,pts2)
    img_affine = cv2.warpAffine(img,M,(cols,rows))
    img_affine = draw_points(img_affine,cnt[0:3])
    img_affine = draw_points(img_affine,pts2)
    images.append(img_affine)
    titles.append("affine")

    # 5) Perspective :
    pts1 = np.float32([cnt[0],cnt[1],cnt[2],cnt[3]]) # Anti-Clockwise
    pts2 = np.float32([[100,67],[120,134],[170,134],[200,67]])
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    img_perspective = cv2.warpPerspective(img,M,(cols,rows))
    img_perspective = draw_points(img_perspective,cnt)
    img_perspective = draw_points(img_perspective,pts2)
    # Adding image to list of images for displaying as a montage
    images.append(img_perspective)
    titles.append("perspective")

    im_shape = (300,200)
    montages = build_montages(images, im_shape, None,titles,False,True)
    for img in montages:
        cv2.imshow("image",img) # Show large image
    cv2.waitKey(0)
    
if __name__ == "__main__":
    main()