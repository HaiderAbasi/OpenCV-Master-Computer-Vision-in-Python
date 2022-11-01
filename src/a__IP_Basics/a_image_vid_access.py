import cv2
from src.utilities import print_h


# 1: [Reading/Displaying Image]:
print_h("1: Reading and displaying image from disk...\n")
image = cv2.imread("Data/sunset.jpg")
cv2.namedWindow("Sunset",cv2.WINDOW_NORMAL)
cv2.imshow("Sunset",image)
cv2.waitKey(0)

# Displaying image dimensions (rows,cols,channels)
print("- Retreiving img details...\n")
(rows,cols,channels) = image.shape
print(f"Image has dimensions {(rows,cols,channels)}")

# 2: [Writing Image]:
print_h("2: Writing img to disk...\n")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imwrite("Sunset_gray.png",gray)

# 3: [Reading/Displaying Video]:
print_h("3: Reading and displaying video from disk...\n")

vid = cv2.VideoCapture("Data/Megamind.avi")

while(vid.read()[0]):
    frame = vid.read()[1]
    cv2.imshow("Megamind",frame)
    k = cv2.waitKey(33)
    if k ==27:
        break
    
# 4: [Writing Video]:
print_h("4: Writing video to disk...\n")
vid = cv2.VideoCapture("Data\Megamind.avi")

# Extracting input video properties to be used for output videowriter initialization
inp_fps = vid.get(cv2.CAP_PROP_FPS)
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
size = (int(width),int(height))

out = cv2.VideoWriter("out.avi",cv2.VideoWriter_fourcc(*'MJPG'),inp_fps,size)
#output= cv2.VideoWrite(output.avi,cv2.VideoWrite_fourcc('M','J','P','G'),fps,size)

while(vid.read()[0]):
    
    frame = vid.read()[1]
    # Converting from bgr to rgb
    frame_rgb = frame[:,:,::-1]
    # Saving rgb frames inside new video
    out.write(frame_rgb)

# When everything done, release 
# the video capture and video 
# write objects
out.release()