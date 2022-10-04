import cv2
from utilities import print_h

# 1: [Reading/Displaying Image]:
print_h("1: Reading and displaying image from disk...\n")
img = cv2.imread("Data/CV/3.jpg")
img = cv2.resize(img,None,fx=0.5,fy=0.5)
cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow("img",img)

# Displaying image dimensions (rows,cols,channels)
print("- Retreiving img details...\n")
rows,cols,channels = img.shape
print(f"- Img has size (rows,cols,channels) = ({rows},{cols},{channels})\n")
cv2.waitKey(0)
cv2.destroyAllWindows()

# 2: [Writing Image]:
print_h("2: Writing img to disk...\n")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imwrite(r"src\a__IP_Basics\sunset.png",gray) #.png (lossless compression)
cv2.waitKey(0)


# 3: [Reading/Displaying Video]:
print_h("3: Reading and displaying video from disk...\n")
vid = cv2.VideoCapture("Data\CV\Megamind.avi")

while(vid.read()[0]):
    frame = vid.read()[1]
    cv2.imshow("frame",frame)
    k = cv2.waitKey(33)
    if k==27: # Break on Esc
        break
cv2.waitKey(0)

# 4: [Writing Video]:
print_h("4: Writing video to disk...\n")
vid = cv2.VideoCapture("Data\CV\Megamind.avi")

# We need to set resolutions.
# so, convert them from float to integer.
inp_fps = vid.get(cv2.CAP_PROP_FPS)
frame_width = int(vid.get(3))
frame_height = int(vid.get(4))
size = (frame_width, frame_height)
# Below VideoWriter object will create a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter(r"src\a__IP_Basics\video.avi", cv2.VideoWriter_fourcc(*'MJPG'),inp_fps, size)

while(vid.read()[0]):
    frame = vid.read()[1]
    # Converting from bgr to rgb
    frame_rgb = frame[:,:,-1]
    # Saving rgb frames inside new video
    result.write(frame_rgb)

# When everything done, release 
# the video capture and video 
# write objects
result.release()