import cv2
import os
from src.utilities import putText



class Cascade_Detector:
    def __init__(self,object = "face"):
        self.detector = cv2.CascadeClassifier()
        
        self.category = object
        
        if object == "TrafficLight":
            cascade_path = "Data/haarcascades/haarcascade_trafficlight.xml"
        elif object == "eyes":
            cascade_path = 'Data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'
        else:
            cascade_path = 'Data/haarcascades/haarcascade_frontalface_alt.xml'
            
        self.cascade_path = os.path.join(os.getcwd(), cascade_path) # Get the absolute path.
        
        if not self.detector.load(cv2.samples.findFile(self.cascade_path)):
            print(f'--(!)Error loading {object} cascade')
            exit(0)
            
    def detect(self,img,display = False):
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # scaleFactor: How much image size is reduced at each image scale default : 1.1 (10%)
        # minNeighbors: How many neighbors each candidate rectangle should have to retain it.
        # minSize: Object size lower limit
        # max Size: Object size upper limit
        bboxes = self.detector.detectMultiScale(gray)
        
        if display and len(bboxes)!=0:
            for bbox in bboxes:
                x,y,w,h = bbox
                
                # Displaying the label
                putText(img,self.category,(x,y-20),bbox_size=(w,h))
                # Drawing the bbox around the detected object
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
                

def main():
    hc_detector = Cascade_Detector("TrafficLight")
    vid_path = "data/city.mp4"
    #-- 2. Read the video stream
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
        
    while True:
        ret, frame = cap.read()
        
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        
        # [Task]: Detect object [Traffic Light] in frame using Haar Cascade
        hc_detector.detect(frame,True)
        
        cv2.imshow("Megamind",frame)
        if cv2.waitKey(10) == 27:
            break
        
        
if __name__ == "__main__":
    main()