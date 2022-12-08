import cv2
import numpy as np
import time
import os
import math
from src.utilities import putText,print_h,disp_fps

from loguru import logger

class Dnn:
    
    def __init__(self,model_path = None,weights_path = None,classes_path = None,conf = 0.75,colors = [(255,0,0,0,0,255,)]):
        
        if model_path == None:
            model_path = r'Data\Dnn\yolov3.cfg'
        if weights_path==None:
            weights_path = r'Data\Dnn\yolov3.weights'
        
        # Read a model stored in Darknet Model format    
        self.net = cv2.dnn.readNetFromDarknet(model_path,weights_path)
        # Set Target Device for computations
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # Target Device For computation
        
        # Set default detection classes to coco dataset path
        self.classes_path = r'Data/Dnn/coco.names'
        
        # If the user specifies a classes_path, Use that
        if classes_path!= None:
            self.classes_path= classes_path
        
        # Reading the classes object detector can detect    
        self.classes = open(self.classes_path).read().strip().split('\n')
        
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')

        #  Getting the output layers (82,94,106) names
        self.ln = self.net.getUnconnectedOutLayersNames()  

        # Setting the initial confidence parameter
        self.conf = conf


    def post_process(self,img,outputs,conf):
        
        H, W = img.shape[:2]
        boxes = []
        confidences = []
        classIDs = []
        
        # [bbox, Po , P_80_classes]
        for output in outputs:
            # Retrieve max confidence
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # Append as a detection if confidence greater then threshold
            if confidence >conf:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)
        
        # Stage 2: Removing overlapping region with IOU > threshold in NonMaxSuppression
        indices = cv2.dnn.NMSBoxes(boxes,confidences,conf,0.1)
        
        # Draw detected bboxes along with their predicted class + confidence
        if len(indices)>0:
            for i in indices.flatten():
                # Drawing bbox
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
                # Display predicted class + confidence
                text = "{}: {:.2f}".format(self.classes[classIDs[i]], confidences[i])
                FONT_SCALE = 2e-3  # Adjust for larger font size in all images
                THICKNESS_SCALE = 1e-3 
                height, width = img.shape[:2]
                font_scale = min(width,height)*FONT_SCALE
                thickness = math.ceil((min(width,height)*THICKNESS_SCALE))
                putText(img, text,  org= (x, y - 15),fontScale=font_scale,color=(255,0,255),thickness=thickness)

        
    def detect(self,frame):
                
        start_time = time.time()
        # Creates 4-dimensional blob[images,channels,w,h] from image. requred as the input to yolo V3.
        # ScaleFactor = scale our images by some factor
        # size =  size that the Network(YoloV3) expects
        # SwapRB = boolean for swaping first <-> last channel
        # crop = Crop image after resizing T/F
        # blobFromImage(image,scaleFactor,Size,swapRb,crop)
        blob = cv2.dnn.blobFromImage(frame,1/255.0,(416,416),swapRB=True,crop = False)
        
        # Sets the new input value for the network.
        self.net.setInput(blob)
        # Forward Propogation (Prediction Using object detector)
        # Yolo -> Stage 1 + Stage 3 
        outputs = self.net.forward(self.ln) 
        # Outputs:  vectors of lenght 85
        # [Po,bbox,P_80_classes]
        
        # combine the 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)
        
        # Non-max Suprresion + Drawing predicted bbox and Class
        self.post_process(frame,outputs, self.conf)
        
        # Displaying fps of yolo detections
        disp_fps(frame,start_time)
                



def main():

    print_h("[Main] Performing (obj) Detection using yolo-v3 [Dnn-module].\n")
    
    # Create object of Dnn Class
    detection_yolo = Dnn()
    
    # Object detection in video
    dashcam_street = r"Data\dash_view.webm"
    vid = cv2.VideoCapture(dashcam_street)
        
    while(vid.isOpened()):
        ret,frame = vid.read()
        
        if ret:
            detection_yolo.detect(frame)
            cv2.imshow('Live Obj-det',frame)
            k=cv2.waitKey(1)
            if k==27:
                break
        else:
            print("Video Ended")
            break
    
    # Using the Yolo-v7-tiny trained on a custom dataset for soccer fans.
    model_path = r"Data\dnn\yolo_v7\football\yolov7-tiny_TL.cfg"
    weights_path = r"Data\dnn\yolo_v7\football\yolov7-tiny_TL_best.weights"
    classes_path = r"Data\dnn\yolo_v7\football\football.names"
    detection_yolo = Dnn(model_path,weights_path,classes_path,0.4)
    
    # Object detection in video
    richarlson_goal = r"Data\richarlison_goal.mp4"
    vid = cv2.VideoCapture(richarlson_goal)
        
    while(vid.isOpened()):
        ret,frame = vid.read()
        
        if ret:
            detection_yolo.detect(frame)
            cv2.imshow('Live Obj-det',frame)
            k=cv2.waitKey(1)
            if k==27:
                break
        else:
            print("Video Ended")
            break

if __name__=="__main__":
    
    main()