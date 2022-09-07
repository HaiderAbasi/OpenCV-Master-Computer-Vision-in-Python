import cv2
import numpy as np
import time
import os
import math
from a3_Advanced.utilities import imshow,get_optimal_font_scale,draw_text




class dnn:
    def __init__(self,model_path = None, weights_path = None):
        if model_path==None:
            model_path = 'Data/NonFree\dnn\yolov3.cfg'
        if weights_path==None:
            weights_path = 'Data/NonFree\dnn\yolov3.weights'
        # Give the configuration and weight files for the model and load the network.
        self.net = cv2.dnn.readNetFromDarknet(model_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # Load names of classes and get random colors
        self.classes = open('Data/NonFree/dnn/coco.names').read().strip().split('\n')
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype='uint8')

        # determine the output layer
        ln = self.net.getLayerNames()
        #print(len(ln), ln)
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


    def post_process(self,img, outputs, conf):
        H, W = img.shape[:2]

        boxes = []
        confidences = []
        classIDs = []

        for output in outputs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                # cv2.rectangle(img, p0, p1, WHITE, 1)

        FONT_SCALE = 2e-3  # Adjust for larger font size in all images
        THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in self.colors[classIDs[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.2f}".format(self.classes[classIDs[i]], confidences[i])

                thickness = math.ceil(min(w, h) * THICKNESS_SCALE)
                #fnt_scale = get_optimal_font_scale(text,w)[0]


                height, width= img.shape[:2]
                font_scale = min(width, height) * FONT_SCALE
                thickness = math.ceil(min(width, height) * THICKNESS_SCALE)
                draw_text(img, text, font_scale=font_scale, pos= (x, y - 15),font_thickness=1,text_color=(255,0,255))
                #cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, fnt_scale, color, thickness)


    def trackbar(self,x,img0,outputs):

        img = img0.copy()
        conf = x/100
        
        self.post_process(img, outputs, conf)

        cv2.imshow('window', img)
        
    def detect(self,img0):
        img = img0.copy() # Duplicating input image

        # construct a blob from the image
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        
        self.net.setInput(blob)
        
        t0 = time.time()
        outputs = self.net.forward(self.ln) # Predicting using Yolov3
        t = time.time()

        # combine the 3 output groups into 1 (10647, 85)
        # large objects (507, 85)
        # medium objects (2028, 85)
        # small objects (8112, 85)
        outputs = np.vstack(outputs)

        self.post_process(img, outputs, 0.5)

        cv2.namedWindow('window')
        #cv2.createTrackbar('confidence', 'blob', 50, 101,lambda x: self.trackbar2(x,r0,outputs) )
        cv2.createTrackbar('confidence', 'window', 50, 100, lambda x: self.trackbar(x,img0,outputs))
        # GoodRead https://stackoverflow.com/questions/40680795/cv2-createtrackbar-pass-userdata-parameter-into-callback

        #cv2.displayOverlay('window', f'forward propagation time={t-t0}')
        cv2.putText(img,f'forward propagation time={t-t0}',(50,100),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),1)

        cv2.imshow('window',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect_in_video(self,vid_path,save_vid = False):

        vid = cv2.VideoCapture(vid_path)

        if save_vid:
            # Default resolutions of the frame are obtained.The default resolutions are system dependent.
            # We convert the resolutions from float to integer.
            frame_width = int(vid.get(3))
            frame_height = int(vid.get(4))
            filename, ext = os.path.splitext(vid_path)
            # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
            out = cv2.VideoWriter(f'{filename}_yolo_V3.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (frame_width,frame_height))

        while(vid.isOpened()):
            ret,frame = vid.read()
            if ret:
                start_time = time.time()
                # Detect using Yolo

                # construct a blob from the image
                blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
                
                self.net.setInput(blob)
                
                t0 = time.time()
                outputs = self.net.forward(self.ln) # Predicting using Yolov3
                t = time.time()

                # combine the 3 output groups into 1 (10647, 85)
                # large objects (507, 85)
                # medium objects (2028, 85)
                # small objects (8112, 85)
                outputs = np.vstack(outputs)

                self.post_process(frame, outputs, 0.75)

                FONT_SCALE = 3e-3  # Adjust for larger font size in all images
                THICKNESS_SCALE = 1e-3  # Adjust for larger thickness in all images

                height, width= frame.shape[:2]
                font_scale = min(width, height) * FONT_SCALE
                thickness = math.ceil(min(width, height) * THICKNESS_SCALE)

                #print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop

                fps_txt = f"FPS: = {1.0 / (time.time() - start_time):.2f}"
                #txt = f'forward propagation time={(t-t0):.3f}'
                #fnt_scale= get_optimal_font_scale(txt,int(frame.shape[1]))[0]
                #cv2.putText(frame,txt,(50,50),cv2.FONT_HERSHEY_PLAIN,font_scale,(255,0,0),thickness)
                draw_text(frame, fps_txt, font_scale=font_scale, pos=(10, 20))

                cv2.imshow('Live Obj-det',frame)
                if save_vid:
                    out.write(frame)
                k=cv2.waitKey(1)
                if k==27:
                    break
            else:
                if save_vid:
                    out.release()
                print("Video Ended")
                break
        if save_vid:
            out.release()


class cascade_detector:

    def __init__(self,face_cascade_path = None,eyes_cascade_path = None):

        if face_cascade_path == None:
            face_cascade_path = 'Data/NonFree/haarcascades/haarcascade_frontalface_alt.xml'
        if eyes_cascade_path == None:
            eyes_cascade_path = 'Data/NonFree/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

        self.face_cascade = cv2.CascadeClassifier()
        self.eyes_cascade = cv2.CascadeClassifier()
        #-- 1. Load the cascades
        if not self.face_cascade.load(face_cascade_path):
            print('--(!)Error loading face cascade')
            exit(0)
        if not self.eyes_cascade.load(eyes_cascade_path):
            print('--(!)Error loading eyes cascade')
            exit(0)

    def detectAndDisplay(self,frame,display = False,detect_eyes = False):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame_gray = cv2.equalizeHist(frame_gray)
        #-- Detect faces
        faces = self.face_cascade.detectMultiScale(frame_gray)
        if display:
            for (x,y,w,h) in faces:
                center = (x + w//2, y + h//2)
                frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
                if detect_eyes:
                    faceROI = frame_gray[y:y+h,x:x+w]
                    #-- In each face, detect eyes
                    eyes = self.eyes_cascade.detectMultiScale(faceROI)
                    for (x2,y2,w2,h2) in eyes:
                        eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                        radius = int(round((w2 + h2)*0.25))
                        frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
            cv2.imshow('Capture - Face detection', frame)
        return faces

class detection:
    def __init__(self,method="cascade"):
        if method == "cascade":
            self.detector = cascade_detector()
        elif method == "yolo":
            self.detector = dnn()
        else:
            print(f"Unknown method specified ({method})")

        self.method = method # Store selected method as an instance variable of class

    def detect(self,img = None,vid_path = "",display = False):
        
        bboxes = []
        if self.method == "cascade":
            bboxes = self.detector.detectAndDisplay(img,display)
        elif self.method =="yolo":
            if vid_path=="":
                self.detector.detect(img)
            else:
                filename, ext = os.path.splitext(vid_path)
                vid_fomrats = [".mp4", ".avi", ".mov", ".mpeg", ".flv", ".wmv",".webm"]
                if ext in vid_fomrats:
                    self.detector.detect_in_video(vid_path)
                else:
                    print(f"Unknown vid path ({vid_path})")
        return bboxes





def main():


    Friends = "Data/NonFree\Friends\Friends_AllClothes.mp4"
    Megamind = "Data/NonFree/Megamind.avi"
    #-- 1. Read the video stream
    cap = cv2.VideoCapture(Megamind)
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    
    detection_ = detection()
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        bboxes = detection_.detect(frame,display=True)

        if cv2.waitKey(1) == 27:
            break
    
    detection_dnn = detection("yolo")

    # #img = cv2.imread('Data/NonFree/test/view.jpg')
    # img = cv2.imread('Data\CV\drag_race.jpg')

    # detection_dnn.detect(img)

    

    vid_path = "Data/NonFree\sheepAndpeople.mp4"
    dashcam_street = "Data/NonFree\dash_view.webm"


    detection_dnn.detect(vid_path = dashcam_street)




if __name__=="__main__":
    main()