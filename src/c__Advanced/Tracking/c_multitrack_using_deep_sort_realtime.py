import cv2
import numpy as np
from src.a__IP_Basics.utilities import putText,get_data,get_fileName,Gui,print_h
import time
import os
# Tensorflow required : For preprocessing and yolov3 stuff
import tensorflow as tf
# DeepSort-Realtime provided by pypy for multi-track
# Forced us to install opencv-python causing multiple issues xD (Double opencv install we needed contrib)
#                                                               (Remval of opencvpython caused other problems)
#                                                               (Upgraded opencv contrib to latest so that all required)
#                                                               (libraries are already available)
#                                                               Suggested Solution: Clone Repo and install from there
#                                                                                   after modifying Setup.py
from deep_sort_realtime.deepsort_tracker import DeepSort
# Deepsort inherently requires a multi-object detector as prerequisites, So we rely on yolo to do that
from src.c__Advanced.Tracking.yolov3.yolov3 import Create_Yolov3
from src.c__Advanced.Tracking.yolov3.utils import load_yolo_weights, image_preprocess, postprocess_boxes, nms, draw_bbox, read_class_names
from src.c__Advanced.Tracking.yolov3.configs import *

# Step 1: Loading our detector Yolo-V3. (Neccesary step of deep-sort algorithm)
input_size = YOLO_INPUT_SIZE
Darknet_weights = YOLO_V3_WEIGHTS#YOLO_DARKNET_WEIGHTS
if TRAIN_YOLO_TINY:
    Darknet_weights = YOLO_V3_TINY_WEIGHTS#YOLO_DARKNET_TINY_WEIGHTS
YoloV3 = Create_Yolov3(input_size=input_size)

load_yolo_weights(YoloV3, os.path.join(YOLO_FOLDER,Darknet_weights)) # use Darknet weights


def demo():
    print_h("[main]: Using deepsort for Multi-obj-Tracking.")
    
    gui = Gui()
    # Use CSRT when you need higher object tracking accuracy and can tolerate slower FPS throughput
    # Use KCF when you need faster FPS throughput but can handle slightly lower object tracking accuracy
    # Use MOSSE when you need pure speed
    ds_tracker = DeepSort(max_age=5)

    #Friends = "Data/NonFree\Friends\Friends_AllClothes.mp4"
    #Megamind = "Data/NonFree/Megamind.avi"
    #Window_Name = get_fileName(Megamind)
    vid_dirs,filenames = get_data("multitracking")
    data_iter = gui.selectdata(filenames)
    Window_Name = filenames[data_iter]
    #Window_Name = "Tracking (DeepSort)"
    #-- 1. Read the video stream
    cap = cv2.VideoCapture(vid_dirs[data_iter])
    #cap = cv2.VideoCapture(r"src\c__Advanced\Tracking\test.mp4")
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    Track_only = ["car"]
    times = []
    CLASSES = "model_data/coco/coco.names"
    NUM_CLASS = read_class_names(os.path.join(YOLO_FOLDER,CLASSES))
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())
    
    while True:
        _, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        else:
            # ------------------------------------- DETECTOR GIVES PREDICTIONS ---------------------------------
            try:
                #original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                original_image = frame.copy()
            except:
                break
            image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
            image_data = tf.expand_dims(image_data, 0)
            
            t1 = time.time()
            pred_bbox = YoloV3.predict(image_data)
            t2 = time.time()
            times.append(t2-t1)
            times = times[-20:]
            # ------------------------------------- OPTIONAL PROCESSING ----------------------------------------            
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]# Reshaping detector predicted bboxes
            pred_bbox = tf.concat(pred_bbox, axis=0) # I dont know XD

            score_threshold=0.3;iou_threshold=0.1
            bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold) # Some sort of processing
            bboxes = nms(bboxes, iou_threshold, method='nms') # Performing non-maximum suppression
            # ----------------------------------- POST PROCESSSING ---------------------------------------------
            # extract bboxes to bbs = [detection = (boxes (x, y, width, height), scores and names)]
            boxes, scores, names = [], [], []
            bbs = []
            for bbox in bboxes:
                if len(Track_only) !=0 and NUM_CLASS[int(bbox[5])] in Track_only or len(Track_only) == 0:
                    box = ([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
                    score = bbox[4]
                    name = NUM_CLASS[int(bbox[5])]
                    bbs.append((box,score,name))
            # ----------------------------------- TRACKING PREDICTED BBOXES ------------------------------------
            tracked_bboxes = []
            if len(bbs)!=0:
                tracks = ds_tracker.update_tracks(bbs, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    ltrb = track.to_ltrb()# Converting to [left(x), top(y), right(x'), bottom(y')] format     
                    class_name = track.get_det_class() #Get the class name of particular object
                    tracking_id = track.track_id # Get the ID for the particular track
                    index = key_list[val_list.index(class_name)] # Get predicted object index by object name
                    tracked_bboxes.append(ltrb.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
            # Calculating fps for deepsort.
            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            # draw detection on frame
            image = draw_bbox(original_image, tracked_bboxes, CLASSES=os.path.join(YOLO_FOLDER,CLASSES), tracking=True)
            image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            cv2.imshow(Window_Name,image)
            k = cv2.waitKey(30)
            if k==27:# If Esc Pressed Quit!
                break


if __name__ =="__main__":
    demo()