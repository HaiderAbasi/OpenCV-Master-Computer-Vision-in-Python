import cv2
import numpy as np
from src.a__IP_Basics.utilities import get_data,Gui,print_h,putText,generate_vibrant_color,find_centroid,closest_bbox_to_pt,add_to_dict_deque
import time
import os
from collections import deque
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
from src.c__Advanced.Tracking.yolov3.utils import load_yolo_weights, image_preprocess, postprocess_preds, draw_bbox, read_class_names
from src.c__Advanced.Tracking.yolov3.configs import *

# Step 1: Loading our detector Yolo-V3. (Neccesary step of deep-sort algorithm)
input_size = YOLO_INPUT_SIZE
Darknet_weights = os.path.join(YOLO_FOLDER,YOLO_V3_WEIGHTS)#YOLO_DARKNET_WEIGHTS
if TRAIN_YOLO_TINY:
    Darknet_weights = os.path.join(YOLO_FOLDER,YOLO_V3_TINY_WEIGHTS)#YOLO_DARKNET_TINY_WEIGHTS
YoloV3 = Create_Yolov3(input_size=input_size)

load_yolo_weights(YoloV3, Darknet_weights) # use Darknet weights


def demo():
    print_h("[main]: Using deepsort for Multi-obj-Tracking.")
    
    ds_tracker = DeepSort(max_age=5)
    Track_only = ["person"]

    gui = Gui()
    vid_dirs,filenames = get_data("DeepSort")
    data_iter = gui.selectdata(filenames)
    Window_Name = filenames[data_iter]

    #-- 1. Read the video stream
    cap = cv2.VideoCapture(vid_dirs[data_iter])
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    times = deque(maxlen=20)
    CLASSES = "model_data/coco/coco.names"
    NUM_CLASS = read_class_names(os.path.join(YOLO_FOLDER,CLASSES))
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())
    
    # Maintain a dictionary
    trajectories = {}
    tracked_clr = {}
    # Get the video width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask = np.zeros((height,width,3),np.uint8)
    
    selected_ID = None
    gui.select_pt(Window_Name)
    while True:
        _, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        else:
            # ------------------------------------- DETECTOR GIVES PREDICTIONS ---------------------------------
            try:
                original_image = frame.copy()
            except:
                break
            image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
            image_data = tf.expand_dims(image_data, 0)
            
            t1 = time.time()
            pred_bbox = YoloV3.predict(image_data)
            
            # -------------------------------- Postprocess Predictions ------------------------------------

            bbs = postprocess_preds(original_image, pred_bbox, NUM_CLASS, Track_only)

            # ----------------------------------- DeepSort Tracking ------------------------------------
            tracked_bboxes = []
            ltrbs = []
            if len(bbs)!=0:
                tracks = ds_tracker.update_tracks(bbs, frame=frame) # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )
                for track in tracks:
                    if not track.is_confirmed():
                        #skip track and continue to next
                        continue
                    ltrb = track.to_ltrb()# Converting to [left(x), top(y), right(x'), bottom(y')] format     
                    class_name = track.get_det_class() #Get the class name of particular object
                    tracking_id = track.track_id # Get the ID for the particular track
                    index = key_list[val_list.index(class_name)] # Get predicted object index by object name
                    tracked_bboxes.append(ltrb.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
                    ltrbs.append(ltrb)
                    
                    add_to_dict_deque(trajectories,tracking_id,find_centroid(ltrb))
                    if tracking_id not in tracked_clr:
                        tracked_clr[tracking_id] = generate_vibrant_color()
                        
            if gui.clicked_pt:
                clst_bb_idx = closest_bbox_to_pt(gui.clicked_pt,ltrbs)[1]
                selected_ID = tracked_bboxes[clst_bb_idx][4]
                mask = np.zeros_like(frame)
                #cv2.circle(mask,gui.clicked_pt[0],5,(255,0,0),2)
                gui.clicked_pt.clear()
                            

            # ----------------------------------- Displaying Tracked Objects ------------------------------------
            t2 = time.time()
            times.append(t2-t1)
            # Calculating fps for deepsort.
            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            # draw detection on frame
            tracking_options = { 'mask'       : mask       , 'trajectories': trajectories,
                                 'tracked_clr': tracked_clr, 'ID_to_track' : selected_ID }
            image = draw_bbox(original_image, tracked_bboxes, **tracking_options, CLASSES=os.path.join(YOLO_FOLDER,CLASSES), tracking=True)
            image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

            if selected_ID:
                putText(image, f"Monitoring = {selected_ID}", (image.shape[1]-300, 30))

            cv2.imshow(Window_Name,image)            
            k = cv2.waitKey(30)
            if k==27:# If Esc Pressed > Quit!
               break

if __name__ =="__main__":
    demo()