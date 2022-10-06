from ast import arg
import cv2
import numpy as np
from math import ceil
from collections import deque

import concurrent.futures
from multiprocessing import Array,freeze_support
import time

from recognition import face_recognition_dlib
from multi_object_tracking import multitracker
from utilities import putText,get_iou,to_ltrd,to_ltwh

# get all active child processes for the main process with no children
from multiprocessing import active_children
import sys
from os import path

import argparse


import face_recognition
# Reference (Cross-Platform-Path) = https://stackoverflow.com/questions/122455/handling-file-paths-cross-platform
import config

import subprocess


class secure_access_cv:

    def __init__(self):
                
        # [1: Face Detector] Loading cascade classifier as the fastest face detector in OpenCV
        self.faceCascade = cv2.CascadeClassifier()
        if not self.faceCascade.load(path.join(path.dirname(__file__),config.cascPathface)):
            print(f'--(!)Error loading face cascade from {config.cascPathface}')
            sys.exit(0)
        # [2: Object Tracker] Loading MultiTracker class for multiple face tracking after detectionh
        self.m_tracker = multitracker(config.tracker_type)
        # [3: Recognition] Loading Face Recognition performing recogntion on detected faces
        self.face_recog = face_recognition_dlib()
        # [4: Profiling] Creating deque object to act as a running average filter for a smoothed FPS estimation
        self.fps_queue = deque(maxlen=10)
        self.elapsed_time = 0
        # [5: Multiprocessing] Creating an object (executor) of the process pool executor in concurrent.futures for multiprocessing
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)
        self.start_time = None
        self.future = None
        self.face_names_chars = Array('c', 100) # Array of integer type of length 4 ==> Used here for sharing the computed bbox

        self.currently_recognizing = "Single"# 1: Single, Multiple, Appended
        
    def update_state(self,frame):
        ideal_processing_time = 28 # ms [Considering 30 fps]
        if self.elapsed_time < ideal_processing_time:
            waitTime = ceil(ideal_processing_time - self.elapsed_time)
        else: # Already too slow --> Wait for minimum possible time
            waitTime = 1
        
        if config.display_state:
            elapsed_time_sec = (self.elapsed_time)/1000
            fps = 1.0 / elapsed_time_sec if elapsed_time_sec!=0 else 100.00
            # Rolling average applied to get average fps estimation
            self.fps_queue.append(fps)
            fps = (sum(self.fps_queue)/len(self.fps_queue))
            #fps_txt = f"{self.tracker.mode}: ( {self.tracker.tracker_type} ) at {fps:.2f} FPS"
            fps_txt = f"Secure Access (CV) at {fps:.2f} FPS"
            cv2.putText(frame, fps_txt ,(20,40) ,cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0))
            cv2.putText(frame, f"Computed WaitTime = {waitTime}" ,(20,80) ,cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))
            cv2.putText(frame, f"Detector = {config.face_detector}, Mode = {self.m_tracker.mode}" ,(20,160) ,cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0))

        return waitTime

    def identify(self):
        # This function is purely developed for the purpose of extraction data from the child process (Face recognizer)
        # Once available and performing the neccesary processing steps to get it to the desired format.
        # Finally once we have the info... We perform the neccesary actions based on the identities
        elasped_time = time.time() - self.start_time
        #print("elasped_time = ",elasped_time)
        if (elasped_time % 1)< 0.2:
            #print("1 sec elasped... Check our executor for result!")
            if not self.future.running():
                # Asynchrnous plant detection has been completed. You may retrieve the plant location now
                # [Beware]: Time has elasped since estimated plantrow mask was passed. Drone would have moved by now
                #            Adjust for the drone pose change [Rotation & Position] applying odometry changes to the computed bbox
                self.face_names_chars = self.future.result()

                # 0) Transferring data from a shared array to a list
                face_names_chars_list = list(self.face_names_chars)
                # 1) Removing garbadge values
                if '0' in face_names_chars_list:
                    end_idx = face_names_chars_list.index('0')
                    face_names_chars_list = face_names_chars_list[0:end_idx]
                # 2) Creating string from the list of chars
                face_names_str = ''.join(face_names_chars_list)
                # 3) Seperating face-names on seperater ','
                face_names = face_names_str.split(",")

                # Based on names identified for each detected frame. We assign
                # a) Class : Name of the individual if known      else Unknown
                # b) Color : Green                  if authorized else Red
                for iter,name in enumerate(face_names):
                    self.m_tracker.Tracked_classes[iter] = name
                    if name!= "Unknown":
                        self.m_tracker.colors[iter] = (0,255,0)
                    else:
                        self.m_tracker.colors[iter] = (0,0,255)

    def activate_sa(self,vid_id,dataset_dir =""):
        print("Activating Secure Access....\n Authorized personnel only!")
        config.vid_id = vid_id
        
        # Get embeddings [encodings + Names]
        data = self.face_recog.get_embeddings(dataset_dir)

        if config.ignore_Warning_MMSF or isinstance(vid_id, str):
            # Debugging on a stored video...
            cap = cv2.VideoCapture(vid_id)
        else:
            # Get a reference to webcam #0 (the default one)
            cap = cv2.VideoCapture(vid_id,cv2.CAP_DSHOW)# Suppress Warning_MMSF by adding flag cv2.CAP_DSHOW Reference : https://forum.opencv.org/t/cant-grab-frame-what-happend/4578
        if not cap.isOpened():
            print(f"\n[Error]: Unable to create a VideoCapture object at {vid_id}")
            return
        
        # Adjusting downscale config parameter based on video size. If video frame is large we can downscale more without losing appropriate data
        vid_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        config.downscale = int(vid_width/config.exp_width) * config.downscale 
        #print(f"Size of video frame = (Width,Height) = ({vid_width},{vid_height}) with dowscaling at {config.downscale}")
                
        if config.face_detector == "hog":
            dth_frame = 15
        else:
            dth_frame = 4
        frame_iter = 0 # Current frame iteration
        while cap.isOpened():
            start_time = time.time() # Get the intial time
            # Grab a single frame of video
            ret, frame = cap.read()
            if not ret:
                print("No-more frames... :(\nExiting!!!")
                break
            frame_draw = frame.copy()
            
            faces = [] # (x,y,w,h) bbox format
            face_locations = [] # (top, right, bottom, left) css
            
            rgb_small_frame = None
            # Step 1: Detecting face on every dth_frame             
            if frame_iter%dth_frame==0:
                if config.face_detector=="hog":
                    # Resize frame of video to 1/4 size for faster face recognition processing
                    small_frame = cv2.resize(frame, (0, 0), fx=(1/config.downscale), fy=(1/config.downscale))
                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_small_frame = small_frame[:, :, ::-1]
                    if config.nvidia_gpu:
                        detector = "cnn" # Use CNN (more accurate)
                    else:
                        detector = "hog"
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame,model=detector)
                    for face_loc in face_locations:
                        face_loc_scaled = (face_loc[0]*config.downscale,face_loc[1]*config.downscale,face_loc[2]*config.downscale,face_loc[3]*config.downscale)
                        top, right, bottom, left = face_loc_scaled
                        cv2.rectangle(frame_draw,(left,top),(right,bottom),(255,0,0),4)
                else:
                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    faces = self.faceCascade.detectMultiScale(gray)          
                    for face in faces:
                        x,y,w,h = face
                        cv2.rectangle(frame_draw,(x,y),(x+w,y+h),(0,255,0),2)

            # Step 2: Start Tracking if faces are present
            if self.m_tracker.mode =="Detection":
                if len(faces)!=0 or len(face_locations)!=0: # Found atleast one face
                    if len(faces)>1 or len(face_locations)>1:
                        self.currently_recognizing = "Multiple"
                    else:
                        self.currently_recognizing = "Single"
                       
                    rgb_small_frame, face_locations,face_locations_frame = self.face_recog.preprocess(frame,frame_draw,rgb_small_frame,faces,face_locations)
                    
                    # Initialize tracker with bboxes or face_locations (based on detector used) 
                    if config.face_detector =="hog":
                        inp_formats = ["css"]*len(face_locations_frame)
                        faces_hog = list( map(to_ltwh,face_locations_frame,inp_formats) )
                        self.m_tracker.track(frame,frame_draw,faces_hog,"face") # Initialize Tracker
                    else:
                        self.m_tracker.track(frame,frame_draw,faces,"face") # Initialize Tracker
                    
                    # Step 3: Face Recognition using dlib (Child process)
                    # Perform recognition on detected faces as a seperate child process to avoid halting main process
                    self.future = self.executor.submit(self.face_recog.recognize,rgb_small_frame,face_locations,data)
                    self.start_time = time.time()
            else:
                # If detector detected faces other then already tracking. Append them to tracked faces
                new_bboxes = []
                if len(faces)!=0 or len(face_locations)!=0: # Found atleast one face
                    # Checking to see if detected faces are being new or not
                    if len(faces)!=0: # Using HaarCascade detector and detected a new faces
                        for i, face in enumerate(faces):
                            t_bboxes = list( map(to_ltrd,self.m_tracker.tracked_bboxes) )
                            face_ltrd = to_ltrd(face)
                            iou_list = [get_iou(face_ltrd,t_bbox) for t_bbox in t_bboxes]
                            iou = max(iou_list)
                            if iou > 0.5:#Found a match
                                if config.display_state:
                                    putText(frame_draw,f"iou of {self.m_tracker.Tracked_classes[0]} is {iou:.2f}", ( 20,140+(20*i) ) )
                                pass
                            elif iou < 0.2:# Detected face not already in tracking list
                                new_bboxes.append(face)
                    else:
                        for i, face_loc in enumerate(face_locations):
                            face_loc_scaled = (face_loc[0]*config.downscale,face_loc[1]*config.downscale,face_loc[2]*config.downscale,face_loc[3]*config.downscale)
                            t_bboxes = list( map(to_ltrd,self.m_tracker.tracked_bboxes) )
                            face_ltrd = to_ltrd(face_loc_scaled,"css")
                            face      = to_ltwh(face_loc_scaled,"css")
                            iou_list = [get_iou(face_ltrd,t_bbox) for t_bbox in t_bboxes]
                            iou = max(iou_list)
                            if iou > 0.5:#Found a match
                                if config.display_state:
                                    putText(frame_draw,f"iou of {self.m_tracker.Tracked_classes[0]} is {iou:.2f}", ( 20,140+(20*i) ) )
                                pass
                            elif iou < 0.2:# Detected face not already in tracking list
                                new_bboxes.append(face) 

                if len(new_bboxes)!=0:
                    already_tracked = [np.array(t_bbox) for t_bbox in self.m_tracker.tracked_bboxes]
                    init_bboxes = [*already_tracked,*new_bboxes]
                    self.m_tracker.mode = "Detection"
                    self.m_tracker.track(frame,frame_draw,init_bboxes,"face") # Appending new found bboxes to already tracked

                    # [New evidence added]: If child process is not already executing --> Start recognizer
                    rgb_small_frame, face_locations,_ = self.face_recog.preprocess(frame,frame_draw,rgb_small_frame,init_bboxes)
                    # Perform recognition on detected faces as a seperate child process to avoid halting main process
                    self.future = self.executor.submit(self.face_recog.recognize,rgb_small_frame,face_locations,data)
                    self.start_time = time.time()

                    self.currently_recognizing = "Appended"
                else:
                    # Check if recognizer has finshed processing?
                    # a) If friendly face present --> Allow access! + Replace (track_id <---> recognized name) and color
                    # b) else                     --> Revoke previliges
                    self.identify()
                    # Track detected faces
                    self.m_tracker.track(frame,frame_draw)
            
            # Updating state to current for displaying...
            waitTime = self.update_state(frame_draw)

            if config.display_state:
                # get all active child processes
                children = active_children()
                cv2.putText(frame_draw,f"Recognizing {self.currently_recognizing} face/es  with {len(children)} subprocess running!", ( 20,200 ) ,cv2.FONT_HERSHEY_DUPLEX, 1, (128,128,128))
                cv2.putText(frame_draw, f"Processing took  = {self.elapsed_time:.2f} ms" ,(20,120) ,cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,128))
            
            cv2.imshow('Video', frame_draw)
            # Hit 'Esc' on the keyboard to quit!
            k = cv2.waitKey(waitTime)
            self.elapsed_time = (time.time() - start_time)*1000 # ms
            if k==27:#Esc pressed
                break
            frame_iter = frame_iter + 1

        # Release handle to the webcam
        cap.release()
        cv2.destroyAllWindows()

    def __getstate__(self):
        d = self.__dict__.copy()
        # Delete all unpicklable attributes.
        del d['executor']
        del d['future']
        del d['face_names_chars']
        del d['m_tracker']
        return d




def secure_live():
    # Step 1: Loading the training and test data
    secure_acc = secure_access_cv()
    
    secure_acc.activate_sa(0,dataset_dir= config.authorized_dir)

def demo():
    # Step 1: Loading the training and test data
    secure_acc = secure_access_cv()

    test_vid_path = r"Data\NonFree\Friends\Avengers_endgame_Assemble.mp4"
    #test_vid_path = r"Data\NonFree\Friends\Jeff_Vs_Elon.mp4"
    #test_vid_path = r"Data\NonFree\Friends\Friends_AllClothes.mp4"
    #test_vid_path = r"Data\NonFree\Friends\Friends_lightning round.mp4"

    dataset_dir = r"src/c__Advanced\Face_Recognition\training-data\dlib\avengers_endgame"
    #dataset_dir = r"src/c__Advanced\Face_Recognition\training-data\dlib\known_people"
    #dataset_dir = r"src/c__Advanced\Face_Recognition\training-data\dlib\friends"

    secure_acc.activate_sa(path.abspath(test_vid_path),dataset_dir= path.abspath(dataset_dir))


if __name__== "__main__":
    # Pyinstaller fix (Reference : https://stackoverflow.com/a/32677108/11432131)
    freeze_support()

    parser = argparse.ArgumentParser("--Secure Access--")
    parser.add_argument("-d","--debug", help="Debug App", type=bool,default=False)
    parser.add_argument("-exp","--experimental", help="Turn Experimental features On/OFF", type=bool,default=False)
    parser.add_argument("-w","--ignore_warnings", help="Ignore warning or not", type=bool,default=True)
    parser.add_argument("-s","--display_state", help="display current (app) state", type=bool,default=False)
    parser.add_argument("-fd","--face_detector", help="type of detector for face detection", type=str,default="hog")
    parser.add_argument("-ft","--tracker_type", help="type of tracker for face tracking", type=str,default="MOSSE")
    parser.add_argument("-ds","--downscale", help="downscale amount for faster recognition", type=int,default=2)# Default = 2 -> Seems adequate for video size of (Width,Height) = (640,480) ... After downsizing (320,240)
    parser.add_argument("-rt","--recog_tolerance", help="Adjust tolerance for recognition e.g: (Strict < 0.5 < Loose) ", type=float,default=0.6)# Default = 2 -> Seems adequate for video size of (Width,Height) = (640,480) ... After downsizing (320,240)
    parser.add_argument("-nu","--new_user", help="Add new user to list of authorized personnels", type=str,default=None)
    args = parser.parse_args()
    
    # Set config variables based on parsed arguments
    config.debug = args.debug
    config.ignore_Warning_MMSF = args.ignore_warnings
    config.display_state = args.display_state
    config.face_detector = args.face_detector
    config.tracker_type = args.tracker_type
    config.downscale = args.downscale
    config.recog_tolerance = args.recog_tolerance
    config.new_user = args.new_user
    config.experimental = args.experimental
    
    if config.experimental:
        # Experimental: Features under rigorous testing.
        try:
            subprocess.check_output('nvidia-smi')
            print('Nvidia GPU detected!')
            config.nvidia_gpu = True
        except Exception: # this command not being found can raise quite a few different errors depending on the configuration
            print('No Nvidia GPU in system!')

    if config.debug:
        demo()
    else:
        secure_live()