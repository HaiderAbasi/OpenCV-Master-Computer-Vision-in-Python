import face_recognition
import cv2
import numpy as np
import os
from imutils import paths
import pickle
from imutils import build_montages
from src.c__Advanced.Tracking.a_object_tracking import Tracking
from src.c__Advanced.utilities import putText
import time

from multiprocessing import Array
import concurrent.futures

class face_recognition_dlib():

    def __init__(self):
        #self.default_embeddings_path = os.path.join(os.getcwd(),"models/live_embeddings-face_enc")
        self.default_embeddings_path = os.path.join(os.path.dirname(__file__),"models/live_embeddings-face_enc")
        #find path of xml file containing haarcascade file
        # cascPathface = r"src/c__Advanced\Face_Recognition\Secure_Access\models\haarcascade_frontalface_alt2.xml"
        # load the harcaascade in the cascade classifier
        # self.faceCascade = cv2.CascadeClassifier()
        # if not self.faceCascade.load(cascPathface):
        #     print('--(!)Error loading face cascade')
        #     exit(0)
        self.tracker = Tracking("KCF")
        
        # Creating an object (executor) of the process pool executor in concurrent.futures for multiprocessing
        self.executor = concurrent.futures.ProcessPoolExecutor()
        self.start_time = None
        self.recognize_results = None
        # To share data acroos multiple processes we need to create a shared adress space. 
        # Here we use multiprocessing.Array which stores an Array of c data types and can be shared along multiple processes
        self.face_names_chars = Array('c', 100) # Array of integer type of length 4 ==> Used here for sharing the computed bbox
        #self.t_bbox = Array('i', 4) # Array of integer type of length 4 ==> Used here for sharing the computed bbox

    def __getstate__(self):
        d = self.__dict__.copy()
        # Delete all unpicklable attributes.
        del d['executor']
        del d['recognize_results']
        del d['face_names_chars']
        del d['tracker']
        return d

    def generate_embeddings(self,data_dir,make_default = False):

        foldername = os.path.basename(data_dir)
        embeddings_path = os.path.join(os.path.dirname(__file__),f"models/{foldername}-live_embeddings-face_enc")
        #get paths of each file in folder named Images
        #Images here contains my data(folders of various persons) i.e. Friends/..
        imagePaths = list(paths.list_images(data_dir))
        knownEncodings = []
        knownNames = []
        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            name = imagePath.split(os.path.sep)[-2]
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Use Face_recognition to locate faces
            boxes = face_recognition.face_locations(rgb,model='hog')
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)
        #save emcodings along with their names in dictionary data
        data = {"encodings": knownEncodings, "names": knownNames}

        #use pickle to save data into a file named after the data folder for later use
        f = open(embeddings_path, "wb")
        f.write(pickle.dumps(data))
        f.close()

        if make_default:
            #use pickle to save data into a file named after the data folder for later use
            f = open(self.default_embeddings_path, "wb")
            f.write(pickle.dumps(data))
            f.close()
    
    def recognize(self,test_path,embeddings_path = ""):
        images = []
        titles = []
        if embeddings_path!="":
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(embeddings_path, "rb").read())
        else:            
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(self.default_embeddings_path, "rb").read())
        
        if os.path.isdir(test_path): # If path is a directory to unknown images, loop over them and perform recogntion one by one
            #imgs_paths = os.listdir(test_path)
            imgs_paths = [os.path.join(test_path, img_name) for img_name in os.listdir(test_path)]
        else:
            imgs_paths = [test_path]
        
        for img_path in imgs_paths:
            #Find path to the image you want to detect face and pass it here
            image = cv2.imread(img_path)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #convert image to Greyscale for haarcascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60, 60),flags=cv2.CASCADE_SCALE_IMAGE)
            
            # the facial embeddings for face in input
            encodings = face_recognition.face_encodings(rgb)
            names = []
            # loop over the facial embeddings incase
            # we have multiple embeddings for multiple fcaes
            for encoding in encodings:
                #Compare encodings with encodings in data["encodings"]
                #Matches contain array with boolean values and True for the embeddings it matches closely
                #and False for rest
                matches = face_recognition.compare_faces(data["encodings"],encoding)
                #set name =inknown if no encoding matches
                name = "Unknown"
                # check to see if we have found a match
                if True in matches:
                    #Find positions at which we get True and store them
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        #Check the names at respective indexes we stored in matchedIdxs
                        name = data["names"][i]
                        #increase count for the name we got
                        counts[name] = counts.get(name, 0) + 1
                        #set name which has highest count
                        name = max(counts, key=counts.get)
            
                    # update the list of names
                    names.append(name)
                    # loop over the recognized faces
                    for ((x, y, w, h), name) in zip(faces, names):
                        # rescale the face coordinates
                        # draw the predicted face name on the image
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
                images.append(image)
                titles.append(name)
                #cv2.imshow(f"Recognized Face: {name}", image)
                #cv2.waitKey(1)
        # Displaying image and threshold result
        montage = build_montages(images,None,None,titles,True,True)
        for montage_img in montage: 
            #imshow("Found Clusters",cluster,cv2.WINDOW_AUTOSIZE)
            cv2.imshow("Face Recognition",montage_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def live_recognition(self,vid_id = 0,dataset_dir="",embeddings_path=""):
        # This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
        # other example, but it includes some basic performance tweaks to make things run a lot faster:
        #   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
        #   2. Only detect faces in every other frame of video.

        # PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
        # OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
        # specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
        if embeddings_path!="":
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(embeddings_path, "rb").read())
        elif dataset_dir!="":
            self.generate_embeddings(dataset_dir)
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(self.default_embeddings_path, "rb").read())
        elif os.path.isfile(self.default_embeddings_path):
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(self.default_embeddings_path, "rb").read())
        else:
            print(f"Face-embedddings {self.default_embeddings_path} is not available.\nProvide dataset to generate new embeddings...")
            exit(0)


        # Get a reference to webcam #0 (the default one)
        video_capture = cv2.VideoCapture(vid_id)

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []

        process_this_frame = True
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Only process every other frame of video to save time
            if process_this_frame:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    #Compare encodings with encodings in data["encodings"]
                    #Matches contain array with boolean values and True for the embeddings it matches closely
                    #and False for rest
                    matches = face_recognition.compare_faces(data["encodings"], face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = data["names"][best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame


            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                if name!= "Unknown":
                    color = (0,255,0)
                else:
                    color = (0,0,255)
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    #def recognize_(self,rgb_small_frame,face_locations,encodings,names):
    def recognize_(self,rgb_small_frame,face_locations,data):

        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names_chars = []
        for face_encoding in face_encodings:
            #Compare encodings with encodings in data["encodings"]
            #Matches contain array with boolean values and True for the embeddings it matches closely
            #and False for rest
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            
            name = "Unknown"
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = data["names"][best_match_index]
            if len(face_names_chars)!=0:
                face_names_chars.append(',')
            face_names_chars = face_names_chars + list(name)
        print(f"face_names_chars initial = {face_names_chars}")
        print("data[names] = ",data["names"])
            
        if (len(face_names_chars)>=100):
            face_names_chars = face_names_chars[0:100]
        else:
            lst = ['0'] * (100-len(face_names_chars))
            face_names_chars = face_names_chars + lst

        print(f"len(face_names_chars) = {len(face_names_chars)}")
        print(f"face_names_chars finally = {face_names_chars}")
        return face_names_chars

    def live_recognition_fast(self,vid_id = 0,dataset_dir="",embeddings_path=""):
        # This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
        # other example, but it includes some basic performance tweaks to make things run a lot faster:
        #   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
        #   2. Only detect faces in every other frame of video.

        # PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
        # OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
        # specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
        if embeddings_path!="":
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(embeddings_path, "rb").read())
        elif dataset_dir!="":
            foldername = os.path.basename(dataset_dir)
            embeddings_path = os.path.join(os.path.dirname(__file__),f"models/{foldername}-live_embeddings-face_enc")
            if not os.path.isfile(embeddings_path): # Only generate if not already present
                self.generate_embeddings(dataset_dir,True)
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(self.default_embeddings_path, "rb").read())
        elif os.path.isfile(self.default_embeddings_path):
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(self.default_embeddings_path, "rb").read())
        else:
            print(f"Face-embedddings {self.default_embeddings_path} is not available.\nProvide dataset to generate new embeddings...")
            exit(0)


        # Get a reference to webcam #0 (the default one)
        video_capture = cv2.VideoCapture(vid_id)

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []

        frame_iter = 0

        elaps_pre_time = 0
        elaps_det_time = 0
        elaps_recog_time = 0
        elaps_track_time = 0

        while True:
            start_time = time.time()

            ## Preprocess timing
            preprocess_stime = time.time()
            # Grab a single frame of video
            ret, frame = video_capture.read()
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            #if ( (time.time() - preprocess_stime) > elaps_pre_time ):
            elaps_pre_time = time.time() - preprocess_stime 
            ## Preprocess timing

            if self.tracker.mode=="Detection":
                print(f"Current Frame = {frame_iter}")
                if frame_iter%3==0:
                    print(f"At {frame_iter} frame Detector was applied")

                    ## detection timing
                    detection_stime = time.time()
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    
                    elaps_det_time = time.time() - detection_stime
                    ## detection timing

                    ## Recognition timing
                    Recognition_stime = time.time()
                    # Schedule a function in the executor and returns a future object (results)
                    if self.start_time == None:
                        #self.p_detector_results = self.executor.submit(self.p_detector.detect,dv_est_nxtrow,self.avg_plant_width)
                        #face_names = self.recognize_(rgb_small_frame,face_locations,data)
                        self.recognize_results = self.executor.submit(self.recognize_,rgb_small_frame,face_locations,data)
                        self.start_time = time.time()
                    else:
                        elasped_time = time.time() - self.start_time
                        print("elasped_time = ",elasped_time)
                        if (elasped_time % 1)< 0.2:
                            print("1 sec elasped... Check our executor for result!")
                            if not self.recognize_results.running():
                                # Asynchrnous plant detection has been completed. You may retrieve the plant location now
                                # [Beware]: Time has elasped since estimated plantrow mask was passed. Drone would have moved by now
                                #            Adjust for the drone pose change [Rotation & Position] applying odometry changes to the computed bbox
                                self.face_names_chars = self.recognize_results.result()
                    #face_names = self.recognize_(rgb_small_frame,face_locations,data)
                    elaps_recog_time = time.time() - Recognition_stime
                    ## Recognition timing

                    # Display the results
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        if name!= "Unknown":
                            color = (0,255,0)
                            self.tracker.tracked_id = name
                            # init tracker with bbox
                            bbox = (left, top, abs(right-left), abs(bottom-top))
                            self.tracker.init(frame,bbox)
                        else:
                            color = (0,0,255)
                        # Draw a box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        # Draw a label with a name below the face
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX
                        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                frame_iter = frame_iter + 1
            else:
                ## Tracking timing
                Tracking_stime = time.time()
                # Tracking mode
                bbox = self.tracker.track(frame)[1]
                left,top,width,height= bbox
                right = left + width
                bottom = top + height
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,255,0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, self.tracker.tracked_id, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                
                elaps_track_time = time.time() - Tracking_stime
                ## Tracking timing


            elapsed_time = time.time() - start_time
            ideal_processing_time = 33 # ms [Considering 30 fps]
            if elapsed_time < ideal_processing_time:
                waitTime = int(ideal_processing_time - elapsed_time)
            else: # Already too slow --> Wait for minimum possible time
                waitTime = 1

            Profiling_txt = f"[Preprocess,Detection,Recognition,Tracking] = [{elaps_pre_time:.2f},{elaps_det_time:.2f},{elaps_recog_time:.2f},{elaps_track_time:.2f}] secs"
            putText(frame,Profiling_txt, (20,60))
            fps = 1.0 / elapsed_time if elapsed_time!=0 else 100.00
            fps_txt = f"{self.tracker.mode}: ( {self.tracker.tracker_type} ) at {fps:.2f} FPS"
            cv2.putText(frame, fps_txt ,(20,40) ,cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0))
            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            k = cv2.waitKey(waitTime)
            if k==27:#Esc pressed
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    def live_recognition_ufast(self,vid_id = 0,dataset_dir="",embeddings_path=""):
        # This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
        # other example, but it includes some basic performance tweaks to make things run a lot faster:
        #   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
        #   2. Only detect faces in every other frame of video.

        # PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
        # OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
        # specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.
        if embeddings_path!="":
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(embeddings_path, "rb").read())
        elif dataset_dir!="":
            foldername = os.path.basename(dataset_dir)
            embeddings_path = os.path.join(os.path.dirname(__file__),f"models/{foldername}-live_embeddings-face_enc")
            if not os.path.isfile(embeddings_path): # Only generate if not already present
                self.generate_embeddings(dataset_dir,True)
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(self.default_embeddings_path, "rb").read())
        elif os.path.isfile(self.default_embeddings_path):
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(self.default_embeddings_path, "rb").read())
        else:
            print(f"Face-embedddings {self.default_embeddings_path} is not available.\nProvide dataset to generate new embeddings...")
            exit(0)


        # Get a reference to webcam #0 (the default one)
        video_capture = cv2.VideoCapture(vid_id)

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []

        frame_iter = 0

        elaps_pre_time = 0
        elaps_det_time = 0
        elaps_recog_time = 0
        elaps_track_time = 0

        while True:
            start_time = time.time()

            ## Preprocess timing
            preprocess_stime = time.time()
            # Grab a single frame of video
            ret, frame = video_capture.read()
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
            
            #if ( (time.time() - preprocess_stime) > elaps_pre_time ):
            elaps_pre_time = time.time() - preprocess_stime 
            ## Preprocess timing

            if self.tracker.mode=="Detection":
                print(f"Current Frame = {frame_iter}")
                if frame_iter%3==0:
                    print(f"At {frame_iter} frame Detector was applied")

                    ## detection timing
                    detection_stime = time.time()
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    
                    elaps_det_time = time.time() - detection_stime
                    ## detection timing

                    ## Recognition timing
                    Recognition_stime = time.time()

                    # Schedule a function in the executor and returns a future object (results)
                    if ( (self.start_time == None) and (len(face_locations)!=0) ): # Found a face ===> Time to perform recognition on it
                        #face_namess = self.recognize_(rgb_small_frame,face_locations,data)
                        #print("face_namess =  ",face_namess)
                        self.recognize_results = self.executor.submit(self.recognize_,rgb_small_frame,face_locations,data)
                        self.start_time = time.time()
                    elif (self.start_time != None):
                        elasped_time = time.time() - self.start_time
                        print("elasped_time = ",elasped_time)
                        if (elasped_time % 1)< 0.2:
                            print("1 sec elasped... Check our executor for result!")
                            if not self.recognize_results.running():
                                # Asynchrnous plant detection has been completed. You may retrieve the plant location now
                                # [Beware]: Time has elasped since estimated plantrow mask was passed. Drone would have moved by now
                                #            Adjust for the drone pose change [Rotation & Position] applying odometry changes to the computed bbox
                                self.face_names_chars = self.recognize_results.result()
                                if self.face_names_chars[0]!='0':
                                    print("[Special] self.face_names_chars = ",self.face_names_chars)
                                    time.sleep(0)
                                print("self.face_names_chars Received",self.face_names_chars)
                                print("len(self.face_names_chars)",len(self.face_names_chars))

                                face_names_chars_list = list(self.face_names_chars)
                                if '0' in face_names_chars_list:
                                    end_idx = face_names_chars_list.index('0')
                                    face_names_chars_list = face_names_chars_list[0:end_idx]
                                #face_names = [i for i in face_names_chars_list]
                                face_names_str = ''.join(face_names_chars_list)
                                print("face_names_str = ",face_names_str)

                                face_names = face_names_str.split(",")
                                #print("face_names = ",face_names)                                
                                if len(face_names)!=0:
                                    print("face_names = ",face_names)
                                    time.sleep(0)
                                #my_list = my_string.split(",")
                                #face_names = self.recognize_(rgb_small_frame,face_locations,data)
                                elaps_recog_time = time.time() - Recognition_stime
                                ## Recognition timing

                                # Display the results
                                for (top, right, bottom, left), name in zip(face_locations, face_names):
                                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                                    top *= 4
                                    right *= 4
                                    bottom *= 4
                                    left *= 4
                                    if name!= "Unknown":
                                        color = (0,255,0)
                                        self.tracker.tracked_id = name
                                        # init tracker with bbox
                                        bbox = (left, top, abs(right-left), abs(bottom-top))
                                        self.tracker.init(frame,bbox)
                                    else:
                                        color = (0,0,255)
                                    # Draw a box around the face
                                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                                    # Draw a label with a name below the face
                                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                                    font = cv2.FONT_HERSHEY_DUPLEX
                                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                frame_iter = frame_iter + 1
            else:
                ## Tracking timing
                Tracking_stime = time.time()
                # Tracking mode
                bbox = self.tracker.track(frame)[1]
                left,top,width,height= bbox
                right = left + width
                bottom = top + height
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0,255,0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, self.tracker.tracked_id, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                
                elaps_track_time = time.time() - Tracking_stime
                ## Tracking timing


            elapsed_time = time.time() - start_time
            ideal_processing_time = 33 # ms [Considering 30 fps]
            if elapsed_time < ideal_processing_time:
                waitTime = int(ideal_processing_time - elapsed_time)
            else: # Already too slow --> Wait for minimum possible time
                waitTime = 1

            Profiling_txt = f"[Preprocess,Detection,Recognition,Tracking] = [{elaps_pre_time:.2f},{elaps_det_time:.2f},{elaps_recog_time:.2f},{elaps_track_time:.2f}] secs"
            putText(frame,Profiling_txt, (20,60))
            fps = 1.0 / elapsed_time if elapsed_time!=0 else 100.00
            fps_txt = f"{self.tracker.mode}: ( {self.tracker.tracker_type} ) at {fps:.2f} FPS"
            cv2.putText(frame, fps_txt ,(20,40) ,cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0))
            # Display the resulting image
            cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            k = cv2.waitKey(waitTime)
            if k==27:#Esc pressed
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

def secure_access():
    facc_recog = face_recognition_dlib()

    #dataset_dir = r"src/c__Advanced\Face_Recognition\training-data\dlib\avengers_endgame"
    dataset_dir = r"src/c__Advanced\Face_Recognition\training-data\dlib\known_people"

    #test_vid_path = r"Data\NonFree\Friends\Avengers_endgame_Assemble.mp4"
    test_vid_path = r"Data\NonFree\Friends\Jeff_Vs_Elon.mp4"
    facc_recog.live_recognition_ufast(vid_id = test_vid_path, dataset_dir=dataset_dir)


if __name__ =="__main__":
    secure_access()