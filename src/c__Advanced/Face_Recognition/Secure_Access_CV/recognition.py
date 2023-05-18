import cv2
import numpy as np
import os
import pickle
import face_recognition

from imutils import paths
from src.c__Advanced.utilities import to_trbl


class face_recognition_dlib():

    def __init__(self):
        #self.default_embeddings_path = os.path.join(os.getcwd(),"models/live_embeddings-face_enc")
        self.default_embeddings_path = os.path.join(os.path.dirname(__file__),"models/live_embeddings-face_enc")
        self.preprocess_resize = 4 # Downscaling 4 times to improve computation time


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

    def get_embeddings(self,dataset_dir="",embeddings_path=""):
        if embeddings_path!="":
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(embeddings_path, "rb").read())
        elif dataset_dir!="":
            foldername = os.path.basename(dataset_dir)
            embeddings_path = os.path.join(os.path.dirname(__file__),f"models/{foldername}-live_embeddings-face_enc")
            if not os.path.isfile(embeddings_path): # Only generate if not already present
                self.generate_embeddings(dataset_dir,False)
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(embeddings_path, "rb").read())
        elif os.path.isfile(self.default_embeddings_path):
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(self.default_embeddings_path, "rb").read())
        else:
            print(f"Face-embedddings {self.default_embeddings_path} is not available.\nProvide dataset to generate new embeddings...")
            exit(0)
        return data

    def preprocess(self,frame,frame_draw,bboxes=[]):
        downscale = 4
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=(1/downscale), fy=(1/downscale))
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        scale = downscale
        scales = [1/scale]*len(bboxes)
        if len(bboxes)==0:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
        else:
            # Convert found bboxes (ltwh) ===> face locations (ltrd)
            face_locations = list( map(to_trbl,bboxes,scales) )

        # Display the detections
        face_names = list(range(len(face_locations)))
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
            # Draw a box around the face
            cv2.rectangle(frame_draw, (left, top), (right, bottom), (255,0,0), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame_draw, (left, bottom - (9*scale)), (right, bottom), (255,0,0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame_draw, str(name), (left + 6, bottom - (2*scale)), font, 0.25*scale, (255, 255, 255), 1)

        return rgb_small_frame,face_locations
    
    def recognize(self,rgb_small_frame,face_locations,data):

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
            
        if (len(face_names_chars)>=100):
            face_names_chars = face_names_chars[0:100]
        else:
            lst = ['0'] * (100-len(face_names_chars))
            face_names_chars = face_names_chars + lst

        return face_names_chars