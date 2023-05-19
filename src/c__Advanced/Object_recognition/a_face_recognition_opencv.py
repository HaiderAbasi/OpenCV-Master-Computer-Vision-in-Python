# Face Recognition with OpenCV
import cv2
import os
import numpy as np
import time

from src.utilities import imshow,putText,Gui,build_montages,download_missing_recog_data,download_missing_training_data
from src.c__Advanced.Object_detection.a_HaarCascades.a_detect_haarCascade import Cascade_Detector

recog_dir = os.path.dirname(__file__)


class FaceRecognizer:
    
    def __init__(self, algorithm_type= "LBPH"):
        download_missing_recog_data(recog_dir)
        self.algorithm_type = algorithm_type
        if algorithm_type == "Eigenfaces":
            self.model = cv2.face.EigenFaceRecognizer_create()
        elif algorithm_type == "Fisherfaces":
            self.model = cv2.face.FisherFaceRecognizer_create()
        elif algorithm_type == "LBPH":
            # cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=DBL_MAX)
            self.model = cv2.face.LBPHFaceRecognizer_create()
            
        self.labels = []
        self.detector = Cascade_Detector()
        
        self.img_s = 100
        
        self.debug = False
        
        # load the landmark predictor
        self.landmark_predictor = cv2.face.createFacemarkLBF()
        self.landmark_predictor.loadModel(r"src\c__Advanced\Object_recognition\recog_data\models\lbfmodel.yaml")

    def train(self, train_dir):
        
        if train_dir=="":
            print(f"\n[Error] Empty Train Directory = {train_dir}/n Recheck please!")
            return
        else:
            print(f"\n[Status] Training {self.algorithm_type} on training recog_dataset at {train_dir}.\n")
        
        images = []
        labels = []
    
        label_list = []  # a list to store unique label names
        
        for label in os.listdir(train_dir):
            label_dir = os.path.join(train_dir, label)
            self.labels.append(label)
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                bboxes = self.detector.detect(gray)
                if len(bboxes)>0:
                    x,y,w,h = bboxes[0]
                    face = gray[y:y+h,x:x+w]
                    face_resized = cv2.resize(face,(self.img_s,self.img_s))
                    images.append(face_resized)
                    if label not in label_list:
                        label_list.append(label)
                    labels.append(label_list.index(label)) # Adding index of each image label to labels 
                    if self.debug:
                        cv2.imshow("Train Image",face_resized)
                        cv2.waitKey(1)
                    
        self.model.train(images, np.array(labels))

    @staticmethod
    def align_face(face,landmarks):
        # Calculate eye centers
        left_eye = landmarks[0][36:42].mean(axis=0)
        right_eye = landmarks[0][42:48].mean(axis=0)

        # Calculate angle between eyes and horizontal axis
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Rotate image
        (h, w) = face.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        face_aligned = cv2.warpAffine(face, M, (w, h), flags=cv2.INTER_CUBIC)
        return face_aligned
     
    def predict(self, test_image):
        
        gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        bboxes = self.detector.detect(gray_image)
        label = -1
        confidence = 0.0
        if len(bboxes)>0:
            x,y,w,h = bboxes[0]
            gray_image = gray_image[y:y+h,x:x+w]
            
            gray_image = cv2.resize(gray_image,(self.img_s,self.img_s))

            label, confidence = self.model.predict(gray_image)
        return label,confidence
    
    # New: function to predict the identity of all people in an image
    def predict_multi(self, test_image, display_faces=False):
        orig_img = test_image.copy()
        gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        bboxes = self.detector.detect(gray_image)
        labels = []
        confidences = []
        if len(bboxes)>0:
            # perform face alignment
            _, landmarks = self.landmark_predictor.fit(orig_img, bboxes)

            for i,bbox in enumerate(bboxes):
                x,y,w,h = bbox
                face = gray_image[y:y+h,x:x+w]
                #face_aligned = cv2.face.createFacemarkKazemi(orig_img, landmarks[i])
                face_aligned = self.align_face(face,landmarks[i])
                face = cv2.resize(face_aligned,(self.img_s,self.img_s))
                label, confidence = self.model.predict(face)
                labels.append(label)
                confidences.append(confidence)
                if display_faces:
                    # New: draw rectangle around detected face
                    cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 165, 255), 2)
                    # New: set text size proportional to the width of the bounding box
                    font_scale = 0.7 * w / 100
                    # New: display label and confidence score
                    text = f"{self.labels[label]} ({confidence:.2f})"
                    cv2.putText(test_image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 165, 255), 2)

        return labels, confidences
    
    
    def identify(self, test_image):
        gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        bboxes = self.detector.detect(gray_image)
        if len(bboxes)>0:
            x,y,w,h = bboxes[0]
            gray_image = gray_image[y:y+h,x:x+w]
            
            gray_image = cv2.resize(gray_image,(self.img_s,self.img_s))
            if self.debug:
                cv2.imshow("Train Image",gray_image)
                cv2.waitKey(1)
            label, confidence = self.model.predict(gray_image)
            label_text = f"Label: {self.labels[label]}"

            putText(test_image, label_text, (10, 20),color = (0,0,0))
            cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            return test_image, self.labels[label]
        else:
            return test_image,"Unknown"





def demo():
    #================================TRAINING=====================================
    # Set the path to the training directory
    train_dir = r"src\c__Advanced\Object_recognition\recog_data\training\cv\Biilionaires"
    
    # Create a FaceRecognizer object
    face_recognizer = FaceRecognizer()
    
    # Train the FaceRecognizer object
    train_recognizer = True
    if train_recognizer:
        face_recognizer.train(train_dir)      
        
    #================================IDENTIFICATIOn=====================================
    # Set the path to the test directory
    test_dir = r"src\c__Advanced\Object_recognition\recog_data\test\cv"
     
    # Initialize empty lists for storing images and titles
    images = []
    titles = []
    
    print(f"\n[Status] Testing {face_recognizer.algorithm_type} on test recog_dataset at {test_dir}.\n")
    # Loop over all files in the directory
    for file_name in os.listdir(test_dir):
        # Check if the file is an image
        if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png')  or file_name.endswith('.webp'):
            # Construct the full path to the image
            file_path = os.path.join(test_dir, file_name)

            # Load test image
            img = cv2.imread(file_path) # prefix the string with r (to produce a raw string)
            
            # Predict the label of the test image using the FaceRecognizer object
            print(f"{face_recognizer.algorithm_type} trained...\nPredicting on test images....")
            start = time.time()
            res_img,label = face_recognizer.identify(img)
            time_elapsed = time.time() - start
            print(f"Time took to predict {time_elapsed}.ms")

            # Append the resulting image and title to the corresponding lists
            images.append(res_img)
            titles.append(label)

    # Display the resulting images with titles as a montage
    montage = build_montages(images, None, None, titles, True, True)
    for montage_img in montage: 
        cv2.imshow("Face Recognition",montage_img)
    cv2.waitKey(0)



if __name__ == "__main__":
    download_missing_training_data(recog_dir)
    demo()