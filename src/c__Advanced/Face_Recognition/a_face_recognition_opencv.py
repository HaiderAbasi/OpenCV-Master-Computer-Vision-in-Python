# Face Recognition with OpenCV
import cv2
import os
import numpy as np
from src.c__Advanced.Detection.a_object_detection import detection
from src.c__Advanced.utilities import draw_rectangle, imshow,putText,GUI
import time


class face_recognition_cv:

    def __init__(self,face_recognizer = "lpbh"):
        self.gui = GUI()

        if face_recognizer == "eigen":
            self.recognizer = cv2.face.createEigenFaceRecognizer()
        elif face_recognizer == "fisher":
            self.recognizer = cv2.face.createFisherFaceRecognizer()
        elif face_recognizer == "lpbh":
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        else:
            print(f"Unknown face recognizer specified {face_recognizer}. Exiting....")
            exit(0)

        self.known_indivdiuals = []
        self.__train_dir = r"src/c__Advanced\Face_Recognition\training-data\cv\Friends"
        self.__face_detector = detection() # Creating instance of cascade detector

    def extract_trainingdata(self,vid_dir,sort = False):
        folder_name = os.path.basename(os.path.dirname(vid_dir))
        #-- 1. Read the video stream
        cap = cv2.VideoCapture(vid_dir)
        if not cap.isOpened:
            print('--(!)Error opening video capture')
            exit(0)

        skip_frames = 10
        img_iter = 0
        categories = ["Add new Category....!","Discard!!!"]
        while True:
            img_iter = img_iter+1
            ret, frame = cap.read()
            if frame is None:
                print('--(!) No captured frame -- Break!')
                break
            else:
                if (img_iter%skip_frames)==0:
                    #detect face
                    bboxes = self.__face_detector.detect(frame)
                    for bbox in bboxes:
                        (x,y,w,h) = bbox
                        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))
                        face = gray[y:y+h,x:x+w]

                        imshow("Identify Category",face)
                        cv2.waitKey(1)
                        #------STEP-4--------
                        #for the purpose of this tutorial
                        #we will ignore faces that are not detected
                        if face is not None:
                            category_idx = self.gui.selectdata(categories,useMouse=True,onTop=True,data_type="label")
                            #print("category (Mouse)= ",category_idx)
                            if category_idx==0:
                                category = input("Who's this guy/girl?\n")
                            elif (category_idx == -1):
                                print("Exiting....")
                                break
                            else:
                                category = categories[category_idx]

                            Not_labels = [categories[0],categories[len(categories)-1]]
                            if category not in Not_labels: # If part of a category
                                if category not in categories:
                                    categories.insert(len(categories)-1,category)
                                    #categories.append(category) # index will be identified by category position
                                
                                category_dir = os.path.join(self.__train_dir,folder_name,category)
                                if not os.path.isdir(category_dir):
                                    os.makedirs(category_dir)
                                img_iter = len(os.listdir(category_dir))+1

                                img_name = str(img_iter)+".png"
                                img_path = os.path.join(category_dir,img_name)
                                cv2.imwrite(img_path,face)

                    if (category_idx == -1):
                        break

        cv2.destroyWindow("Identify Category")

    def train(self,train_dir ="",detect_face = True):

        if train_dir!= "": # If train directory manually specified 
            self.__train_dir = train_dir

        #this function will read all persons' training images, detect face from each image
        #and will return two lists of exactly same size, one list 
        # of faces and another list of labels for each face
        def prepare_training_data(train_path):
            #------STEP-1--------
            #get the directories (one directory for each subject) in data folder
            dirs = os.listdir(train_path)
            #list to hold all subject faces
            faces = []
            #list to hold labels for all self.known_indivdiuals
            labels = []
            categories = []

            #let's go through each directory and read images within it
            for dir_name in dirs:
                if dir_name in categories:
                    label = (categories.index(dir_name)+1)
                else:
                    categories.append(dir_name)
                    label = len(categories)-1

                #build path of directory containin images for current subject subject
                #sample subject_dir_path = "train_path/label_name"
                subject_dir_path = train_path + "/" + dir_name
                #get the images names that are inside the given subject directory
                subject_images_names = os.listdir(subject_dir_path)
                #------STEP-3--------
                #go through each image name, read image, 
                #detect face and add face to list of faces
                for image_name in subject_images_names:
                    #ignore system files like .DS_Store
                    if image_name.startswith("."):
                        continue;
                    #build image path
                    #sample image path = train_path/label_name/1.png
                    image_path = subject_dir_path + "/" + image_name
                    #read image
                    image = cv2.imread(image_path)
                    #display an image window to show the image 
                    cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                    cv2.waitKey(100)
                    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    if detect_face:
                        #detect face
                        bboxes = self.__face_detector.detect(image)
                        (x,y,w,h) = bboxes[0]
                        face = gray[y:y+h,x:x+w]
                    else:
                        face = gray
                    #cv2.imshow("face",face)
                    #------STEP-4--------
                    #for the purpose of this tutorial
                    #we will ignore faces that are not detected
                    if face is not None:
                        #add face to list of faces
                        faces.append(face)
                        #add label for this face
                        labels.append(label)
                    
            
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()

            return faces, labels, categories
        

        #data will be in two lists of same size
        #one list will contain all the faces
        #and other list will contain respective labels for each face
        print("Preparing data...")
        faces, labels, categories = prepare_training_data(self.__train_dir)
        print("Data prepared")
        labels_unique = set(labels)
        print("Total faces: ", len(faces))
        print("Total labels: ", len(labels))
        print("Total categories: ", len(categories))
        print("Found categories: ", categories)

        #train our face recognizer of our training faces
        self.recognizer.train(faces, np.array(labels))


        self.known_indivdiuals = categories

    def train_backup(self,train_dir =""):

        if train_dir!= "": # If train directory manually specified 
            self.__train_dir = train_dir

        #this function will read all persons' training images, detect face from each image
        #and will return two lists of exactly same size, one list 
        # of faces and another list of labels for each face
        def prepare_training_data(train_path):
            #------STEP-1--------
            #get the directories (one directory for each subject) in data folder
            dirs = os.listdir(train_path)
            #list to hold all subject faces
            faces = []
            #list to hold labels for all self.known_indivdiuals
            labels = []

            #let's go through each directory and read images within it
            for dir_name in dirs:
                #our subject directories start with letter 's' so
                #ignore any non-relevant directories if any
                if not dir_name.startswith("s"):
                    continue;
                #------STEP-2--------
                #extract label number of subject from dir_name
                #format of dir name = slabel
                #, so removing letter 's' from dir_name will give us label
                label = int(dir_name.replace("s", ""))
                #build path of directory containin images for current subject subject
                #sample subject_dir_path = "training-data/s1"
                subject_dir_path = train_path + "/" + dir_name
                #get the images names that are inside the given subject directory
                subject_images_names = os.listdir(subject_dir_path)
                #------STEP-3--------
                #go through each image name, read image, 
                #detect face and add face to list of faces
                for image_name in subject_images_names:
                    #ignore system files like .DS_Store
                    if image_name.startswith("."):
                        continue;
                    #build image path
                    #sample image path = training-data/s1/1.pgm
                    image_path = subject_dir_path + "/" + image_name
                    #read image
                    image = cv2.imread(image_path)
                    #display an image window to show the image 
                    cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                    cv2.waitKey(100)
                    #detect face
                    bboxes = self.__face_detector.detect(image)
                    (x,y,w,h) = bboxes[0]
                    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    face = gray[y:y+h,x:x+w]
                    #cv2.imshow("face",face)
                    #------STEP-4--------
                    #for the purpose of this tutorial
                    #we will ignore faces that are not detected
                    if face is not None:
                        #add face to list of faces
                        faces.append(face)
                        #add label for this face
                        labels.append(label)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()

            return faces, labels

        #data will be in two lists of same size
        #one list will contain all the faces
        #and other list will contain respective labels for each face
        print("Preparing data...")
        faces, labels = prepare_training_data(self.__train_dir)
        print("Data prepared")
        labels_unique = set(labels)
        print("Total faces: ", len(faces))
        print("Total categories: ", len(labels_unique))

        #train our face recognizer of our training faces
        self.recognizer.train(faces, np.array(labels))


        if len(self.known_indivdiuals)==0:
            self.known_indivdiuals.append("") # zeroth label is no person for our directory structure
            for iter in range(len(labels_unique)):
                face_to_display = faces[labels.index(iter+1)]
                imshow("Face Labeler",face_to_display)
                cv2.waitKey(1)
                self.known_indivdiuals.append(input(f"{iter}: Who's this guy/girl?\n"))
            cv2.destroyWindow("Face Labeler")

    def predict(self,test_img):
        if len(self.known_indivdiuals)==0:
            self.train()
            
        #make a copy of the image as we don't want to chang original image
        img = test_img.copy()
        label_text = -1

        #detect face from the image
        bboxes = self.__face_detector.detect(img)
        if len(bboxes)!=0:
            bbox = bboxes[0]
            (x,y,w,h) =bbox
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            face = gray[y:y+h,x:x+w]
            cv2.imshow("face",face)

            #predict the image using our face recognizer 
            label, confidence = self.recognizer.predict(face)
            #get name of respective label returned by face recognizer
            label_text = self.known_indivdiuals[label]

            #draw a rectangle around face detected
            draw_rectangle(img, bbox)
            #draw name of predicted person
            putText(img, label_text, (bbox[0], bbox[1]-10))
        return img,label_text


def demo():

    face_recognizer = face_recognition_cv()

    train_recognizer = True
    perform_recognition = True

    #load test images
    test_img1 = cv2.imread(r"src/c__Advanced\Face_Recognition\test-data\cv/test1.jpg") # prefix the string with r (to produce a raw string)
    test_img2 = cv2.imread(r"src/c__Advanced\Face_Recognition\test-data\cv/test2.jpg")
    
    if train_recognizer:
        face_recognizer.train()


    if perform_recognition:
        #perform a prediction
        predicted_img1,label_1 = face_recognizer.predict(test_img1)
        predicted_img2,label_2 = face_recognizer.predict(test_img2)
        #display both images
        imshow(label_1, cv2.resize(predicted_img1, (400, 500)),cv2.WINDOW_AUTOSIZE)
        imshow(label_2, cv2.resize(predicted_img2, (400, 500)),cv2.WINDOW_AUTOSIZE)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():

    face_recognizer = face_recognition_cv()

    train_recognizer = True
    perform_recognition = True

    train_video = r"Data\NonFree\Friends\Friends_lightning round.mp4"
    training_data = r"src/c__Advanced\Face_Recognition\training-data/cv\Friends"
    test_video = r"Data\NonFree\Friends\Friends_AllClothes.mp4"

    if train_recognizer:
        #face_recognizer.extract_trainingdata(train_video)
        face_recognizer.train(training_data,False)


    if perform_recognition:

        #-- 1. Read the video stream
        cap = cv2.VideoCapture(test_video)
        if not cap.isOpened:
            print('--(!)Error opening video capture')
            exit(0)
            
        while True:
            ret, frame = cap.read()
            if frame is None:
                print('--(!) No captured frame -- Break!')
                break
            else:
                start_time = time.time()
                predicted_img,label = face_recognizer.predict(frame)
                fps = 1.0 / (time.time() - start_time) if (time.time() - start_time)!=0 else 100.00
                fps_txt = f"FPS: = {fps:.2f}"
                putText(predicted_img, f"{fps_txt}",(20,20))
                #display both images
                imshow("Face Recognition", cv2.resize(predicted_img, (400, 500)),cv2.WINDOW_AUTOSIZE)
                cv2.waitKey(1)
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()