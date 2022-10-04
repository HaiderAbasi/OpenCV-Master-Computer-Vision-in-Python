from imutils import paths
# Reference : Dlib Installation Without conda : https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f
import face_recognition
import pickle
import cv2
import os
 
class face_recognition_dlib():
    def __init__(self):
        self.lastknown_embeddings_path = ""
        #find path of xml file containing haarcascade file
        cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
        # load the harcaascade in the cascade classifier
        self.faceCascade = cv2.CascadeClassifier(cascPathface)

    def generate_embeddings(self,data_dir):
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
        self.lastknown_embeddings_path = f"{data_dir}-face_enc"
        #use pickle to save data into a file named after the data folder for later use
        f = open(f"{data_dir}-face_enc", "wb")
        f.write(pickle.dumps(data))
        f.close()


def demo():
    data_dir = r"src/c__Advanced\Face_Recognition\training-data\dlib\known_people"
    face_recog = face_recognition_dlib()
    face_recog.generate_embeddings(data_dir)

if __name__ =="__main__":
    demo()   