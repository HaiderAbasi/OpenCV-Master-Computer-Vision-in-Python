import face_recognition
import pickle
import cv2
import os
from b_1_generate_embeddings import face_recognition_dlib
from src.c__Advanced.utilities import imshow
from imutils import build_montages

class face_recognition_dlib(face_recognition_dlib):

    def recognize(self,test_path,embeddings_path = ""):
        images = []
        titles = []
        if embeddings_path!="":
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(embeddings_path, "rb").read())
        else:            
            # load the known faces and embeddings saved in last file
            data = pickle.loads(open(self.lastknown_embeddings_path, "rb").read())
        
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
            print(f"encoding.shape = {len(encodings[0])}")
            cv2.waitKey(0)
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
            



def demo():
    data_dir = r"src/c__Advanced\Face_Recognition\training-data\dlib\known_people"
    face_recog = face_recognition_dlib()
    face_recog.generate_embeddings(data_dir)

    test_dir = r"src/c__Advanced\Face_Recognition\test-data\dlib\Guess Who"
    face_recog.recognize(test_dir)


if __name__ =="__main__":
    demo()
