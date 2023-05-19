import cv2
import pandas as pd
import datetime
import os

from src.c__Advanced.Object_recognition.a_face_recognition_opencv import FaceRecognizer
from src.utilities import dataextractor,download_missing_training_data


recog_dir = os.path.dirname(__file__)


class attendance_record:
    def __init__(self,path_to_csv):
        try:
            self.df = pd.read_csv(path_to_csv)
        except FileNotFoundError:
            print("Unable to read csv from disk.... Creating!")
            self.df = pd.DataFrame(columns=['Date','Name','Present?'])
            self.df.to_csv(path_to_csv,index=False)
            
        self.path_to_csv = path_to_csv
        
    def add_daily_entries(self,labels):
        
        today = datetime.datetime.today().strftime("%Y-%m-%d")
        
        if self.df.empty or (self.df['Date']!=today).all():
            print("Preparing attendance sheet for marking...")
            df_list = [self.df]
            for name in labels:
                row = pd.DataFrame([[today,name,'-']],columns=['Date','Name','Present?'])
                df_list.append(row)
            
            self.df = pd.concat(df_list,ignore_index=True)
            self.df.to_csv(self.path_to_csv,index=False)
            
    def mark_attendance(self,pred_ids,pred_confs,labels,max_allowed_dist= 100):
        
            today = datetime.datetime.today().strftime("%Y-%m-%d")
            
            for i in range(len(pred_ids)):
                pred_id = pred_ids[i]
                pred_conf = pred_confs[i]
                
                if pred_conf<max_allowed_dist:
                    
                    for row in range(len(self.df)):
                        if self.df.loc[row,'Date'] == today and self.df.loc[row,'Name'] == labels[pred_id]:
                            self.df.loc[row,'Present?'] = "Present"
                            break
                        
                    self.df.to_csv(self.path_to_csv,index=False)
    
            




def demo(train_dir):
    test_video_path = r"src\c__Advanced\Object_recognition\recog_data\vids\hard_s.mp4"
    path_to_csv = r"src\c__Advanced\Object_recognition\recog_data/attendance.csv"    
        
    face_recog = FaceRecognizer()
    record = attendance_record(path_to_csv)
    
    # Train the face recognizer on the recog_dataset
    face_recog.train(train_dir)
    
    # Open the video file
    cap = cv2.VideoCapture(test_video_path)


    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        exit()
        
    # Loop over each frame in the video
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # Check if the frame was read successfully
        if not ret:
            print("Error reading frame")
            break
        
        # Step 0: Prepare attendance sheet for the day
        record.add_daily_entries(face_recog.labels)
        
        # Step 1: Perform Facial Recognition here
        pred_labels, pred_confs = face_recog.predict_multi(frame,True)
        
        # Step 2: Based on recognition predictions mark attendance
        if pred_labels is not None:
            record.mark_attendance(pred_labels,pred_confs,face_recog.labels)

        # Display the frame
        cv2.imshow("Frame", frame)
        # Wait for a key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video file and close all windows
    cap.release()
    cv2.destroyAllWindows()
    
    


if __name__ == "__main__":
    download_missing_training_data(recog_dir)
    # Create recog_data extractor object for training recog_dataset extraction
    data_ext= dataextractor()
    vid_path = r"src\c__Advanced\Object_recognition\recog_data\vids\hard.mp4" # video to use to create train recog_dataset
    train_dir = r"src\c__Advanced\Object_recognition\recog_data\training\cv\faces" # training recog_dataset for face recognizer
    # Extract faces from video and save to train_directory...
    data_ext.extract(vid_path=vid_path,save_dir=train_dir,skip_frames=15,use_isort=True)
    
    demo(train_dir)