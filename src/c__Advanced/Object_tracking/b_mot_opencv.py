import cv2
from src.utilities import putText,get_data,Gui,print_h,random_bright_color,disp_Fps
import time
from collections import deque


class multitracker:
    # Initialize MultiTracker with specified Sot as a backend
    def __init__(self,tracker_type = "CSRT"):
        self.m_tracker = cv2.MultiTracker_create()
        self.tracker_type = tracker_type
        # Initialize mode and empty lists for tracking classes and colors
        self.mode = "Detection"
        self.tracked_classes = []
        self.colors = []    
        
    # Create a specified tracker
    def tracker_create(self):
        if self.tracker_type == "MOSSE":
            return(cv2.TrackerMOSSE_create())
        elif self.tracker_type == "KCF":
            return(cv2.TrackerKCF_create())
        if self.tracker_type == "CSRT":
            return(cv2.TrackerCSRT_create())
        
    # Initialize multitracker with bboxes, clear previous tracker and colors
    def init(self,frame,bboxes):
        
        if (len(self.m_tracker.getObjects())!= 0):
            del self.m_tracker
            self.m_tracker = cv2.MultiTracker_create()            
            self.colors.clear() 
            self.tracked_classes.clear()
        
        obj_iter = 1
        for bbox in bboxes:
            # check if bbox is valid
            if ( (bbox !=[]) and (bbox[2] != 0) and (bbox[3] != 0) ):
                tracker = self.tracker_create()
                self.m_tracker.add(tracker,frame,bbox)
                self.mode = "Tracking"
                self.tracked_classes.append(f"{obj_iter}")
                self.colors.append(random_bright_color())
                obj_iter = obj_iter +1
                
    # Get updated location of objects in subsequent frames
    def track(self,frame,frame_draw):
        
        tracked_bboxes = []
        
        success, boxes = self.m_tracker.update(frame)
        
        if success:
            for i, rct in enumerate(boxes):
                tracked_bboxes.append((round(rct[0],1),round(rct[1],1),round(rct[2],1),round(rct[3],1)))
                # Draw rectangle and text on frame with class and color
                p1 = (int(rct[0]), int(rct[1]))
                p2 = (int(rct[0] + rct[2]), int(rct[1] + rct[3]))
                cv2.rectangle(frame_draw, p1, p2, self.colors[i], 3, 1)
                cv2.putText(frame_draw,f"{self.tracked_classes[i]}",(p1[0],p1[1]-20),cv2.FONT_HERSHEY_DUPLEX,1,(128,0,255))
        else:
            self.mode = "Detection"
            # Tracking failure
            cv2.putText(frame_draw, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            
        return tracked_bboxes

def demo():
    print_h("[main]: Investigating OpenCV Multi-Tracker module.")
    
    gui = Gui()
    m_tracker = multitracker()
    # Use CSRT when you need higher object tracking accuracy and can tolerate slower FPS throughput
    # Use KCF when you need faster FPS throughput but can handle slightly lower object tracking accuracy
    # Use MOSSE when you need pure speed

    vid_dirs,filenames = get_data("multitracking")

    data_iter = gui.selectdata(filenames)
    Window_Name = filenames[data_iter]
    cv2.namedWindow(Window_Name,cv2.WINDOW_NORMAL)


    #-- 1. Read the video stream
    cap = cv2.VideoCapture(vid_dirs[data_iter])
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)
    
    # Creating a deque for robust fps measurement for sot algorithms
    processing_times = deque(maxlen = 10)
    while True:
        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
            
        frame_disp = frame.copy()
        
        # If the tracker is in tracking mode
        if m_tracker.mode=="Tracking":
            # Start the timer
            start_time = time.perf_counter()
            tracked_bboxes = m_tracker.track(frame,frame_disp)
            disp_Fps(frame_disp,processing_times,start_time)

        cv2.imshow(Window_Name,frame_disp)
        
        k = cv2.waitKey(30)
        # If 'c' is pressed, select the regions of interest
        if k==ord('c'):
            bboxes = gui.selectROIs(frame)
            m_tracker.init(frame,bboxes)
        elif k==27:# If Esc Pressed Quit!
            break


if __name__ =="__main__":
    demo()