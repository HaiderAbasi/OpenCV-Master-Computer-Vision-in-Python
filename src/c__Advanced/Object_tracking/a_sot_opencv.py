import cv2
from src.utilities import putText,get_data,print_h,disp_Fps
import time
from collections import deque


class Tracking():
    def __init__(self,tracker_type = "CSRT"):
        print(f"\n[Tracking]: Created a {tracker_type} tracker!\n")
        self.tracker_type = tracker_type
        self.mode = "Detection"
        self.bbox_clr = None
        self.tracker_initialized = False
        self.tracker_create()
        
    def tracker_create(self):
        if self.tracker_type=="KCF":
            self.tracker = cv2.TrackerKCF_create()
            self.bbox_clr = (255,0,0)
        elif self.tracker_type == "MOSSE":
            self.tracker = cv2.TrackerMOSSE_create()
            self.bbox_clr = (0,255,0)
        elif self.tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
            self.bbox_clr = (0,0,255)
            
    def init(self,frame,bbox):
        # Checking if tracker is already initialized or not
        if self.tracker_initialized:
            #self.tracker.clear() # Crashes without error in OpenCV-Contrib-python 4.4
            del self.tracker # Deleting old tracker
            self.tracker_create() # Recreating the tracker
        
        self.tracker.init(frame,bbox)
        self.mode = "Tracking"
        self.tracker_initialized = True
        
    def track(self,frame):
        ok, bbox = self.tracker.update(frame)
        if not ok:
            self.mode = "Detection"
        # Returning int bbox
        tracked_bbox = (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
        return ok,tracked_bbox

            


def main():
    print_h("[main]: Investigating OpenCV Tracking module.")
    
    trackers_to_test = ["CSRT","MOSSE","KCF"]

    trackers = []
    for tracker in trackers_to_test:
        trackers.append(Tracking(tracker))

    # Use CSRT when you need higher object tracking accuracy and can tolerate slower FPS throughput
    # Use KCF when you need faster FPS throughput but can handle slightly lower object tracking accuracy
    # Use MOSSE when you need pure speed

    # Fetch the videos available for testing Sot Modules
    vid_dirs,filenames = get_data("tracking")
    data_iter = int(input("Please select one from the following:\n"+"\n".join(filenames)+"\n"))

    # Find and set the Window Name the video file name
    Window_Name = filenames[data_iter]
    cv2.namedWindow(Window_Name,cv2.WINDOW_NORMAL)
    # Ensure Window Stays on Top
    cv2.setWindowProperty(Window_Name, cv2.WND_PROP_TOPMOST, 1)



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
        else:
            start_frame_time = time.perf_counter()
            
            # Creating a duplicate of frame for displaying
            frame_disp = frame.copy()
            
            for idx,tracker in enumerate(trackers):
                if tracker.mode =="Tracking":
                    ok,bbox = tracker.track(frame)
                    if ok:
                        x,y,w,h = bbox
                        putText(frame_disp, f"{tracker.tracker_type}",( x , y - (10*idx) ),color=tracker.bbox_clr)
                        cv2.rectangle(frame_disp, ( x , y ),( x + w , y + h),tracker.bbox_clr,3)
                else:
                    putText(frame_disp, f"Detection",(20,(20*(idx+1) + 30)),color=tracker.bbox_clr)
                
            
            disp_Fps(frame_disp,processing_times,start_frame_time)
            
            # Displaying Frame
            cv2.imshow(Window_Name,frame_disp)
            k = cv2.waitKey(30)
            if k == ord('c'):
                cv2.namedWindow("Select ROI",cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Select ROI", cv2.WND_PROP_TOPMOST, 1)
                bbox = cv2.selectROI("Select ROI",frame)
                cv2.destroyWindow("Select ROI")
                for tracker in trackers:
                    tracker.init(frame,bbox)  
            elif k==27:# If Esc Pressed Quit!
                break




if __name__ =="__main__":
    main()