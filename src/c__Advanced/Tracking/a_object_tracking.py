import cv2
from src.a__IP_Basics.utilities import putText,get_data,debugger,get_fileName,print_h
import time



class Tracking:

    def __init__(self,tracker_type = "CSRT"):
        print(f"\n[Tracking]: Created a {tracker_type} tracker!\n")
        self.tracker_type = tracker_type
        self.mode = "Detection"
        self.tracked_id = "Unknown"
        self.tracker_initialized = False
        self.tracker_create()

    def tracker_create(self):
        if self.tracker_type == "MOSSE":
            self.tracker = cv2.TrackerMOSSE_create()
        elif self.tracker_type == "KCF":
            self.tracker = cv2.TrackerKCF_create()
        if self.tracker_type == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()

    def track(self,img):
        # Track obj through the current frame
        ok,bbox = self.tracker.update(img)
        tracked_bbox = [int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])]
        if not ok:
            self.mode = "Detection"
        return ok,tracked_bbox


    def init(self,img,bbox):
        testing = True
        if testing:
            # Testing piece of Code!

            if self.tracker_initialized:
                #self.tracker.clear() # Crashes without error in OpenCV-Contrib-python 4.4
                del self.tracker # Deleting old tracker
                self.tracker_create()
            # Testing piece of Code!
        self.tracker_initialized = True
        self.tracker.init(img,bbox)
        self.mode = "Tracking"


def main():
    print_h("[main]: Investigating OpenCV Tracking module.")

    # Use CSRT when you need higher object tracking accuracy and can tolerate slower FPS throughput
    # Use KCF when you need faster FPS throughput but can handle slightly lower object tracking accuracy
    # Use MOSSE when you need pure speed
    tracker_type = "CSRT"
    tracker = Tracking(tracker_type)


    # Friends = "Data/NonFree\Friends\Friends_AllClothes.mp4"
    # Megamind = "Data/NonFree/Megamind.avi"
    # Window_Name = get_fileName(Megamind)

    vid_dirs,filenames = get_data("tracking")
    data_iter = int(input("Please select one from the following:\n"+"\n".join(filenames)+"\n"))

    Window_Name = filenames[data_iter]

    #debugger = debugger(Window_Name,["data_iter"],[len(filenames)])

    #-- 1. Read the video stream
    cap = cv2.VideoCapture(vid_dirs[data_iter])
    if not cap.isOpened:
        print('--(!)Error opening video capture')
        exit(0)

    while True:
        #debugger.update_variables() # Get updated variables
        #data_iter = debugger.debug_vars[0]

        ret, frame = cap.read()
        if frame is None:
            print('--(!) No captured frame -- Break!')
            break
        else:
            if tracker.mode=="Tracking":
                start_time = time.time()
                ok,bbox = tracker.track(frame)
                if ok:
                    cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),1)
                    fps = 1.0 / (time.time() - start_time) if (time.time() - start_time)!=0 else 100.00
                    fps_txt = f"FPS: = {fps:.2f}"
                    putText(frame, f"Tracking ( {tracker_type} ) at {fps_txt}",(20,20))
                else:
                    tracker.mode = "Detection" # Reset to Detection
            else:
                putText(frame, f"Detection",(20,20))

            cv2.imshow(Window_Name,frame)
            k = cv2.waitKey(30)
            if k==ord('c'):
                bbox = cv2.selectROI("Select ROI",frame)
                print(f"bbox = {bbox}")
                cv2.destroyWindow("Select ROI")
                # Initliaze Tracker
                tracker.init(frame,bbox)
            elif k==27:# If Esc Pressed Quit!
                break


if __name__ =="__main__":
    main()